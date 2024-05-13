import os
import numpy as np
import argparse
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset
from lib.utils.vismo import render_and_save

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-j', '--json_dir_path', type=str, help='alphapose detection result json path')
    parser.add_argument('-v', '--vid_dir_path', type=str, help='video path')
    parser.add_argument('-a', '--audio_feat_dir_path', type=str, help='audio feat path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    opts = parser.parse_args()
    return opts

opts = parse_args()
args = get_config(opts.config)

model_backbone = load_backbone(args)
if torch.cuda.is_available():
    model_backbone = nn.DataParallel(model_backbone)
    model_backbone = model_backbone.cuda()

print('Loading checkpoint', opts.evaluate)
checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
model_pos = model_backbone
model_pos.eval()
testloader_params = {
          'batch_size': 1,
          'shuffle': False,
          'num_workers': 0,
          'pin_memory': True,
          'persistent_workers': False,
          'drop_last': False
}

video_files = os.listdir(opts.vid_dir_path)
json_files = os.listdir(opts.json_dir_path)
feat_dir = os.path.join(opts.out_path, "feats")
feat_flip_dir = os.path.join(opts.out_path, "feats_flip")
pose_dir = os.path.join(opts.out_path, "pose")
os.makedirs(feat_dir, exist_ok=True)
os.makedirs(feat_flip_dir, exist_ok=True)
os.makedirs(pose_dir, exist_ok=True)

for json_file in json_files:
    file_name = os.path.splitext(os.path.basename(json_file))[0]
    video_file = os.path.join(opts.vid_dir_path, file_name + ".mp4")
    json_file = os.path.join(opts.json_dir_path, json_file)
    audio_feat_file = os.path.join(opts.audio_feat_dir_path, file_name + ".pkl")
    assert os.path.isfile(audio_feat_file), str(f"{audio_feat_file} is not exist")
    assert os.path.isfile(video_file), str(f"{video_file} is not exist")

    vid = imageio.get_reader(video_file,  'ffmpeg')
    fps_in = vid.get_meta_data()['fps']
    vid_size = vid.get_meta_data()['size']
    frames = vid.count_frames()
    print(f'frames: {frames}')

    with open(audio_feat_file, 'rb') as handle:
        audio_data = pickle.load(handle)
    beat_frames = audio_data['beat_frames']
    beat_unit = audio_data['beat_unit']
    audio_feat = audio_data["feat"]
    beat_interval = (beat_frames[1:] - beat_frames[:-1]).mean() * beat_unit
    beat_interval = np.ceil(beat_interval)
    beat_interval = int(beat_interval)
    assert beat_interval < opts.clip_len, str(f"beat_interval: {beat_interval} over {opts.clip_len}")

    if opts.pixel:
        # Keep relative scale with pixel coornidates
        wild_dataset = WildDetDataset(json_file, clip_len=beat_interval, vid_size=vid_size, scale_range=None, focus=opts.focus)
    else:
        # Scale to [-1,1]
        wild_dataset = WildDetDataset(json_file, clip_len=beat_interval, scale_range=[1,1], focus=opts.focus)

    test_loader = DataLoader(wild_dataset, **testloader_params)

    pose_results_all = []
    feat_results_all = []
    feat_flip_results_all = []

    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            N, T = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]

            predicted_feat = model_pos(batch_input, return_rep=True)

            if args.flip:
                batch_input_flip = flip_data(batch_input)
                predicted_flip_feat = model_pos(batch_input_flip, return_rep=True)
                # print(f"flip feat diff: {abs(predicted_feat - predicted_flip_feat).mean()}")
                feat_flip_results_all.append(predicted_flip_feat.squeeze(0))
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                predicted_3d_pos[:, 0, 0, 2] = 0
                pass
            if args.gt_2d:
                predicted_3d_pos[..., :2] = batch_input[..., :2]

            pose_results_all.append(predicted_3d_pos.squeeze(0))
            feat_results_all.append(predicted_feat.squeeze(0))

    pose_results_all = torch.cat(pose_results_all).cpu().numpy()
    feat_results_all = torch.cat(feat_results_all).cpu().numpy()
    feat_flip_results_all = torch.cat(feat_flip_results_all).cpu().numpy()

    print(f'feats frames: {feat_results_all.shape[0]}')
    if abs(audio_feat.shape[0] - feat_results_all.shape[0]) < 5:
        np.save(f'{pose_dir}/{file_name}.npy', pose_results_all)
        np.save(f'{feat_dir}/{file_name}.npy', feat_results_all)
        np.save(f'{feat_flip_dir}/{file_name}.npy', feat_flip_results_all)
