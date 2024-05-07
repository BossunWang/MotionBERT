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
from lib.model.model_action import ActionNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/action/MB_train_NTU60_xsub.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/action/MB_train_NTU60_xsub/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
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
model = ActionNet(backbone=model_backbone,
                  dim_rep=args.dim_rep,
                  num_classes=args.action_classes,
                  dropout_ratio=args.dropout_ratio,
                  version=args.model_version,
                  hidden_dim=args.hidden_dim,
                  num_joints=args.num_joints)
model.eval()
if torch.cuda.is_available():
    model = nn.DataParallel(model)
    model = model.cuda()
print('Loading checkpoint', opts.evaluate)
checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['model'], strict=True)

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
feat_dir = os.path.join(opts.out_path, "action")
cls_dir = os.path.join(opts.out_path, "class")
os.makedirs(feat_dir, exist_ok=True)
os.makedirs(cls_dir, exist_ok=True)

for json_file in json_files:
    file_name = os.path.splitext(os.path.basename(json_file))[0]
    video_file = os.path.join(opts.vid_dir_path, file_name + ".mp4")
    json_file = os.path.join(opts.json_dir_path, json_file)
    audio_feat_file = os.path.join(opts.audio_feat_dir_path, file_name + ".pkl")
    assert os.path.isfile(video_file), str(f"{video_file} is not exist")
    assert os.path.isfile(audio_feat_file), str(f"{audio_feat_file} is not exist")
    print(f'file_name: {file_name}')

    vid = imageio.get_reader(video_file,  'ffmpeg')
    fps_in = vid.get_meta_data()['fps']
    vid_size = vid.get_meta_data()['size']
    frames = vid.count_frames()
    print(f'vid_size: {vid_size}')
    print(f'frames: {frames}')

    with open(audio_feat_file, 'rb') as handle:
        audio_data = pickle.load(handle)
    beat_frames = audio_data['beat_frames']
    beat_unit = audio_data['beat_unit']
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

    feat_results_all = []
    cls_results_all = []

    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            N, T = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()

            predicted_cls, predicted_feat = model(batch_input.unsqueeze(0), return_rep=True)
            predicted_feat = predicted_feat.squeeze(0).expand(T, predicted_feat.shape[-1])
            predicted_cls = predicted_cls.squeeze(0).expand(T, predicted_cls.shape[-1])
            # print(f'batch_input: {batch_input.shape}')
            # print(f'predicted_feat: {predicted_feat.shape}')
            # print(f'predicted_cls: {predicted_cls.shape}')

            feat_results_all.append(predicted_feat)
            cls_results_all.append(predicted_cls)

    feat_results_all = torch.cat(feat_results_all).cpu().numpy()
    cls_results_all = torch.cat(cls_results_all).cpu().numpy()

    print(f'feats frames: {feat_results_all.shape[0]}')
    if abs(frames - feat_results_all.shape[0]) < 5:
        np.save(f'{feat_dir}/{file_name}.npy', feat_results_all)
        np.save(f'{cls_dir}/{file_name}.npy', cls_results_all)
