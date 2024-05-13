import os
import numpy as np
import time
import sys
import argparse
import errno
from collections import OrderedDict
import tensorboardX
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from info_nce import InfoNCE, info_nce
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_action import NTURGBD
from lib.model.model_aligned_music_motion import MusicMotionNet

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('-a', '--audio_feat_dir_path', type=str, help='audio feat path')
    parser.add_argument('-m', '--motion_feat_dir_path', type=str, help='motion feat path')
    parser.add_argument('-mf', '--motion_feat_flip_dir_path', type=str, help='motion feat flip path')
    parser.add_argument('-ca', '--cached_data_path', type=str, help='slice data path')
    parser.add_argument('-uc', '--use_cached', action='store_true', help='used cached files')
    parser.add_argument('-e', '--eval', action='store_true', help='evaluate model')
    opts = parser.parse_args()
    return opts


def analysis(data_list, model):
    model.eval()

    audio_embed_list = []
    motion_embed_list = []

    for data_path in tqdm(data_list):
        with open(data_path, 'rb') as handle:
            data = pickle.load(handle)

            motion_feat = torch.from_numpy(data["motion_feat"]).unsqueeze(0).cuda()  # (2, T, 17, 512)
            audio_feat = torch.from_numpy(data["audio_feat"]).unsqueeze(0).cuda()  # (T, 24)

            audio_embed, motion_embed = model(audio_feat, motion_feat)
            audio_embed_list.append(audio_embed.squeeze(0).detach().cpu().numpy())
            motion_embed_list.append(motion_embed.squeeze(0).detach().cpu().numpy())

    audio_embed_list = np.array(audio_embed_list)
    motion_embed_list = np.array(motion_embed_list)
    np.save("audio_embed_features.npy", audio_embed_list)
    np.save("motion_embed_features.npy", motion_embed_list)

    audio_embed_list = np.load("audio_embed_features.npy")
    motion_embed_list = np.load("motion_embed_features.npy")

    audio_motion_embed = np.concatenate([audio_embed_list, motion_embed_list])
    print(audio_motion_embed.shape)

    audio_embed_tsne = TSNE(n_components=2).fit_transform(audio_embed_list)
    motion_embed_tsne = TSNE(n_components=2).fit_transform(motion_embed_list)
    audio_motion_embed_tsne = TSNE(n_components=2).fit_transform(audio_motion_embed)

    plt.figure("audio_embed_tsne")
    plt.scatter(audio_embed_tsne[:, 0], audio_embed_tsne[:, 1])

    plt.figure("motion_embed_tsne")
    plt.scatter(motion_embed_tsne[:, 0], motion_embed_tsne[:, 1])

    plt.figure("audio_motion_embed_tsne")
    plt.scatter(audio_motion_embed_tsne[:audio_embed_list.shape[0], 0],
                audio_motion_embed_tsne[:audio_embed_list.shape[0], 1],
                alpha=0.3)
    plt.scatter(audio_motion_embed_tsne[audio_embed_list.shape[0]:, 0],
                audio_motion_embed_tsne[audio_embed_list.shape[0]:, 1],
                alpha=0.3)
    plt.show()


def train(data_list, batch_size, epochs, model, optimizer, scheduler, criterion, train_writer):
    # Training
    for epoch in range(epochs):
        print('Training epoch %d.' % epoch)
        losses_audio = AverageMeter()
        losses_motion = AverageMeter()

        rand_list = np.arange(len(data_list))
        np.random.shuffle(rand_list)
        slice_size = len(rand_list) // batch_size
        random_idx = rand_list[:slice_size * batch_size].reshape(slice_size, batch_size)

        model.train()
        for idx, (batch_idx) in tqdm(enumerate(random_idx)):
            # positive pairs
            audio_pos_feat_list = []
            motion_pos_feat_list = []
            for ri in batch_idx:
                with open(data_list[ri], 'rb') as handle:
                    data = pickle.load(handle)

                motion_feat = torch.from_numpy(data["motion_feat"]).cuda()  # (2, T, 17, 512)
                audio_feat = torch.from_numpy(data["audio_feat"]).cuda()  # (T, 24)
                audio_pos_feat_list.append(audio_feat.unsqueeze(0))
                motion_pos_feat_list.append(motion_feat.unsqueeze(0))

            negative_list = np.array(list(set(rand_list) - set(batch_idx)))
            np.random.shuffle(negative_list)
            random_neg_idx = negative_list[:batch_size * 2]

            # negative pairs
            audio_neg_feat_list = []
            motion_neg_feat_list = []
            for ri in random_neg_idx:
                with open(data_list[ri], 'rb') as handle:
                    data = pickle.load(handle)

                motion_feat = torch.from_numpy(data["motion_feat"]).cuda()  # (2, T, 17, 512)
                audio_feat = torch.from_numpy(data["audio_feat"]).cuda()  # (T, 24)
                audio_neg_feat_list.append(audio_feat.unsqueeze(0))
                motion_neg_feat_list.append(motion_feat.unsqueeze(0))

            audio_pos_feats = torch.cat(audio_pos_feat_list)
            motion_pos_feats = torch.cat(motion_pos_feat_list)
            audio_neg_feats = torch.cat(audio_neg_feat_list)
            motion_neg_feats = torch.cat(motion_neg_feat_list)

            audio_pos_embed, motion_pos_embed = model(audio_pos_feats, motion_pos_feats)
            audio_neg_embed, motion_neg_embed = model(audio_neg_feats, motion_neg_feats)
            optimizer.zero_grad()
            loss_audio = criterion(audio_pos_embed, motion_pos_embed.detach(), motion_neg_embed.detach())
            loss_motion = criterion(motion_pos_embed, audio_pos_embed.detach(), audio_neg_embed.detach())
            loss_train = loss_audio + loss_motion

            losses_audio.update(loss_audio.item(), batch_size)
            losses_motion.update(loss_motion.item(), batch_size)

            loss_train.backward()
            optimizer.step()

            if (idx + 1) % (slice_size // 2) == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'loss audio {loss_audio.val:.3f} ({loss_audio.avg:.3f})\t'
                      'loss motion {loss_motion.val:.3f} ({loss_motion.avg:.3f})\t'.format(
                    epoch, idx + 1, slice_size, loss_audio=losses_audio, loss_motion=losses_motion))
                sys.stdout.flush()
        train_writer.add_scalar('train_audio_loss', losses_audio.avg, epoch + 1)
        train_writer.add_scalar('train_motion_loss', losses_motion.avg, epoch + 1)

        scheduler.step()

        # Save latest checkpoint.
        chk_path = os.path.join(opts.checkpoint, 'latest_epoch.pt')
        print('Saving checkpoint to', chk_path)
        torch.save({
            'epoch': epoch + 1,
            'lr': scheduler.get_last_lr(),
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
        }, chk_path)


def main(opts):
    os.makedirs(opts.checkpoint, exist_ok=True)
    os.makedirs(opts.cached_data_path, exist_ok=True)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))

    model = MusicMotionNet(music_dim_rep=24, motion_dim_rep=512, num_joints=17, hidden_dim=256)
    criterion = InfoNCE(negative_mode='unpaired')  # negative_mode='unpaired' is the default value
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    epochs = 1500
    batch_size = 1024
    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    scheduler = StepLR(optimizer, step_size=epochs // 2, gamma=0.1)

    if opts.eval:
        chk_path = os.path.join(opts.checkpoint, 'latest_epoch.pt')
        checkpoint = torch.load(chk_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)

    if not opts.use_cached:
        # preprocess dataset
        data_list = []
        motion_feat_files = os.listdir(opts.motion_feat_dir_path)
        for motion_file_name in tqdm(motion_feat_files):
            file_name = os.path.splitext(os.path.basename(motion_file_name))[0]
            motion_file = os.path.join(opts.motion_feat_dir_path, motion_file_name)
            motion_flip_file = os.path.join(opts.motion_feat_flip_dir_path, motion_file_name)
            audio_feat_file = os.path.join(opts.audio_feat_dir_path, file_name + ".pkl")

            motion_feat = np.load(motion_file)
            motion_flip_feat = np.load(motion_flip_file)

            with open(audio_feat_file, 'rb') as handle:
                audio_data = pickle.load(handle)
                beat_frames = audio_data['beat_frames']
                beat_unit = audio_data['beat_unit']
                audio_feat = audio_data["feat"]
                beat_interval = (beat_frames[1:] - beat_frames[:-1]).mean() * beat_unit
                beat_interval = np.ceil(beat_interval)
                beat_interval = int(beat_interval)
                sequence_len = beat_interval

            if motion_feat.shape[0] > audio_feat.shape[0]:
                motion_feat = motion_feat[:audio_feat.shape[0]]
            else:
                audio_feat = audio_feat[:motion_feat.shape[0]]

            T, J, C = motion_feat.shape
            _, M = audio_feat.shape
            sequence_size = T // sequence_len
            motion_feat \
                = motion_feat[:sequence_size * sequence_len].reshape(sequence_size, sequence_len, J, C)[:, None, :, :, :]
            motion_flip_feat \
                = motion_flip_feat[:sequence_size * sequence_len].reshape(sequence_size, sequence_len, J, C)[:, None, :, :, :]
            motion_feat = np.concatenate([motion_feat, motion_flip_feat], axis=1)
            audio_feat = audio_feat[:sequence_size * sequence_len].reshape(sequence_size, sequence_len, M)

            motion_feat = motion_feat.mean(axis=2)
            audio_feat = audio_feat.mean(axis=1)

            for s in range(sequence_size):
                data_dict = {"motion_feat": motion_feat[s], "audio_feat": audio_feat[s]}
                data_path = os.path.join(opts.cached_data_path, f'{file_name}_{s}.pkl')
                data_list.append(data_path)
                with open(data_path, 'wb') as handle:
                    pickle.dump(data_dict, handle)
    else:
        cached_file_names = os.listdir(opts.cached_data_path)
        data_list = [os.path.join(opts.cached_data_path, fn) for fn in cached_file_names]

    if not opts.eval:
        train(data_list, batch_size, epochs, model, optimizer, scheduler, criterion, train_writer)
    else:
        analysis(data_list, model)
    

if __name__ == "__main__":
    opts = parse_args()
    main(opts)