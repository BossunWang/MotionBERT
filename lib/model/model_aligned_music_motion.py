import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class MotionHead(nn.Module):
    def __init__(self, dim_rep=512, num_joints=17, hidden_dim=256):
        super(MotionHead, self).__init__()
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim * 4)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2, momentum=0.1)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, feat):
        '''
            Input: (N, M, J, C)
        '''
        N, M, J, C = feat.shape
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = self.bn1(feat)
        feat = self.relu1(feat)
        feat = self.fc2(feat)
        feat = self.bn2(feat)
        feat = self.relu2(feat)
        feat = self.fc3(feat)
        return feat


class MusicHead(nn.Module):
    def __init__(self, dim_rep=24, hidden_dim=256):
        super(MusicHead, self).__init__()
        self.fc1 = nn.Linear(dim_rep, hidden_dim * 2)
        self.bn = nn.BatchNorm1d(hidden_dim * 2, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, feat):
        '''
            Input: (N, C)
        '''
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat = self.fc2(feat)
        return feat


class MusicMotionNet(nn.Module):
    def __init__(self, music_dim_rep=24, motion_dim_rep=512, num_joints=17, hidden_dim=256):
        super(MusicMotionNet, self).__init__()
        self.music_head = MusicHead(music_dim_rep, hidden_dim)
        self.motion_head = MotionHead(motion_dim_rep, num_joints, hidden_dim)

    def forward(self, music_feat, motion_feat):
        music_embed = self.music_head(music_feat)
        motion_embed = self.motion_head(motion_feat)
        return music_embed, motion_embed