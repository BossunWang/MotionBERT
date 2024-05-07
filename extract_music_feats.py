import numpy as np
import os
import subprocess
import math
import allin1
from allin1.models import load_pretrained_model
import pickle

TARGET_FPS = 30
FEATURE_FPS = 100


def extract_feats(source_dir, target_dir):
    pretrained_model = load_pretrained_model(
        model_name='harmonix-all',
        device='cuda',
    )

    os.makedirs(target_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        for f in files:
            file_path = os.path.join(root, f)

            extract_data = allin1.analyze(file_path,
                                          include_activations=True,
                                          include_embeddings=True,
                                          pretrained_model=pretrained_model,
                                          )
            allin1_feats = extract_data.embeddings.mean(axis=-1).mean(axis=0)

            # mapping beat frame
            unit_feature_time = 1 / FEATURE_FPS
            feature_time_stamps = np.arange(allin1_feats.shape[0]) * unit_feature_time
            beats = np.array(extract_data.beats)
            beat_frames = np.zeros_like(beats)
            beat_unit = np.max(extract_data.beat_positions)

            for i, beat_time_stamp in enumerate(beats):
                closed_index = np.argmin(abs(feature_time_stamps - beat_time_stamp))
                beat_frames[i] = closed_index

            # mapping closed frame
            unit_feature_time = 1 / FEATURE_FPS
            unit_FPS_time = 1 / TARGET_FPS
            total_sec = allin1_feats.shape[0] / FEATURE_FPS
            feature_time_stamps = np.arange(allin1_feats.shape[0]) * unit_feature_time
            frame_time_stamps = np.arange(math.ceil(total_sec * TARGET_FPS)) * unit_FPS_time
            frame_time_stamp_index = np.zeros_like(frame_time_stamps).astype(np.int32)
            beat_time_stamps = beat_frames * unit_feature_time
            beat_time_stamp_index = np.zeros_like(beat_time_stamps).astype(np.int32)

            for i, frame_time_stamp in enumerate(frame_time_stamps):
                closed_index = np.argmin(abs(feature_time_stamps - frame_time_stamp))
                frame_time_stamp_index[i] = closed_index

            for i, beat_time_stamp in enumerate(beat_time_stamps):
                closed_index = np.argmin(abs(frame_time_stamps - beat_time_stamp))
                beat_time_stamp_index[i] = closed_index

            feats_dict = {}
            feats_dict["feat"] = allin1_feats[frame_time_stamp_index]
            feats_dict["beat_frames"] = beat_time_stamp_index
            feats_dict["beat_unit"] = beat_unit

            fn = os.path.splitext(f)[0]
            target_pkl_path = os.path.join(target_dir, f"{fn}.pkl")

            with open(target_pkl_path, 'wb') as handle:
                pickle.dump(feats_dict, handle)


if __name__ == '__main__':
    source_music_dir = '/media/bossun/新增磁碟區/Datasets/DanceDatasets/DanceDatasets_all_audios'
    target_music_dir = '/media/bossun/新增磁碟區/Datasets/DanceDatasets/DanceDatasets_all_audios_feats'

    extract_feats(source_music_dir, target_music_dir)