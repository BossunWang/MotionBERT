set -x

CHECKPOINT="checkpoint/music_motion"
#AUDIO="/media/bossun/新增磁碟區/Datasets/DanceDatasets/DanceDatasets_all_audios_feats/"
#MOTION="/media/bossun/新增磁碟區/Datasets/DanceDatasets_video_3d_pose/feats/"
#MOTION_FLIP="/media/bossun/新增磁碟區/Datasets/DanceDatasets_video_3d_pose/feats_flip/"
#CACHED="/media/bossun/新增磁碟區/Datasets/DanceDatasets_video_3d_pose/feats_slice/"

AUDIO="/workspace/Datasets/DanceDatasets/DanceDatasets_all_audios_feats/"
MOTION="/workspace/Datasets/DanceDatasets_video_3d_pose/feats/"
MOTION_FLIP="/workspace/Datasets/DanceDatasets_video_3d_pose/feats_flip/"
CACHED="/workspace/Datasets/DanceDatasets_video_3d_pose/feats_slice/"

python train_music_motion_aligned.py \
--checkpoint ${CHECKPOINT} \
--audio_feat_dir_path ${AUDIO} \
--motion_feat_dir_path ${MOTION} \
--motion_feat_flip_dir_path ${MOTION_FLIP} \
--cached_data_path ${CACHED} \
--use_cached