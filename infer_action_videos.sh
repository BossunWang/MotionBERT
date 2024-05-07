set -x

VIDEO="/media/bossun/新增磁碟區/Datasets/DanceDatasets/DanceDatasets_all_videos"
AUDIO="/media/bossun/新增磁碟區/Datasets/DanceDatasets/DanceDatasets_all_audios_feats/"
JSON="/media/bossun/新增磁碟區/Datasets/DanceDatasets_video_2d_pose_data"
OUTDIR="/media/bossun/新增磁碟區/Datasets/DanceDatasets_video_action/"
CLIP=243

python infer_action_videos.py \
--vid_dir_path ${VIDEO} \
--audio_feat_dir_path ${AUDIO} \
--json_dir_path ${JSON} \
--out_path ${OUTDIR} \
--clip_len ${CLIP}

#VIDEO="/media/bossun/新增磁碟區/Datasets/Kpop_demo_part_videos/"
#JSON="/media/bossun/新增磁碟區/Datasets/Kpop_demo_part_2d_pose_data/"
#OUTDIR="/media/bossun/新增磁碟區/Datasets/Kpop_demo_part_action/"
#CLIP=243
#
#python infer_action_videos.py \
#--vid_dir_path ${VIDEO} \
#--json_dir_path ${JSON} \
#--out_path ${OUTDIR} \
#--clip_len ${CLIP}