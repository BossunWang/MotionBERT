set -x

VIDEO="/media/bossun/新增磁碟區/Datasets/DanceDatasets/DanceDatasets_part_video"
JSON="/home/bossun/Projects/3rd_party/AlphaPose/DanceDatasets_part_video_2d_pose_data"
OUTDIR="/media/bossun/新增磁碟區/Datasets/DanceDatasets_part_video_3d_pose/"
CLIP=243

python infer_wild_videos.py \
--vid_dir_path ${VIDEO} \
--json_dir_path ${JSON} \
--out_path ${OUTDIR} \
--clip_len ${CLIP}