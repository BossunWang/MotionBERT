set -x

VIDEO="/media/bossun/新增磁碟區/Datasets/DanceDatasets/DanceDatasets_video/delight/delight_006_20231017_0#0-32#0.mp4"
JSON="/home/bossun/Projects/3rd_party/AlphaPose/examples/res_video/data.json"
OUTDIR="DanceDatasets_output/delight_006_20231017_0#0-32#0"
CLIP=243

python infer_wild.py \
--vid_path ${VIDEO} \
--json_path ${JSON} \
--out_path ${OUTDIR} \
--clip_len ${CLIP}