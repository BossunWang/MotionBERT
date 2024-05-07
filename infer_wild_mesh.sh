set -x

VIDEO="/media/bossun/新增磁碟區/Datasets/Kpop_mesh_test/OneSparkp1.mp4"
JSON="/media/bossun/新增磁碟區/Datasets/Kpop_mesh_test/OneSparkp1.json"
OUTDIR="/media/bossun/新增磁碟區/Datasets/Kpop_mesh_test/pose"
CLIP=243

python infer_wild_mesh.py \
--vid_path ${VIDEO} \
--json_path ${JSON} \
--out_path ${OUTDIR} \
--clip_len ${CLIP}