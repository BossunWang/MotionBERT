import os
import json
import re
from tqdm import tqdm


def concated(source_json_dir, target_json_dir):
    os.makedirs(target_json_dir, exist_ok=True)
    json_dict = {}
    pattern = r'^(.+?)(p\d+)\.json'
    for root, dirs, files in os.walk(source_json_dir):
        for f in files:
            matches = re.match(pattern, f)
            if not matches:
                continue
            file_name = matches.group(1)
            if file_name not in json_dict.keys():
                json_dict[file_name] = []
            json_dict[file_name].append(f)

    for k in tqdm(json_dict.keys()):
        json_dict[k].sort()
        data_list = []
        for idx, f in enumerate(json_dict[k]):
            file_path = os.path.join(source_json_dir, f)
            with open(file_path, 'r') as openfile:
                json_object = json.load(openfile)
            for data in json_object:
                data_list.append(data)

        tar_json_file_path = os.path.join(target_json_dir, k + ".json")
        with open(tar_json_file_path, 'w') as f:
            json.dump(data_list, f)


if __name__ == '__main__':
    source_json_dir = "/media/bossun/新增磁碟區/Datasets/DanceDatasets_part_video_2d_pose_data"
    target_json_dir = "/media/bossun/新增磁碟區/Datasets/DanceDatasets_video_2d_pose_data"
    concated(source_json_dir, target_json_dir)