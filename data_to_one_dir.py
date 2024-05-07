import os
import shutil


def run(source_dir, target_dir, ext):
    os.makedirs(target_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        for f in files:
            source_path = os.path.join(root, f)
            filename, file_extension = os.path.splitext(source_path)
            if file_extension != ext:
                continue
            tar_path = os.path.join(target_dir, f)
            shutil.copy(source_path, tar_path)


if __name__ == '__main__':
    source_video_dir = "/media/bossun/新增磁碟區/Datasets/DanceDatasets/DanceDatasets_video"
    target_video_dir = "/media/bossun/新增磁碟區/Datasets/DanceDatasets/DanceDatasets_all_videos"
    run(source_video_dir, target_video_dir, ".mp4")

    source_audio_dir = "/media/bossun/新增磁碟區/Datasets/DanceDatasets/DanceDatasets_audio"
    target_audio_dir = "/media/bossun/新增磁碟區/Datasets/DanceDatasets/DanceDatasets_all_audios"
    run(source_audio_dir, target_audio_dir, ".wav")