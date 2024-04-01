import os
import numpy as np
from tqdm import tqdm
import imageio
from PIL import Image
import io
# import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pixel2world_vis_motion(motion, dim=2):
#     pose: (17,2,N)
    N = motion.shape[-1]
    if dim==2:
        offset = np.ones([2,N]).astype(np.float32)
    else:
        offset = np.ones([3,N]).astype(np.float32)
        offset[2,:] = 0
    return (motion + offset) * 512 / 2


def get_img_from_fig(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = np.array(Image.open(io.BytesIO(img_arr)))
    buf.close()
    return img


def motion2video_3d(motion, save_path, fps=25, keep_imgs=False):
    #     motion: (17,3,N)
    videowriter = imageio.get_writer(save_path, fps=fps)
    vlen = motion.shape[-1]
    save_name = save_path.split('.')[0]
    frames = []
    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10],
                   [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]

    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"
    for f in tqdm(range(vlen)):
        j3d = motion[:, :, f]
        fig = plt.figure(0, figsize=(10, 10))
        ax = plt.axes(projection="3d")
        ax.set_xlim(-512, 0)
        ax.set_ylim(-256, 256)
        ax.set_zlim(-512, 0)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        ax.view_init(elev=12., azim=80)
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        for i in range(len(joint_pairs)):
            limb = joint_pairs[i]
            xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
            if joint_pairs[i] in joint_pairs_left:
                ax.plot(-xs, -zs, -ys, color=color_left, lw=3, marker='o', markerfacecolor='w', markersize=3,
                        markeredgewidth=2)  # axis transformation for visualization
            elif joint_pairs[i] in joint_pairs_right:
                ax.plot(-xs, -zs, -ys, color=color_right, lw=3, marker='o', markerfacecolor='w', markersize=3,
                        markeredgewidth=2)  # axis transformation for visualization
            else:
                ax.plot(-xs, -zs, -ys, color=color_mid, lw=3, marker='o', markerfacecolor='w', markersize=3,
                        markeredgewidth=2)  # axis transformation for visualization
        frame_vis = get_img_from_fig(fig)
        videowriter.append_data(frame_vis)
        plt.close()
    videowriter.close()


def main():
    source_dir = "/media/bossun/新增磁碟區/Datasets/DanceDatasets_part_video_3d_pose/pose"
    target_dir = "/media/bossun/新增磁碟區/Datasets/DanceDatasets_part_video_3d_pose/pose_videos"

    os.makedirs(target_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        for f in files:
            results_all = np.load(os.path.join(root, f))
            print(results_all.shape)
            motion = np.transpose(results_all, (1, 2, 0))  # (T,17,D) -> (17,D,T)
            motion_world = pixel2world_vis_motion(motion, dim=3)
            save_path = f'{target_dir}/{f.replace(".npy", ".mp4")}'
            motion2video_3d(motion_world, save_path=save_path, keep_imgs=False, fps=30)
            

if __name__ == '__main__':
    main()