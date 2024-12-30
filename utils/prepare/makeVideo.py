#this file will split the pac-nerf datasets into per-frame format
import os
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm
import cv2
from scipy.ndimage import binary_dilation

def make_video(path, image_num, output_name="video.mp4", fps=60, concat_delta=False, resize=1, bg_color="white"):
    render_rgbs = []
    for i in tqdm(range(image_num)):
        if i in [497, 498, 499]:
            continue
        render_rgb = imageio.imread(os.path.join(path, f"{i:05d}.png"))
        if resize > 1:
            new_width = render_rgb.shape[1] // resize
            new_height = render_rgb.shape[0] // resize
            render_rgb = cv2.resize(render_rgb, (new_width, new_height))
        if concat_delta:
            delta_img = imageio.imread(os.path.join(path.replace("/renders", "/deltas"), f"{i:05d}.jpg"))
            render_rgb = np.concatenate([render_rgb, delta_img], axis=1)
        if render_rgb.sum(axis=-1).max() == 255 * 3 and bg_color == "black":
            mask = (render_rgb.sum(axis=-1) >= 700)
            mask = binary_dilation(mask, structure=np.ones((7, 7))).astype(mask.dtype)
            render_rgb[mask, 0] = 0.
            render_rgb[mask, 1] = 0.
            render_rgb[mask, 2] = 0.
        render_rgbs.append(render_rgb)
    render_rgbs = np.stack(render_rgbs, axis=0)
    imageio.mimwrite(os.path.join(path, output_name), render_rgbs, fps=fps)

