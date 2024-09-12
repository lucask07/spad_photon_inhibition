""" Create Bernoulli sampled frames 

Open images and create a set of Bernoulli sampled frames that are saved
to a numpy file (using np.save).
File name is created from the original source image, threshold, and maximum frames

calls create_bern_frames & bernoulli_lookup from bernoulli_inhibit.py

Configuration info is saved to JSON:
    thresh: the intensity threshold for 0/1 decision. Lower threshold is equivalent to a longer exposure time.
    roi_x: tuple of the bounding box
    roi_y: tuple of the bounding box
    bernoulli_rand_steps: bernoulli sampling is done using a look-up table to improve speed
                          this is the number of finite steps in that lookup table
    max_frames: number of bernoulli frames
                note that 1000 frames creates a 92.9 MB output file (a Bool consumes 1 byte!)
    seed: for the random number generator

Lucas J. Koerner, koerner.lucas@stthomas.edu
Aug 2022
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import sys
import json
import copy
from datetime import datetime  # Current date time in local system
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
from bernoulli_inhibit import bernoulli_lookup, create_bern_frames, determine_masks, obj_function
from utils import disp_img, create_img

np.seterr(divide='ignore') # prevent warnings when calculating divide by zero and log(0)
np.seterr(invalid='ignore')

plt.ion()

def rev_gamma(srgb):
    """ 
    reverse gamma of an sRGB image
    so that the result is linear in intensity

    See https://en.wikipedia.org/wiki/SRGB
    """
    if srgb<0.040449936:
        return srgb/12.92
    else:
        # Note 
        return ((srgb+0.055)/1.055)**(1/0.416666666) # 1/0.4166666 = 2.4

# generate a vecotrized version of the reverse gamma function
rev_gamma_mat = np.vectorize(rev_gamma)


def open_img(file_name, dir_name, roi_x=(128,232), roi_y=(128,400), Thumbnail=True):

    # -------- Load HDR image ------------
    # IMREAD_ANYDEPTH is needed because even though the data is stored in 8-bit channels
    # when it's read into memory it's represented at a higher bit depth
    img = cv2.imread(os.path.join(dir_name, file_name), flags=cv2.IMREAD_UNCHANGED)

    if len(np.shape(img)) > 2: # if 3 channels convert to YUV
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)
    else:
        y = img
    if ('.jpg' in file_name) or ('.jpeg' in file_name):
        y = rev_gamma_mat(y)

    # thumbnail
    if Thumbnail:
        thumb_x = roi_x[0]  # size of the thumbnail
        thumb_y = roi_y[0]  # size of the thumbnail
        x_start = roi_x[1]
        y_start = roi_y[1]
        y = y[ x_start:(x_start+thumb_x), y_start:(y_start+thumb_y)]

    y = y/np.max(y) # normalize the image to the range of [0,1]

    # TODO: is there a reason for returning the roi_x, roi_y? 
    return y, roi_x, roi_y

cfg = {}
cfg['bernoulli_rand_steps'] = 4096
cfg['max_frames'] = 1000
cfg['seed'] = 10


def main(figure_dir='/home/lkoerner/lkoerner/contour/BSDS300/images/test/',
         img_file_range = None,
         thresholds=[(None, 1.0)], 
	 extension='.exr',
     max_imgs=20):

    img_names = [f for f in os.listdir(figure_dir) if (os.path.isfile(os.path.join(figure_dir, f)) and f.endswith(extension))]
    print(img_names)
    rv_arrays, rv_idx, rv_dict_len = bernoulli_lookup(num_steps=cfg['bernoulli_rand_steps'], seed=cfg['seed'])

    img_num = 0

    if img_file_range is None:
        imgs_to_process = img_names[0:max_imgs]
    elif isinstance(img_file_range, str):
        imgs_to_process = [img_file_range]
    else:
        imgs_to_process = img_names[img_file_range[0]:img_file_range[1]]
    
    total_imgs = len(imgs_to_process)*len(thresholds)
    saved_image_names = []
    for img_name in imgs_to_process:  # todo allow this to be a sys.argv
        print(f'image name: {img_name}, processing {cfg["max_frames"]} frames')
        print(f'Img number {img_num} of {total_imgs}')
        # reset the random variable index
 
        y, roi_x, roi_y = open_img(img_name, figure_dir, roi_x=(128,600), roi_y=(128,600), Thumbnail=False)
        cfg['roi_x'] = roi_x
        cfg['roi_y'] = roi_y

        for th in thresholds:
            for k in rv_idx:
                rv_idx[k] = 0
            frame, acc_counts, y_prob_full_precision = create_bern_frames(y, rv_arrays, rv_idx, rv_dict_len,
                                                                        num_frames=cfg['max_frames'], 
                                                                        threshold_ranges=[th])
            cfg['thresh'] = th

            shape_3d = tuple(list(frame[0].shape) + [len(frame) + 1]) 

            frames3d = np.zeros(shape_3d, dtype=np.bool_) # convert to a 3d numpy matrix. frames is a dictionary of 2d matrices

            for frame_idx in range(len(frame)):
                frames3d[:,:,frame_idx] = frame[frame_idx]

            img_filename = img_name.replace('.png', '').replace('.jpg', '').replace('.exr','') + f'_bernoulli_thresh{cfg["thresh"]}' + f'_maxframes{cfg["max_frames"]}'
            img_filename = img_filename.replace(' ', '_') # replace spaces with underscores. from (None, val) in threshold

            saved_image = os.path.join(figure_dir, img_filename)
            print(f'Saving image {saved_image}')
            np.save(saved_image, frames3d)    # .npy extension is added if not given

            json_file = os.path.join(figure_dir, f'{img_filename}.json')
            with open(json_file, 'w') as fp:
                json.dump(cfg, fp)

            img_num = img_num + 1
            saved_image_names.append(saved_image)

    return saved_image_names


if __name__ == '__main__':

    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(figure_dir=sys.argv[1])
    elif len(sys.argv) == 3:
        main(figure_dir=sys.argv[1], thresholds=[(None, float(sys.argv[2]))])
    elif len(sys.argv) == 4: # allows for specification of a specific image
        main(figure_dir=sys.argv[1], img_file_range=sys.argv[3], thresholds=[(None, float(sys.argv[2]))])
    elif len(sys.argv) == 5: # allows for specification of the number of images
        main(figure_dir=sys.argv[1], img_file_range=None, thresholds=[(None, float(sys.argv[2]))], max_imgs=int(sys.argv[4]))
