"""
Lucas J. Koerner, koerner.lucas@stthomas.edu
May 2022

Create a reference image with edges and apply inhibition protocols 
to determine impact on confidence in edge detection or other vision task.
Assess the quality of the gradient via SNR 

F[n](x,y) -- frame n 
M[n](x,y) -- mask n 
R[n](x,y) -- rejected n 

Start with a single exposure time when creating binary images 

"""
import os
import sys
import copy
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import pandas as pd
import imageio
import platform
from scipy.stats import bernoulli
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d
from scipy.optimize import minimize, differential_evolution
import cv2
from utils import kernels_from_x, my_savefig
# https://www.mathworks.com/help/matlab/matlab_external/python-setup-script-to-install-matlab-engine-api.html -- installed using the setup.py file within MATLAB directory 

if platform.system() == 'Linux': # MATLAB engine functions
    try:
        from segmentation.python_fscore import eval_fscore, eval_fscore_hdr     
    except:
        def eval_fscore(x):
            return x
        eval_fscore_hdr = eval_fscore

from load_config import config

plt.rcParams['animation.ffmpeg_path'] = config['ffmpeg_path'] 

np.seterr(divide='ignore') # prevent warnings when calculating divide by zero and log(0)
np.seterr(invalid='ignore')

plt.ion()
# directory for source images 
hdr_img_dir = config['hdr_img_dir']

# figure_dir is the output directory of saved plots 
figure_dir = config['figure_dir']

# location for output data 
data_dir = config['data_dir']

SAVE_FIGS = True
PLT_FIGS = True
OPTIMIZE_KERNEL = False

# for bernoulli lookup table 
num_steps = 4096 # of discrete probabilities. Mapped to 0 - 1 later


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array looks like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.

    See here: https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays/16873755#16873755
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def bernoulli_sample(rate):
    """ draw a bernoulli sample 
    given that the probability is 1-exp(-rate)"""

    return bernoulli.rvs(1-np.exp(-rate))

def img_probability(value):
    """ take the value in an image pixel (lambda*T_{exp})
    and map to a probability. 
    Do so with rounding (via int16) so that there is a finite number 
    of options
    """
    return np.int16((1-np.exp(-value))*num_steps) 

def prob_to_rate(img, frames):
    """ take the average counts in an image  
        and convert to a rate image 
        must handle an average of 0 
        must handle an average of 1 
    """
    rate = -np.log(1-img)
    max_rate = -np.log(1- (2*frames-1)/2*frames)
    min_rate = 0
    rate[rate==np.inf] = max_rate
    rate[rate==-np.inf] = 0

    return np.int16((1-np.exp(-value))*num_steps) 


def inhibit_policy(frame_bool, mask, kernel):
    """
    considers a single frame and single mask (snapshot in time), 
    and applies a kernel to return a new mask (that is not yet processed across time or thresholded)
    so this only uses spatial information. 

    frame_bool : boolean frame (photon = 1, no photon = 0)
    mask : if mask = 1 pixel is inhibited  
    kernel: np.ndarray (for a linear operation)
            or string: for a non-linear operation that includes thresholding, etc. 

    """
    frame = frame_bool.astype(int)

    if type(kernel) == np.ndarray:
        # change to photon = 1, no photon = -1
        frame[frame == 0] = -1
        if mask is not None:
            frame = frame*(1-mask)
        new_mask = convolve2d(frame, kernel, mode='same')
    elif kernel == 'no_nophotons': # all ones of non-masked 
        k = np.array([[1,1,1],
                     [1,1,1],
                     [1,1,1]])
        frame[frame == 0] = -1
        frame[frame == 1] = 0        
        frame = frame*(1-mask)
        new_mask = convolve2d(frame, k, mode='same')
        new_mask = (new_mask>=0).astype(int)

    elif kernel == 'all_ones': 
        k = np.array([[1,1,1],
             [1,1,1],
             [1,1,1]])
        frame[frame == 0] = -1
        frame = frame*(1-mask)
        new_mask = convolve2d(frame, k, mode='same')
        new_mask = (new_mask==9).astype(int)

    # logical mask 
    elif kernel == 'no_v_discrepancy': 
        k = np.array([[1,1,1],
                     [0,0,0],
                     [-1,-1,-1]])
        frame[frame == 0] = -1
        frame = frame*(1-mask)
        new_mask = convolve2d(frame, k, mode='same')
        new_mask = (np.abs(new_mask)<6).astype(int)

    return new_mask

def time_fir(spatial_masks, filt=np.array([1,0.2,0])):
    """
    filter along the time axis 
    """
    # use mode full so that t=0 is zero padded 
    #  and then crop the extra elements off the end
    return np.apply_along_axis(lambda m: np.convolve(m, filt, mode='full'), 
        axis=2, arr=spatial_masks)


def task_metric(frame, mask, result):
    """
    Using a single frame from a sequence and a single mask from a sequence
    determine the:
        detected counts key: 'counts'
        acculated unmasked frames 'unmasked'

    This function previously also calculated the task metric by finding the confidence in the gradient.
        However this is very slow and is no longer used.
        So this function is used for two lines:
            result['counts'] += frame*(1-mask)
            result['unmasked'] += -1*(mask-1)
    Args:
        frame (ndarray): 2d image of 0s / 1s
        mask (ndarray): 2d matrix of 0s / 1s. 1 indciates that the pixel is inhibited 
        result (dict): accumulates 'counts' and 'unmasked'. if input is None this is initialized 

    Returns:
        result (dict):initialized if empty, result is an input parameter so that it can be passed in
                      and incremented. Important keys are 'counts', 'unmasked' which accumulate for each pixel. 
    """
    frame = frame.astype(int) # convert to 0s, 1s for addition, subtraction

    if result is None: # initialize a result dictionary
        result={}
        result['counts'] = np.zeros(frame.shape)
        result['unmasked'] = np.zeros(frame.shape)
        result['inhibited'] = np.zeros(frame.shape)
    
    result['counts'] += frame*(1-mask) # was a pixel unmasked and struck by a photon?
    result['unmasked'] += 1*(1-mask)  # was a pixel unmasked?
    result['inhibited'] += frame*mask
    
    return result

def to_intensity(x, skip=False):
    """
    convert a probability image to exposure (intensity). 
    Replace undefined with the maximum defined value in the image
    None: the replacement fails if no pixels have a defined intensity, in this scenario return inf. 
    """
    
    if skip:
        return x
    else:
        intensity = -np.log(1-x)

        try:
            imax = intensity[np.isfinite(intensity)].max()
            intensity[np.isinf(intensity)] = imax 
        except: # no finite values in the image 
            pass # return an infinite intensity
        return intensity


def obj_function(summary_stats, task, fp_metric, lam=1e-5, method='grad_mse', max_frame=39, crop_mse=False, convert_to_intensity=False):
    """
    calculate the objective function as:
        OBJ = MSE + lam*(photons detected)
        OBJ = Error + weight*energy   [more generally]
    various modes are avaliable for the error calculation

    Args:
        summary_stats (dict): uses the keys 'total_photons', 'masked_photons', 'iid'
        task (dict): uses the values in the max_frame for 'counts' in each pixel divided by the number 'unmasked' frames.
                     [strangely named]
        fp_metric (ndarray): a matrix of the full prescision metric (to calculate the mean square error against)
        lam (float): weighting of the energy term in the objective function
        method (string): method to determine the error. 'grad_mse', 'sq_grad_mse', 'intensity_mse', 'matlab_fscore'
            'grad_confidence' is no longer calculated 
        max_frames (int): the frame number to accumulate counts and masks up to. This is a key to the task dictionary 

    Returns:
        (tuple) error term, error_term + energy term
    """
    grad_y_kernel= np.array([[-1],[1]])
    y_cnts = task[max_frame]['counts']/task[max_frame]['unmasked']

    if 'matlab' not in method:
        if convert_to_intensity:
            y_cnts = to_intensity(y_cnts)
            fp_metric = to_intensity(fp_metric)

    if method == 'grad_mse':
        grad_y_cnts = convolve2d(y_cnts, grad_y_kernel)
        if crop_mse:
            grad_mse = mse(grad_y_cnts[1:-1, 1:-1], fp_metric[1:-1, 1:-1])
        else:
            grad_mse = mse(grad_y_cnts, fp_metric)
        return grad_mse, grad_mse + lam*(summary_stats['total_photons'] - summary_stats['masked_photons'])

    if method == 'sq_grad_mse':
        grad_y_cnts = convolve2d(y_cnts, grad_y_kernel)
        if crop_mse:
            grad_mse = mse(grad_y_cnts[1:-1, 1:-1], fp_metric[1:-1, 1:-1], 
                           element_scale=(fp_metric[1:-1, 1:-1]**2)) # emphasize edges by multiplying by the square of the gradient
        else:
            grad_mse = mse(grad_y_cnts[1:-1, 1:-1], fp_metric[1:-1, 1:-1], 
                           element_scale=(fp_metric[1:-1, 1:-1]**2)) # emphasize edges by multiplying by the square of the gradient
        return grad_mse, grad_mse + lam*(summary_stats['total_photons'] - summary_stats['masked_photons'])

    if method == 'mse':
        if crop_mse: # remove the borders before calculating the metric
            int_mse = mse(y_cnts[1:-1, 1:-1], fp_metric[1:-1, 1:-1])
        else:
            int_mse = mse(y_cnts, fp_metric)
        return int_mse, int_mse + lam*(summary_stats['total_photons'] - summary_stats['masked_photons'])

    if method == 'msre':
        if crop_mse: # remove the borders before calculating the metric
            int_mse = msre(y_cnts[1:-1, 1:-1], fp_metric[1:-1, 1:-1])
        else:
            int_mse = msre(y_cnts, fp_metric)
        return int_mse, int_mse + lam*(summary_stats['total_photons'] - summary_stats['masked_photons'])

    if method == 'grad_confidence':
        return (summary_stats['conf_nomask'] - summary_stats['conf_mask'])/summary_stats['conf_nomask'], (summary_stats['conf_nomask'] - summary_stats['conf_mask'])/summary_stats['conf_nomask'] + lam*(summary_stats['total_photons'] - summary_stats['masked_photons'])

    if method == 'matlab_fscore':
        fscore, segs, pb = eval_fscore(im=y_cnts, iid=summary_stats['iid'], convert_to_intensity=convert_to_intensity) 
        return fscore, fscore + lam*(summary_stats['total_photons'] - summary_stats['masked_photons']), pb

    if method == 'matlab_fscore_hdr':
        roi = summary_stats['img_roi']
        y_crop = y_cnts[(roi[0]-1):roi[1], (roi[2]-1):roi[3]] # -1 to match with MATLAB indexing
        fscore, pb = eval_fscore_hdr(im=y_crop, ref_name=summary_stats['iid'], roi=summary_stats['ref_roi']) 
        return fscore, fscore + lam*(summary_stats['total_photons'] - summary_stats['masked_photons']), pb


def determine_masks(frame, 
                      spatial_kernel=np.array([[0,1,0],
                                               [1,1,1],
                                               [0,1,0]]),
                      time_filter = np.array([1,1,1,1]),
                      threshold=5,
                      inhibition_length=1,
                      random_mask=False,
                      random_thresh=0.5,
                      inhibit_result=None,
                      combine_operators = ['and']):
    '''
    Params:
        frame (dict of 2d nd.arrays or a 3d nd.array)
        spatial_kernel (ndarray) or (list): 
        time_filter (ndarray, 1D): temporal kernel 
        threshold (float): the value for which if greater or equal to the pixel is masked. Can be a list or a list of lists  
        inhibition_length (int): if the mask exceeds or is equal to the threshold inhibited for this many frames (dead-time / holdoff time) 
        random_mask (bool): all zeros or all ones mask determine randomly
        random_thresh (float): fraction of time random mask is all zeros 
        inhibit_result: tracks metadata and data  
        combine_operators: list of either 'and', 'or' that determines how masks are combined when a list of spatial_kernels are used  
    Returns:
        (tuple) task: (dict) (key is the frame index) of 2d array of counts and unmasked
        task_nomask: (dict) of 2d array of counts and unmasked  
        mask_final (3darray): the mask for each frame and each pixel
        summary_stats (dict)
    '''
    # prepare to apply inhibition policy
    if type(frame) == np.ndarray:
        shape_3d = np.shape(frame)
        shape_2d = np.shape(frame[:,:,0])
    else: # dictionary
        shape_3d = tuple(list(frame[0].shape) + [len(frame) + 1]) 
        shape_2d = frame[0].shape

    mask_space = np.zeros(shape_3d)
    mask_time = np.zeros(shape_3d)
    mask_final = np.zeros(shape_3d)
    mask_zeros = np.zeros(shape_2d)
    mask_ones = np.ones(shape_2d)
    
    frames3d = np.zeros(shape_3d, dtype=np.bool_) # convert to a 3d numpy matrix. frames is a dictionary of 
                                                  #  2d matrices

    if type(frame) == np.ndarray:
        frames3d = frame
    else:
        for frame_idx in range(len(frame)):
            frames3d[:,:,frame_idx] = frame[frame_idx]

    # Inhibition: 
    #       space, time, thresholding all at once since the 0/1 mask is required to determine the inhibition policy
    #       for the next frame
    #  For each kernel need to store:  mask_space, mask_time. At the end of each frame must calculate mask_final 
    if type(spatial_kernel) is list:
        mask_space_list = [mask_space]*len(spatial_kernel)
        mask_time_list = [mask_time]*len(spatial_kernel)
        mask_idx_list = [None]*len(spatial_kernel)

        for frame_idx in range(frames3d.shape[2]-1):
            for k_idx, sp_kernel in enumerate(spatial_kernel):        
                # spatial kernel 
                mask_space_list[k_idx][:,:,frame_idx + 1] = inhibit_policy(frames3d[:,:,frame_idx], mask_final[:,:,frame_idx], 
                                                                kernel=sp_kernel)
                # apply inhibition policy filter along time dimesion
                start_idx = np.max([0, (frame_idx + 1 - len(time_filter))]) # ensure at least 0
                # TODO: time_filter is the same for each spatial kernel. Allow these to be different
                mask_time_tmp2 = convolve1d(mask_space[:,:,(start_idx):(frame_idx + 1)], time_filter, axis=2, mode='constant', cval=0)
                if (len(time_filter)-3) < (mask_time_tmp2.shape[2]):
                    mask_time_list[k_idx][:,:,frame_idx+1] = mask_time_tmp2[:,:, len(time_filter)-3]
                else:
                    sz = mask_time_tmp2.shape
                    mask_time_list[k_idx][:,:,frame_idx+1] = np.zeros((sz[0], sz[1]))
                #pdb.set_trace()

                # apply thresholding on the spatial and time filtered mask 
                # three options 
                if type(threshold[k_idx]) is list: # In between two values. check if its less than value 0 or greater than value 1 - useful for Laplacian and edge detection   
                    # example:  threshold = [-2, 2] -- mask if small value in between -2 and 2 of the Lap
                    mask_idx_list[k_idx] = (mask_time[:,:,frame_idx + 1]>=threshold[k_idx][0]) & (mask_time[:,:,frame_idx + 1]<=threshold[k_idx][1])
                elif threshold[k_idx] > 0: # if not list assume number. if positive check if value is greater 
                    mask_idx_list[k_idx] = mask_time[:,:,frame_idx + 1]>=threshold[k_idx]
                elif threshold[k_idx] < 0: # check if value is less than (the negative of the threshold
                    mask_idx_list[k_idx] = mask_time[:,:,frame_idx + 1]<=-threshold[k_idx]
            # now create the final mask
            mask_idx = mask_idx_list[0]
            for co_idx, co in enumerate(combine_operators): 
                if co == 'and': # note that there is no order of operation precendence here. It works as ((a operator b) operator c) etc. 
                                # Order the spatial kernels based on desired precendence
                    mask_idx = mask_idx & mask_idx_list[co_idx+1] 
                elif co == 'or':
                    mask_idx = mask_idx | mask_idx_list[co_idx+1] 
            mask_final[mask_idx, (frame_idx + 1):(frame_idx + 1 + inhibition_length)] = 1

    else: # conventional inhibition with only one spatial kernel
        for frame_idx in range(frames3d.shape[2]-1):
            # spatial kernel 
            mask_space[:,:,frame_idx + 1] = inhibit_policy(frames3d[:,:,frame_idx], mask_final[:,:,frame_idx], 
                                                            kernel=spatial_kernel)
            # apply inhibition policy filter along time dimesion
            start_idx = np.max([0, (frame_idx + 1 - len(time_filter))]) # ensure at least 0
            mask_time_tmp2 = convolve1d(mask_space[:,:,(start_idx):(frame_idx + 1)], time_filter, axis=2, mode='constant', cval=0)
            if (len(time_filter)-3) < (mask_time_tmp2.shape[2]):
                mask_time[:,:,frame_idx+1] = mask_time_tmp2[:,:, len(time_filter)-3]
            else:
                sz = mask_time_tmp2.shape
                mask_time[:,:,frame_idx+1] = np.zeros((sz[0], sz[1]))
            #pdb.set_trace()

            # apply thresholding on the spatial and time filtered mask 
            # three options 
            if type(threshold) is list: # In between two values. check if its less than value 0 or greater than value 1 - useful for Laplacian and edge detection   
                # example:  threshold = [-2, 2] -- mask if small value in between -2 and 2 of the Lap
                mask_final[(mask_time[:,:,frame_idx + 1]>=threshold[0]) & (mask_time[:,:,frame_idx + 1]<=threshold[1]), (frame_idx + 1):(frame_idx + 1 + inhibition_length)] = 1
            elif threshold > 0: # if not list assume number. if positive check if value is greater 
                mask_final[(mask_time[:,:,frame_idx + 1]>=threshold), (frame_idx + 1):(frame_idx + 1 + inhibition_length)] = 1
            elif threshold < 0: # check if value is less than (the negative of the threshold
                mask_final[(mask_time[:,:,frame_idx + 1]<=-threshold), (frame_idx + 1):(frame_idx + 1 + inhibition_length)] = 1

    # evaluate counts and mask counts
    summary_stats = {}
    task = {}
    task[-1] = None
    task_nomask = {}
    task_nomask[-1] = None
    for frame_idx in range(frames3d.shape[2]-1):
        task_prev = copy.deepcopy(task[frame_idx-1]) 
        if random_mask:
            if ((np.random.rand()>random_thresh) and (frame_idx>0)): #ensure the first frame is not all masked
                mask_final[:,:,frame_idx]=mask_ones
            else:
                mask_final[:,:,frame_idx]=mask_zeros
        # task_metric is no longer the best name for this function
        task[frame_idx] = task_metric(frames3d[:,:,frame_idx], mask_final[:,:,frame_idx], task_prev)
        # no mask calculations 
        task_nomask_prev = copy.deepcopy(task_nomask[frame_idx-1]) 
        task_nomask[frame_idx] = task_metric(frames3d[:,:,frame_idx], mask_zeros, task_nomask_prev)

    # independent of task
    summary_stats['total_photons'] = np.sum(frames3d[:,:,0:-1]) 
    masked_frames = frames3d*mask_final
    summary_stats['masked_photons'] = np.sum(masked_frames[:,:,0:-1]) 
    summary_stats['total_masked'] = np.sum(mask_final[:,:,0:-1]) 
    summary_stats['pixels_times_frames'] = np.product(frames3d[:,:,0:-1].shape)
    summary_stats['avg_mask'] = np.average(mask_final[:,:,0:-1], axis=2)

    if inhibit_result is not None:
        #metadata 
        inhibit_result.spatial_kernel = spatial_kernel
        inhibit_result.time_kernel = time_filter
        inhibit_result.inhibit_thresh = threshold
        inhibit_result.inhibit_length = inhibition_length 
        # data
        inhibit_result.captures = task
        inhibit_result.captures_nomask = task_nomask
        inhibit_result.mask = mask_final

    return task, task_nomask, mask_final, summary_stats


def img_gif(figure_dir, figure_prefix, img, num_imgs=40, fps=3):
    """
        Create a GIF (movie) of the image reconstruction with the 3rd dimension (i.e. time)
        a new frame.

        create another function that is always used to calculated intensity from image and mask
        and that summarizes the total number of photon dections

        Args:
            figure_dir (string): directory to save GIF
            figure_prefix (string): Name for GIF omit extension
            img (np.ndarray): should be 3D
    """
    from matplotlib.animation import FuncAnimation

    composite = True
    
    # num_imgs can be an array of img indices or an int that describes how many steps in log space
    if type(num_imgs) == int:
        fig_range = np.floor(np.logspace(0, np.log10(img.shape[2]-1), num_imgs)).astype(int)
        fig_range = np.unique(fig_range) # remove duplicates 
    
    fig,ax=plt.subplots(nrows=1, ncols=1)

    def animate(i):
        # plot the image matrix
        if composite:
            im = np.mean(img[:,:,0:i], axis=2)
        else:
            im = img[:,:,i]
        ax.title.set_text('n={}'.format(i))
        im_out = ax.imshow(im, cmap='gray')
        
        return im_out,

    # interval is in ms (fps seems more impactful)
    anim = FuncAnimation(fig, animate, frames=fig_range, interval=200, blit=True)

    #anim.save(os.path.join(figure_dir, f'{figure_prefix}_masks.mp4'), writer='ffmpeg', fps=fps)
    anim.save(os.path.join(figure_dir, f'{figure_prefix}_masks.gif'), fps=fps)

def mask_gif(figure_dir, figure_prefix, mask_final):
    """
        Create a GIF (movie) of the mask with the 3rd dimension (i.e. time)
        a new frame.

        TODO: generalize this GIF creation so that it can be an intensity image and be scaled appropriately
        create another function that is always used to calculated intensity from image and mask
        and that summarizes the total number of photon dections

        Args:
            figure_dir (string): directory to save GIF
            figure_prefix (string): Name for GIF omit extension
            mask_final (np.ndarray): should be 3D
    """
    composite = True
    num_imgs = 40
    fig,ax=plt.subplots(nrows=1, ncols=1)
    filenames = []
    fig_range = np.floor(np.logspace(0, np.log10(mask_final.shape[2]-1), num_imgs)).astype(int)
    fig_range = np.unique(fig_range) # remove duplicates 
    for i in fig_range:
        # plot the image matrix
        if composite:
            im = np.mean(mask_final[:,:,0:i], axis=2)
        else:
            im = mask_final[:,:,i]
        ax.imshow(im, cmap='gray')
        ax.title.set_text('n={}'.format(i))
        # create file name and append it to a list
        filename = os.path.join(figure_dir, f'{i}.png')
        filenames.append(filename)        
        # save frame
        plt.savefig(filename)

    print('Filenames list')
    print(filenames)
    # build GIF
    with imageio.get_writer(os.path.join(figure_dir, f'{figure_prefix}_masks.gif'), mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    # Remove temporary files for the GIF
    for filename in set(filenames):
        print(f'removing {filename}')
        os.remove(filename)

    def plot_avg_mask(figure_dir, figure_prefix, mask_final):
        pass
        """
        fig,ax=plt.subplots(nrows=1, ncols=1)
        im = ax.imshow(np.average(mask_final[:,:,0:-1], axis=2), cmap='gray', clim=(0,1))
        ax.title.set_text('Avg. mask')
        fig.colorbar(im)
        my_savefig(fig, figure_dir, f'{figure_prefix}_avg_mask')
        """

def mse(x1, x2, element_scale=1):
    ''' calculate the mean squared error of two matrices
        with an optional scaling to emphasize certain values
    '''
    return (element_scale*(x1 - x2) ** 2).mean(axis=None)

def msre(x1, x2):
    '''
    mean square relative error (order of inputs matters). Reference image first.

    '''
    re = np.divide((x1 - x2), x1)**2
    re = re[np.isfinite(re)] # remove values when x1 is 0
    return re.mean(axis=None) 

def total_opt_f(x0, frame, lam=1e-4, threshold=1, method='grad_mse', mode='sym',
                inhibition_length=1, max_frame=39, crop_mse=False):

    # mode = 'sym'    
    #   only three indepenent variables for K_s 
    # mode = 'four'
    #   four independent variables for K_s 
    # mode = 'all'
    #   9 independet variables for K_s
    # 4 variables for K_t 
    # force threshold to 1. Can scale others later 
    total_opt_f.count += 1

    K_s, K_t = kernels_from_x(x0, mode=mode)

    task, task_nomask, mask_final, summary_stats = determine_masks(frame, 
                                                       spatial_kernel=K_s,     
                                                       time_filter = K_t, 
                                                       threshold=threshold,
                                                       inhibition_length=inhibition_length)

    grad_mse, obj = obj_function(summary_stats, task, grad_y_fp, lam=lam, method=method, 
                                 max_frame=max_frame, crop_mse=crop_mse)

    total_opt_f.res['obj'] = np.append(total_opt_f.res['obj'], obj)
    total_opt_f.res['mse'] = np.append(total_opt_f.res['mse'], grad_mse)
    if total_opt_f.res['x0'] is None:
        total_opt_f.res['x0'] = x0
    else:
        total_opt_f.res['x0'] = np.vstack((total_opt_f.res['x0'], x0))
    total_opt_f.res['stats'].append(summary_stats)

    return obj

def create_bern_frames(y, rv_arrays, rv_idx, rv_dict_len, 
                       num_frames=40, threshold_ranges=np.array([0.1])):
    '''
    create Bernoulli frames 
    smaller threshold is equivalent to the image being captured with a "longer exposure time" 
    image flux = lambda*Tint = (pixel value)/threshold 
    the interesting threshold range is compressed by Bernoulli sampling 

    Args:
    y (ndarray): full precision image from 0 to 1 
    rv_arrays (ndarray): random variable lookup 
    rv_idx (ndarray): index of next up lookup value
    rv_dict_len (ndarray): length of lookup array -- to know when to circle back
    threshold_ranges: nd.array only changes the accumulated counts, the frame returned is from the last threshold value
                      threshold is (None,val) the image is normalized so that the average flux is val

    '''
    acc_counts = {}  # store the total accumulated counts at each frame index for further processing

    # create Bernoulli frames
    for thresh_idx, thresh in enumerate(threshold_ranges):
        if type(thresh) is tuple:
            y_rate = y/np.mean(y)*thresh[1]  # normalize image to a certain flux value
        else: # thresh is a single value (float or int)
            # create a probability frame after scaling by exposure time (1/thresh = exposure time)
            y_rate = y/thresh  # image rate values 
        y_prob = img_probability(y_rate) # discrete values 
        y_prob_full_precision = 1-np.exp(-y_rate)

        # measured acc_counts 
        acc_counts[thresh_idx] = np.zeros(np.shape(y_prob), dtype=np.uint8)
        frame = {}

        for frame_idx in range(num_frames):
            frame[frame_idx] = np.full(np.shape(y_prob), False, dtype=bool)
            for iy, ix in np.ndindex(y_prob.shape):
                disc_prob = y_prob[iy,ix]
                frame[frame_idx][iy,ix] = rv_arrays[disc_prob][rv_idx[disc_prob]]
                rv_idx[disc_prob] = (rv_idx[disc_prob] + 1) % rv_dict_len

            # accumulated counts over each of the frames (max value of frames)
            acc_counts[thresh_idx] += frame[frame_idx]

    return frame, acc_counts, y_prob_full_precision


def bernoulli_lookup(num_steps=4096, seed=None):

    if seed is not None:
        from numpy.random import Generator, PCG64

        numpy_randomGen = Generator(PCG64(seed))
        bernoulli.random_state=numpy_randomGen

    ################################################
    ## Create bernoulli lookup function
    ################################################
    discrete_probs = np.arange(0, num_steps + 1, 1, dtype=np.int16)

    # Bernoulli random variable (rv) array and index to keep track of last sample.
    rv_arrays = {}
    rv_idx = {}
    rv_dict_len = 9973 # make this prime so that there is no strange repeating if the image has say 128 of a certain value 8192

    for dp in discrete_probs:
        rv_arrays[dp] = bernoulli.rvs(dp/num_steps, size=rv_dict_len)
        rv_idx[dp] = 0 # advance this index when a sample is drawn

    return rv_arrays, rv_idx, rv_dict_len



def make_figures(y, y_prob_full_precision, task, task_nomask, mask_final, summary_stats, 
                 figure_prefix, figure_dir, data_dir):

    num_frames = len(task) - 1
    img_w, img_h = y.shape
    grad_y_kernel = np.array([[-1],[1]])

    fig,ax=plt.subplots(nrows=1, ncols=4, figsize=(9,6.72))
    ax[0].imshow(y, cmap='gray')
    ax[1].imshow(task[num_frames-1]['down']['diff'], cmap='gray')
    ax[2].imshow(task[num_frames-1]['down']['counts']/2, cmap='gray')
    ax[3].imshow(mask_final[:,:,10], cmap='gray')
    for idx,title in enumerate(['Full Precision Image', '$Grad_y$', '$I_1 + I_2$', 'Inhibit [n=10]']):
        ax[idx].title.set_text(title)
    my_savefig(fig, figure_dir, f'{figure_prefix}_task_performance_mask')


    fig, ax=plt.subplots(nrows=1, ncols=4, figsize=(9,6.72))
    ax[0].imshow(y, cmap='gray')
    ax[1].imshow(task_nomask[num_frames-1]['down']['diff'], cmap='gray')
    ax[2].imshow(task_nomask[num_frames-1]['down']['counts']/2, cmap='gray')
    ax[3].imshow(mask_final[:,:,10], cmap='gray')
    for idx,title in enumerate(['Full Precision Image', '$Grad_y$', '$I_1 + I_2$', 'Inhibit [n=10]']):
        ax[idx].set_title(title, fontdict={'fontsize': 10})
    my_savefig(fig, figure_dir, f'{figure_prefix}_task_performance_mask')

    ## average line cuts of gradient & gradient confidence 
    fig, ax=plt.subplots(nrows=3, ncols=1)
    # plot gradient difference 
    axis_num = 1
    ax[0].plot(np.mean(task[num_frames-1]['down']['diff'], axis=axis_num), label='Inhibit', marker='*')
    ax[0].plot(np.mean(task_nomask[num_frames-1]['down']['diff'], axis=axis_num), label='No Inhibit')
    ax[0].set_ylabel('$Grad_y$')

    ax[1].plot(np.mean(task[num_frames-1]['down']['counts'], axis=axis_num), label='Inhibit', marker='*')
    ax[1].plot(np.mean(task_nomask[num_frames-1]['down']['counts'], axis=axis_num), label='No Inhibit')
    ax[1].plot(np.mean(y*(num_frames)*2, axis=axis_num), label='Ref. img.', linestyle='--')
    ax[1].set_ylabel('$I_1 + I_2$')

    ax[2].plot(np.mean(task[num_frames-1]['down']['conf'], axis=axis_num), label='Inhibit', marker='*')
    ax[2].plot(np.mean(task_nomask[num_frames-1]['down']['conf'], axis=axis_num), label='No Inhibit')
    ax[2].set_ylabel('$Grad_y$ Confidence ($\mu / \sigma$)')

    for a in [0,1,2]:
        ax[a].set_xlim([0, img_w])
        ax[a].set_xlabel('Row')
        ax[a].legend(prop={'size': 8})
    my_savefig(fig, figure_dir, f'{figure_prefix}_gradient_line_cuts')

    ## Image of the Gradient confidence with and without masking 
    # assess the gradients

    max_frame = len(task)-2
    grad_y_fp = convolve2d(y_prob_full_precision, grad_y_kernel)  # for comparsion / calculation of
    grad_y_cnts = convolve2d(task[max_frame]['counts']/task[max_frame]['unmasked'], grad_y_kernel)
    grad_y_cnts_nomask = convolve2d(task_nomask[max_frame]['counts']/task_nomask[max_frame]['unmasked'], grad_y_kernel)

    # image and gradient. Full precision, binary, masked binary
    fig, ax = plt.subplots(2,3)
    ax[0,0].imshow(y_prob_full_precision)
    ax[0,1].imshow(task[max_frame]['counts']/task[max_frame]['unmasked'])
    ax[0,2].imshow(task_nomask[max_frame]['counts']/task_nomask[max_frame]['unmasked'])
    for idx,title in enumerate(['Full Precision Img', 'Masked Img', 'No mask Img']):
        ax[0,idx].title.set_text(title)

    clim=(np.percentile(grad_y_fp[1:-1,:], 2), np.percentile(grad_y_fp[1:-1,:], 98))
    im = ax[1,0].imshow(grad_y_fp[1:-1,:], clim=clim)
    clim=im.properties()['clim']
    ax[1,1].imshow(grad_y_cnts[1:-1,:], clim=clim)
    ax[1,2].imshow(grad_y_cnts_nomask[1:-1,:], clim=clim)
    for idx,title in enumerate(['Full Precision $G_y$', 'Masked $G_y$', 'No mask $G_y$']):
        ax[1,idx].title.set_text(title)
    my_savefig(fig, figure_dir, f'{figure_prefix}_img_gradient_y')

    fig, ax = plt.subplots(1,3)
    clim=(np.percentile(grad_y_fp[1:-1,:], 2), np.percentile(grad_y_fp[1:-1,:], 98))
    im = ax[0].imshow(grad_y_fp[1:-1,:], clim=clim)
    clim=im.properties()['clim']
    ax[1].imshow(grad_y_cnts[1:-1,:], clim=clim)
    ax[2].imshow(grad_y_cnts_nomask[1:-1,:], clim=clim)
    for idx,title in enumerate(['Full Precision $G_y$', 'Masked $G_y$', 'No mask $G_y$']):
        ax[idx].title.set_text(title)
    my_savefig(fig, figure_dir, f'{figure_prefix}_gradient_y')

    print(f'MSE of masked: {mse(grad_y_cnts, grad_y_fp)}')
    mse_nomask = mse(grad_y_cnts_nomask, grad_y_fp)
    print(f'MSE of no masked: {mse(grad_y_cnts_nomask, grad_y_fp)}')

    ## create a GIF of the mask
    mask_gif(figure_dir, figure_prefix, mask_final)
    plot_avg_mask(figure_dir, figure_prefix, mask_final)

    df = pd.DataFrame.from_dict([summary_stats])
    # save entire Pandas data frame to CSV
    df.to_csv(os.path.join(data_dir, f'{figure_prefix}_data_summary.csv'))

