"""
Various utility functions used throughout this project.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def kernels_from_x(x0, mode='sym'):
    ''' optimize takes a single array x 
        take this 1d array and make the spatial (K_s) and temporal (K_t) kernels

    Args:
        x0 (array or list): array or list (length of list depends on mode)
        mode (string): sets mode of how the spatial and temporal kernels are created
                       options: 'sym', 'four', 'all'

    Returns:
        K_s, K_t : two Kernel matrices
    '''

    if mode == 'sym':
        udlr = x0[1]
        center = x0[0]
        corner = x0[2]

        K_s = np.array([[corner, udlr, corner],
                        [udlr,   center, udlr],
                         [corner, udlr, corner]])
        K_t = x0[3:7]

    if mode == 'four':
        ud = x0[1]
        lr = x0[2]
        center = x0[0]
        corner = x0[3]

        K_s = np.array([[corner, ud, corner],
                        [lr,   center, lr],
                         [corner, ud, corner]])
        K_t = x0[4:8]

    if mode == 'all':
        K_s = np.reshape(x0[0:9],(3,3))
        K_t = x0[9:13]

    return K_s, K_t

def my_savefig(fig, figure_dir, figname, tight=True):
    """ 
    save a figure given a handle and directory
    Params: 
        fig: matplotlib handle
        figure_dir: directory 
        figname: desired output name 

    Returns:
        None
    """
    if tight:
        fig.tight_layout()
    for e in ['.png', '.pdf','.svg']: # use pdf rather than eps to support transparency
        fig.savefig(os.path.join(figure_dir,
                                 figname + e), dpi=600)


def disp_img(img):
    """
    display image 
    """
    plt.imshow(img)


def create_img(name):
    """
    create a toy image with a given contrast and edge spacing
    Params:
        name (string): only options right now is 'toy'
    
    Returns:
        y : an image matrix 
    """

    if name == 'toy':
        y = np.zeros((64,16))
        img_w, img_h = y.shape
        # to start edges only run in one direction
        edge_width = 6
        edge_spacing = 6

        contrast = 0.1
        contrast_step = 0.3 

        w = 0
        while w < img_w - (edge_width*2+edge_spacing):
            y[w:(w+edge_width),:] = 1
            y[(w+edge_width):(w+2*edge_width),:] = 1*contrast 
            contrast = contrast + contrast_step
            w += edge_width*2 + edge_spacing 

    return y



def GMSD(img, ref, rescale=True, ret_map=False):
    """
    Copied from neutompy since there were way too many requirements

    This function computes the Gradient Magnitude Similarity Deviation (GMSD).
    This is a Python version of the Matlab script provided by the authors in [3]_

    Parameters
    ----------
    img : 2d array
        Image to compare.

    ref : 2d array
        Reference image.

    rescale : bool, optional
        If True the input images were rescaled in such a way that `ref` has a
        maximum pixel value of 255. If False no rescaling is performed.
        Default value is True.

    ret_map : bool, optional
        If True the GMSD and GMS map are returned in a tuple.
        Default value is False.

    Returns
    -------
    gmsd_val : float
        The GMSD value.

    gms_map : 2d array
        The GMS map, returned only if `map` is True.

    References
    ----------
    .. [3] Wufeng Xue, Lei Zhang, Xuanqin Mou, and Alan C. Bovik,  "Gradient Magnitude
        Similarity Deviation: A Highly Efficient Perceptual Image Quality Index",
        http://www.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
    """
    if rescale:
        k = 255.0 / ref.max()
    else:
        k = 1.0

    T = 170
    downscale = 2
    hx = (1.0/3.0) * np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]])
    hy = hx.T

    avg_kernel = np.ones((2, 2)) / 4.0
    avg_ref = signal.convolve2d(k * ref, avg_kernel, mode='same', boundary='symm')[::downscale,::downscale]
    avg_img = signal.convolve2d(k * img, avg_kernel, mode='same', boundary='symm')[::downscale,::downscale]

    ref_dx =  signal.convolve2d(avg_ref, hx, mode='same', boundary='symm')
    ref_dy =  signal.convolve2d(avg_ref, hy, mode='same', boundary='symm')
    MG_ref =  np.sqrt(ref_dx**2 + ref_dy**2)

    img_dx =  signal.convolve2d(avg_img, hx, mode='same', boundary='symm')
    img_dy =  signal.convolve2d(avg_img, hy, mode='same', boundary='symm')
    MG_img =  np.sqrt(img_dx**2 + img_dy**2)

    gms_map = (2*MG_ref*MG_img + T) / (MG_img**2 + MG_ref**2 + T)

    gmsd_val = np.std(gms_map)
    gms_avg = np.mean(gms_map)
    
    if ret_map:
        return gmsd_val, gms_avg, gms_map
    else:
        return gmsd_val, gms_avg

def GradientMag(img, rescale=False, downscale=1):

    """
    Copied from GMSD above
    (not used above since need the rescaling factor should be the same for reference and image)

    Modified from GMSD function above 
    This is a Python version of the Matlab script provided by the authors in [3]

    Parameters
    ----------
    img : 2d array
        Image to compare.

    rescale : bool, optional
        If True the input images were rescaled in such a way that `ref` has a
        maximum pixel value of 255. If False no rescaling is performed.
        Default value is True.

    downscale : int
        For a 1-1 mapping of GradientMagnitude and the original image this needs to be 1
        GMSD calculations use 2

    Returns
    -------
    gm : 2d array float
        The gradient magnitude.

    References
    ----------
    .. [3] Wufeng Xue, Lei Zhang, Xuanqin Mou, and Alan C. Bovik,  "Gradient Magnitude
        Similarity Deviation: A Highly Efficient Perceptual Image Quality Index",
        http://www.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
    """
    if rescale:
        k = 255.0 / img.max()
    else:
        k = 1.0

    T = 170
    hx = (1.0/3.0) * np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]])
    hy = hx.T

    avg_kernel = np.ones((2, 2)) / 4.0
    avg_img = signal.convolve2d(k * img, avg_kernel, mode='same', boundary='symm')[::downscale,::downscale]

    img_dx =  signal.convolve2d(avg_img, hx, mode='same', boundary='symm')
    img_dy =  signal.convolve2d(avg_img, hy, mode='same', boundary='symm')
    MG_img =  np.sqrt(img_dx**2 + img_dy**2)

    return MG_img
