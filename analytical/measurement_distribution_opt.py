# photon inhibition by exposure time 
# And optimizing MSE by distributing frames
#
# 2022/10/11, Lucas Koerner 
# 
# system-level limitations: 
# 1) sensing latency 
# 2) number of frames 
# 3) energy / power 
# 4) SNR limit

from scipy.stats import poisson, bernoulli 
import numpy as np
import matplotlib.pyplot as plt
import copy
import glob
import os 
from skimage.metrics import structural_similarity as ssim
import pandas as pd

from utils import disp_img, create_img, my_savefig, GMSD, GradientMag
from create_binary_images import open_img 
figure_dir = '../manuscript/figures/'
results_dir = '../data/optimal_detections/'
plt.ion()

from load_config import * # creates a dictionary named 'config'
print(f'BSDS path: {config["bsds_dir"]}')
bsds_folders = ['test', 'train', 'val']

def img_files_in_dir(directory, extension):
    # return a list of image files in a directory with a given extension

    # returns a list of the whole path 
    file_list = glob.glob(os.path.join(directory, f'*{extension}' ))
    return file_list


def img_mse(y_or_p, texp, N, mode='flux'):

    # y_or_p: matrix of flux or binary rate (probability)
    # t_exp: only needed if mode is flux 
    # N: measurements for each pixel 
    # mode: (str) flux or probability 

    # given the inensity/flux as y and 
    # the exposure time and number of frames 
    # determine the image MSE if the image is sampled by a Bernoulli imager 
    # N can be a matrix or scalar 
    if mode == 'flux':
        img_std = np.multiply(1/np.sqrt(N), np.sqrt(1-np.exp(-y_or_p*texp))/np.sqrt(np.exp(-y_or_p*texp)))
        max_sig = np.max(np.multiply(y_or_p*texp,N))
        min_sig = np.min(np.multiply(y_or_p*texp,N))
    elif mode == 'probability':
        # TODO: check this 
        img_std = np.multiply(1/np.sqrt(N), np.sqrt(y_or_p*(1-y_or_p)))
        img_rel_std = np.multiply(img_std, (1/y_or_p)) 
        max_sig = np.max(np.multiply(-1/np.log(y_or_p), N))
        min_sig = np.min(np.multiply(-1/np.log(y_or_p), N))
  
    img_var = img_std**2
    mse = np.mean(img_var, axis=None)
    msre = np.mean(img_rel_std**2, axis=None)
    psnr = (max_sig/min_sig)/mse  # definition from Aydm "Extending quality metrics to full luminance"

    return mse, psnr, img_std, msre


def avalanche_pwr(N, H):
    return N*(1 - np.exp(-H)) # = N*p

def snr_exp(N, H):
    # exposure referred SNR from Fossum 
    return np.sqrt(N)*H*1/(np.sqrt(np.exp(H) - 1))

def two_by_two(img, meas_mat, lbl=['uniform', 'weighted']):
    """
    img is a probability image 
    meas_mat: is a list or tuple of measurements matrices, 
        assumption is the first one is uniform

    """

    print('Plot two by two' + '-'*80)

    fig,ax=plt.subplots(2,2, figsize=(10,8))
    im = ax[0,0].imshow(img)
    plt.colorbar(im)

    ax[0,0].set_title('Probability image')

    ax[0,1].imshow(meas_mat[1])
    ax[0,1].set_title('Measurement matrix')
    
    for m, l in zip(meas_mat, lbl):
        ax[1,0].semilogy(img.ravel(), m.ravel(), label=l, 
                        linestyle='none', marker='*')
        ax[1,0].set_xlabel('p')
        ax[1,0].set_ylabel('Measurements')
        ax[1,0].legend()

    baseline_avalanches = np.sum(np.multiply(img, meas_mat[0]))
    for m, l in zip(meas_mat, lbl):
        snr = snr_exp_vec(m, -np.log(1-img))    
        print(f'{l}: Total number of measurements: {np.sum(m)}')
        avalanches = np.sum(np.multiply(img, m))
        print(f'{l}: Expected number of avalanches: {avalanches}')
        mse, psnr, img_std, msre = img_mse(p, None, m, mode='probability')
        print(f'{l}: MSE = {mse}, psnr = {psnr}')
        mse, psnr, img_std, msre = img_mse(p, None, m*baseline_avalanches/avalanches, 
            mode='probability')
        print(f'{l}: (scaled) MSE = {mse}, psnr = {psnr}')
        scaled_avalanches = np.sum(np.multiply(img, m*baseline_avalanches/avalanches))
        print(f'{l}: (scaled) Expected number of avalanches: {scaled_avalanches}')
        ax[1,1].hist(snr.ravel(), bins=100, histtype=u'step',label=l) # ravel to create 1D array
        print('-'*20)

    ax[1,1].legend()
    ax[1,1].set_xlabel('SNR')
    ax[1,1].set_ylabel('Occurences')
    ax[1,0].legend()

    print('-'*80)

    return fig 


def search_better(p, N, metric_type, best_metric, searches = 100):

    best_search = np.inf 

    p_frac = np.random.uniform(size=(np.shape(p)))
    N_wt = p_frac/np.sum(p_frac)
    Nt_opt = N*N_wt*len(p.ravel())

    mse, psnr, img_std, msre = img_mse(p, None, Nt_opt, mode='probability')

    if metric_type == 'MSE':
        metric = mse
    elif metric_type == 'PSNR':
        metric = psnr
    elif metric_type == 'MSRE':
        metric = msre

    new_best_search = (metric-best_metric)/best_metric 

    if new_best_search < best_search: 
        best_search = new_best_search
        N_wt_best = N_wt 

    return best_search 

def optimal_msre_measurements(p, target_measurements_pix):

    # determine the optimal MSRE given a probability image and a number of measurements per pixel 
    
    # optimal distribution 
    N_wt = 1/((1-p)*p)**0.5/np.sum(1/((1-p)*p)**0.5) # see Blue notebook page 2023/2/26 - optimal MSRE (also 2022/11/18 for minimal MSE)
    Nt_opt = N_wt*len(p.ravel())
    num_pixels = np.prod(np.shape(p))
    m_pix = np.sum(Nt_opt)/num_pixels # number of measurements per pixel 
    Nt_opt = target_measurements_pix/m_pix*Nt_opt
    mse, psnr, img_std, optimal_msre = img_mse(p, None, Nt_opt, mode='probability')

    Nt = len(p.ravel())
    m_pix = np.sum(Nt_opt)/num_pixels
    Nt = target_measurements_pix/m_pix*Nt
    mse, psnr, img_std, msre_uniform = img_mse(p, None, Nt, mode='probability')

    # with improvement this metric will be > 1 
    improve_factor = msre_uniform/optimal_msre

    return optimal_msre, Nt_opt, improve_factor 

def optimal_mse(p, target_detections_pix):

    '''
    Optimize with a constraint on detections 

    sigma_i^2 = 1/W_i * p_i*(1-p_i)

    MSE = sum sigma_i^2 

    minimize MSE with respect to Wi 
    calculate dMSE / dWi and set to zero 

    then replace Wj with WT - Wi 
    where WT is fixed and is the constraint on maximum measurements 

    Otherwise if constraining a total number of detections (DT) 
    DT = pi*Wi + pj*Wj 
    ... DT = pi*Wi + pj*Wj + pk*Wk  
    and substitute Wj = (DT - pi*Wi - pk*Wk)/pj 

    '''
    # determine the optimal MSE given a probability image and a number of detections per pixel 
    
    # optimal distribution 
    N_wt = (p*(1-p))**0.5/np.sum( (p**2*(1-p))**0.5 )/np.sqrt(p) # this is optimal for a contstraint on detections 

    Nt_opt = N_wt*len(p.ravel())
    num_pixels = np.prod(np.shape(p))
    detections_pix = np.sum(np.multiply(Nt_opt, p))/num_pixels
    Nt_opt = target_detections_pix/detections_pix*Nt_opt
    mse_optimal_calc, psnr, img_std, optimal_msre = img_mse(p, None, Nt_opt, mode='probability')

    # get uniform measurements for a given number of detections/pix 
    Nt = len(p.ravel())
    detections_pix = np.sum(np.multiply(Nt, p))/num_pixels
    Nt = target_detections_pix/detections_pix*Nt
    mse_uniform, psnr, img_std, msre_uniform = img_mse(p, None, Nt, mode='probability')

    # with improvement this metric will be > 1 and a better MSE is lower 
    improve_factor = mse_uniform/mse_optimal_calc

    return mse_optimal_calc, Nt_opt, improve_factor 

def optimal_mse_measurements(p, target_per_pix):

    # determine the optimal MSE given a probability image and a contraint on the number of measurements per pixel 
    # TODO: need to update since this is the same as optimal_mse 
    # optimal distribution 
    # N_wt = p**0.5/((1-p)**0.5)/np.sum(p**0.5/((1-p)**0.5)) 
    N_wt = (p*(1-p))**0.5/np.sum((p*(1-p))**0.5) # this is optimal detections constrained by measurements 

    Nt_opt = N_wt*len(p.ravel())
    num_pixels = np.prod(np.shape(p))
    measures_pix = np.sum(Nt_opt)/num_pixels
    Nt_opt = target_per_pix/measures_pix*Nt_opt
    mse_optimal_calc, psnr, img_std, optimal_msre = img_mse(p, None, Nt_opt, mode='probability')

    Nt = len(p.ravel())
    detections_pix = np.sum(Nt)/num_pixels
    Nt = target_per_pix/measures_pix*Nt
    mse_uniform, psnr, img_std, msre_uniform = img_mse(p, None, Nt, mode='probability')

    # with improvement this metric will be > 1 and a better MSE is lower 
    improve_factor = mse_uniform/mse_optimal_calc

    return mse_optimal_calc, Nt_opt, improve_factor 

def optimal_msre(p, target_detections_pix):

    # determine the optimal MSRE given a probability image and a number of detections per pixel 
    # find the measurement weighting that optimizes the MSRE 

    # optimal distribution - modify the derivation from optimal_mse
    N_wt = (1/p*(1-p))**0.5/np.sum( ((1-p))**0.5 )/np.sqrt(p) # this is optimal for a contstraint on detections 
    # N_wt = 1/((1-p)**0.5*p)/np.sum(1/((1-p)**0.5*p)) 

    Nt_opt = N_wt*len(p.ravel())
    num_pixels = np.prod(np.shape(p))
    # normalize to detections per pixel 
    detections_pix = np.sum(np.multiply(Nt_opt, p))/num_pixels
    Nt_opt = target_detections_pix/detections_pix*Nt_opt
    mse, psnr, img_std, optimal_msre = img_mse(p, None, Nt_opt, mode='probability')

    Nt = len(p.ravel())
    detections_pix = np.sum(np.multiply(Nt, p))/num_pixels
    Nt = target_detections_pix/detections_pix*Nt
    mse, psnr, img_std, msre_uniform = img_mse(p, None, Nt, mode='probability')

    # with improvement this metric will be > 1 
    improve_factor = msre_uniform/optimal_msre

    return optimal_msre, Nt_opt, improve_factor 

def simulate_mse_msre(p=0.5, trials=10000, N=1000):
    '''
    simulate and calculate MSE and MSRE 
    of a probability image 
    The simulation draws from a Bernoulli random variable 
    '''
    measures = bernoulli.rvs(p, size=(N, trials))

    mse = np.mean((p-np.sum(measures, axis=1)/trials)**2)
    msre = np.mean(((p-np.sum(measures, axis=1)/trials)/p)**2)
    print(f'MSE = {mse}; MSRE = {msre}')

    mse_calc = p*(1-p)/trials
    msre_calc = (1-p)/p/trials
    print(f' MSE calc = {mse_calc}; MSRE calc = {msre_calc}')

    results = {'mse': mse, 'mse_calc': mse_calc, 'msre': msre, 'msre_calc': msre_calc}

    return results


def noisey_image(img, meas):
    """
    create a noisey binary rate image given a number of measurements
    assumes that the binary rate noise can be approximated as Gaussian (e.g. sufficient samples)
    """
    img_var = np.divide(np.multiply(1-img, img), meas)
    img_std = np.sqrt(img_var)

    add_noise = np.random.normal(loc=np.zeros(np.shape(img_std)), scale=img_std)
    return_img = img + add_noise 
    # don't allow a probability image to be less than 0 or greater than 1
    return_img[return_img<0] = 0
    return_img[return_img>1] = 1

    return return_img


# TODO: 
# 1) scale frames for equal detections 
# 2) search for a better weighting 

if __name__ == "__main__":
    plt.ion()
    fz = 16
    N = 100


    # check MSE per detection for flat field images
    mps = np.linspace(0.02,0.98,100) # maximum probability in the images 
    msre_arr = np.array([])
    improve_arr = np.array([])
    for max_prob in mps:

        p = np.ones((100,100))*max_prob
        Nt = N*np.ones(np.shape(p))
        num_pixels = np.prod(np.shape(p))
        
        print(f'Equal frame weighting, uniform probability of {max_prob}')
        detections = np.sum(np.multiply(Nt, p))
        mse, psnr, img_std, msre = img_mse(p, None, Nt, mode='probability')
        msre_arr = np.append(msre_arr, mse)
        print(f'MSE = {mse}, psnr = {psnr}, msre = {msre}, total frames {np.sum(Nt)}, detections per pixel {np.sum(np.multiply(Nt, p))/num_pixels}')
        print(f'detections = {detections}')
        print(f'measurements per pixel = {np.sum(Nt)/num_pixels}')

        # the improvement level is set by how close to zero a probability gets since MSRE emphasizes 
        # low intensities. 
        p = np.random.uniform(size=(100,100))*max_prob + 0.01 # add 0.01 to avoid nearly p =0 
        msre, Nt_opt, improve_factor = optimal_msre(p, 1)
        improve_arr = np.append(improve_arr, improve_factor)

    fig,ax=plt.subplots()
    ax.plot(mps, msre_arr)
    ax.set_xlabel('Max probability')
    ax.set_ylabel('MSRE')

    fig,ax=plt.subplots()
    ax.plot(mps, improve_arr)
    ax.set_xlabel('Max probability')
    ax.set_ylabel('MSRE Improvement')


    for optimize in ['MSE', 'MSRE', 'MSE_detections']:
        print('---'*40)
        print('---'*40)
        print(f'Optimize: {optimize}')

        for max_prob in [1.0, 1/8]:

            print('---'*40)
            print(f'Maximum probability of {max_prob}')

            # create a random image 
            p = np.random.uniform(size=(100,100))*max_prob

            Nt = N*np.ones(np.shape(p))
            num_pixels = np.prod(np.shape(p))
            
            print('Equal frame weighting')
            detections = np.sum(np.multiply(Nt, p))
            mse, psnr, img_std, msre = img_mse(p, None, Nt, mode='probability')
            print(f'MSE = {mse}, psnr = {psnr}, msre = {msre}, total frames {np.sum(Nt)}, detections per pixel {np.sum(np.multiply(Nt, p))/num_pixels}')
            print(f'detections = {detections}')
            print(f'measurements per pixel = {np.sum(Nt)/num_pixels}')
            
            if optimize == 'MSE': # optimal weighting for MSE 
                p_var = np.multiply(p, (1-p))
                N_wt = p_var**0.5/np.sum(p_var**0.5) # see Blue notebook page 2022/11/18 - optimal MSE 
                mse_optimal_calc, N_wt, improve_factor = optimal_mse_measurements(p, 100)
            elif optimize == 'MSRE': # optimal weighting for MSRE 
                N_wt = ((1-p)/p)**0.5/np.sum(((1-p)/p)**0.5) # see Blue notebook page 2023/2/23 - optimal MSRE 
                mse_optimal_calc, N_wt, improve_factor = optimal_msre(p, 100)
            elif optimize == 'MSRE_detections': # optimal measurement weighting when detections are constrained 
                N_wt = ((1-p)**0.5/p)/np.sum(((1-p)**0.5/p))
                mse_optimal_calc, N_wt, improve_factor = optimal_mse(p, 100)
            elif optimize == 'MSE_detections': # optimal measurement weighting when detections are constrained 
                mse_optimal_calc, N_wt, improve_factor = optimal_mse(p, 100)

            Nt_opt = N*N_wt*len(p.ravel())
            detections = np.sum(np.multiply(Nt_opt, p))
            mse, psnr, img_std, msre = img_mse(p, None, Nt_opt, mode='probability')
            print('Optimal frame weighting')
            print(f'MSE = {mse}, psnr = {psnr}, msre = {msre}, total frames {np.sum(Nt)}, detections per pixel {np.sum(np.multiply(Nt_opt, p))/num_pixels}')
            print(f'detections = {detections}')
            print(f'measurements per pixel = {np.sum(Nt)/num_pixels}')

            if (optimize != 'MSRE_detections') and (optimize != 'MSE_detections'):
                if optimize == 'MSE':
                    metric = mse
                elif optimize == 'MSRE':
                    metric = msre
                search_metric = search_better(p, N, metric_type=optimize, best_metric=metric, searches = 1000)
                print(f'Search metric: {search_metric}')

            if optimize == 'MSRE_detections':
                detection_scale = np.sum(np.multiply(Nt, p))/np.sum(np.multiply(Nt_opt, p))
                detections = np.sum(np.multiply(Nt_opt*detection_scale, p))
                mse, psnr, img_std, msre = img_mse(p, None, Nt_opt*detection_scale, mode='probability')
                print('Optimal frame weighting with detection scaling')
                print(f'MSE = {mse}, psnr = {psnr}, msre = {msre}, total frames {np.sum(Nt)}, detections per pixel {detections/num_pixels}')
                print(f'detections = {detections}')
                print(f'measurements per pixel = {np.sum(Nt)/num_pixels}')

            # print(f'Optimal weighting \n {Nt_opt} \n for a probability matrix of \n {p}')

            fig,ax=plt.subplots(2,1)
            fig.suptitle(f'max p = {max_prob}, optimize = {optimize}')
            ax[0].semilogy(p.ravel(), N_wt.ravel(), marker='*', linestyle='none')
            ax[0].set_xlabel('binary rate')
            ax[0].set_ylabel('Measurements')

            ax[1].semilogy(p.ravel(), p.ravel()*N_wt.ravel(), marker='*', linestyle='none')
            ax[1].set_xlabel('binary rate')
            ax[1].set_ylabel('Detections')
            # determine p for the minimum measurements 

        if 0:

            snr_orig = snr_exp_vec(Nt, -np.log(1-p))    
            snr_wt_mat = snr_exp_vec(Nt_opt, -np.log(1-p))

            # MSE weighting BUT set a Hmin so that low probability pixels have a min SNR 
            p_min_val = 0.05
            p_min = copy.deepcopy(p)
            p_min[p<p_min_val] = p_min_val
            p_frac = np.divide(p_min, (1-p_min))
            N_wt =p_frac**0.5/np.sum(p_frac**0.5)
            Nt_opt_minp = N*N_wt*len(p.ravel())

            mse, psnr, img_std, msre = img_mse(p, None, Nt_opt_minp, mode='probability')
            print(f'MSE = {mse}, psnr = {psnr}, msre = {msre}, total frames {np.sum(Nt)}, measurements per pixel {np.sum(np.multiply(Nt_opt, p))/num_pixels}')
            snr_wt_min = snr_exp_vec(Nt_opt_minp, -np.log(1-p))    

            fig = two_by_two(p,[Nt, Nt_opt, Nt_opt_minp], lbl=['uniform', 'weighted', 'weighted-minp'])
            my_savefig(fig, figure_dir, f'optimize_MSE_maxp_{max_prob}')

            # MSE weighting BUT set a Hmin so that certain pixels don't get too many frames 
            p_max_val = 0.7
            p_max = copy.deepcopy(p)
            p_max[p>p_max_val] = p_max_val
            p_frac = np.divide(p_max, (1-p_max))
            N_wt = p_frac**0.5/np.sum(p_frac**0.5)
            Nt_opt_maxp = N*N_wt*len(p.ravel())

            mse, psnr, img_std, msre = img_mse(p, None, Nt_opt_maxp, mode='probability')
            print(f'MSE = {mse}, psnr = {psnr}, msre = {msre}, total frames {np.sum(Nt)}')
            snr_wt_max = snr_exp_vec(Nt_opt_maxp, -np.log(1-p))    

            fig, ax = plt.subplots()
            ax.hist(snr_orig.ravel(), bins=100, histtype=u'step',label='uniform frames') # ravel to create 1D array
            ax.hist(snr_wt_mat.ravel(), bins=100, histtype=u'step',label='MSE weighting') # ravel to create 1D array
            ax.hist(snr_wt_min.ravel(), bins=100, histtype=u'step',label='Min SNR')
            ax.hist(snr_wt_max.ravel(), bins=100, histtype=u'step',label='Min SNR')

            ax.set_xlabel('SNR')
            ax.set_ylabel('Occurences')    
            ax.legend()

            fig2, ax2 = plt.subplots()
            ax2.semilogy(p.ravel(), Nt_opt.ravel(), label='Best MSE', linestyle='none', marker='*')
            ax2.semilogy(p.ravel(), Nt_opt_maxp.ravel(), label='Best MSE w/ max-p', linestyle='none', marker='*')

            ax2.set_xlabel('p')
            ax2.set_ylabel('N')
            ax2.legend()

            print('-'*40)
            # Equalize SNR 
            mse, psnr, img_std, msre = img_mse(p, None, Nt, mode='probability')
            print(f'MSE = {mse}, psnr = {psnr}, total frames {np.sum(Nt)}')

            h = -np.log(1-p)
            numerator = (np.exp(h) - 1)/h**2
            N_wt = N*numerator/np.sum(numerator)*len(p.ravel())
            fig2, ax2 = plt.subplots()
            ax2.semilogy(p.ravel(), N_wt.ravel(), label='Equal SNR', linestyle='none', marker='*')

            mse, psnr, img_std, msre = img_mse(p, None, N_wt, mode='probability')
            print(f'MSE = {mse}, psnr = {psnr}, total frames {np.sum(N_wt)}')
            snr_wt = snr_exp_vec(N_wt, -np.log(1-p))    

            fig = two_by_two(p, [Nt, N_wt], lbl=['uniform', 'weighted'])
            my_savefig(fig, figure_dir, f'equalize_SNR_{max_prob}')

            # equalize SNR with a frame limit of 8*N
            factor = 8
            N_wt_limit = copy.deepcopy(N_wt)
            N_wt_limit[N_wt_limit > (N*factor)] = (N*factor)
            N_wt_limit = N_wt_limit * np.sum(N_wt)/np.sum(N_wt_limit) # due to this renormalization the limit is not actually 8! 
            ax2.semilogy(p.ravel(), N_wt_limit.ravel(), label='Equal SNR with limit', linestyle='none', marker='*')

            ax2.set_xlabel('p')
            ax2.set_ylabel('N')
            ax2.legend()

            mse, psnr, img_std, msre = img_mse(p, None, N_wt_limit, mode='probability')    
            print(f'MSE = {mse}, psnr = {psnr}, total frames {np.sum(N_wt_limit)}')
            snr_limit = snr_exp_vec(N_wt_limit, -np.log(1-p))    

            fig, ax = plt.subplots()
            ax.hist(snr_orig.ravel(), bins=100, histtype=u'step',label='uniform frames') # ravel to create 1D array
            ax.hist(snr_wt.ravel(), bins=100, histtype=u'step',label='SNR equal') # ravel to create 1D array
            ax.hist(snr_limit.ravel(), bins=100, histtype=u'step',label='frame limit') # ravel to create 1D array

            ax.set_xlabel('SNR')
            ax.set_ylabel('Occurences')    
            ax.legend()

def det_per_pixel(img, meas):
    """
    determine the detection per pixel for a given image matrix (binary rate) 
    and a matrix of measurements per pixel 
    """
    det_per_pix = np.multiply(img, meas)
    det_per_pix = np.average(det_per_pix.ravel())
    meas_per_pix = np.average(meas.ravel())
    return det_per_pix, meas_per_pix 

w = 100
h = 100
optimal_ssim = True
PLT = False
if optimal_ssim:

    res = {}
    metrics = ['img', 'detections', 'distribution', 'ssim', 'msre',
                'mse', 'mse_img', 'meas_per_pix', 'det_per_pix',
                'ssim_intensity', 'msre_intensity',
                'mse_intensity',
                'mse_det_perc', 'ssim_det_perc']
    for m in metrics:
        res[m] = []

    bsds_csv_dir = '/Users/koer2434/My Drive/UST/research/bernoulli_imaging/msi_summary_figures'
    imgs = pd.read_csv(os.path.join(bsds_csv_dir, 'teaser_image_metrics.csv'))
    imgs = np.unique(imgs['img_name'])
    # use random image actual images 
    img_name = 'random'
    ref_img = np.random.uniform(size=(w,h)) + 0.01

    # use actual images     
    full_dir = os.path.join(config['bsds_dir'], 'test')
    #file_list = img_files_in_dir(full_dir, '.jpg')
    file_list = []
    for img in imgs:
        file_list.append(os.path.join(full_dir, str(img)) + '.jpg')

#    for img_name_long in file_list[0:2]:
    for img_name_long in file_list:

        img_name = os.path.split(img_name_long)[1]
        print(f'Evaluating: {img_name} ----------------------- ')
        ref_img, roi_x, roi_y = open_img(img_name, full_dir, 
            roi_x=(128,232), roi_y=(128,400), Thumbnail=False)

        ref_img[ref_img<0.01] = 0.01
        ref_img[ref_img>0.99] = 0.99
        ref_img_int = -np.log(1-ref_img)
        ref_img_scale = np.max(ref_img_int)
        ref_img_int = ref_img_int / ref_img_scale 

        # for each configuration calculate / track 
        # ssim, mse, mse_img, msre, distribution_method, meas_per_pix, det_per_pix, image_name   

        # cycle through five distribution methods: uniform, mse_meas, mse_detections, msre_meas, msre_detections 

    #   for measurements in np.arange(1,1000):   
        for detections in [2, 5, 12, 30, 100]:  
            if PLT:
                fig,ax=plt.subplots(2,2)
                fig,axp=plt.subplots(2,2)

            # uniform measurements 
            meas_uniform = np.ones(np.shape(ref_img))
            # calculate average detections_per_pixel and average measurements per pixel 
            d_per_pix_baseline, m_per_pix_baseline = det_per_pixel(ref_img, meas_uniform)

            # scale the uniform measurements to match the target detections 
            meas_uniform = detections/d_per_pix_baseline*meas_uniform

            for axs, distribution in zip([(0,0),(0,1),(1,0),(1,1)], 
                ['uniform', 'mse_meas', 'mse_det', 'msre_det','msre_meas']):

                if distribution=='uniform':
                    meas = meas_uniform 
                elif distribution == 'mse_meas':
                    opt_mse, meas, improve_factor = optimal_mse_measurements(ref_img, detections)
                elif distribution == 'mse_det':
                    opt_mse, meas, improve_factor = optimal_mse(ref_img, detections)
                elif distribution == 'msre_meas':
                    opt_msre, meas, improve_factor = optimal_msre_measurements(ref_img, detections)
                elif distribution == 'msre_det':
                    opt_msre, meas, improve_factor = optimal_msre(ref_img, detections)

                # generate a noisey image based on the measurement matrix 
                n_img = noisey_image(ref_img, meas)
                ssim_t = ssim(ref_img, n_img)
                mse, psnr, img_std, msre = img_mse(ref_img, None, meas, mode='probability')
                res['distribution'].append(distribution)

                # save uniform metrics so we can search for the detections needed to match 
                if distribution=='uniform':
                    uniform_metrics = {'mse': mse, 'psnr': psnr, 'img_std': img_std, 'msre': msre}
                    res['mse_det_perc'].append(0)
                    res['ssim_det_perc'].append(0)

                res['img'].append(img_name)
                res['detections'].append(detections)
                res['ssim'].append(ssim_t)
                res['mse'].append(mse)
                res['msre'].append(msre)
                mse_img = np.average( (n_img-ref_img)**2)
                res['mse_img'].append(mse_img) 
                # mse_img and mse are calculated slightly differently but converge around 15 measurements 
                d_per_pix, m_per_pix = det_per_pixel(ref_img, meas)
                res['meas_per_pix'].append(m_per_pix)
                res['det_per_pix'].append(d_per_pix)

                # scale uniform measurements until the uniform SSIM matches the current SSIM (ssim_t)
                # search 
                if distribution != 'uniform':
                    det_factor = np.linspace(-30,20,100)
                    ssim_search = []
                    mse_search = []
                    for d in det_factor:
                        meas_search = meas_uniform*(1+d/100)
                        n_img_search = noisey_image(ref_img, meas_search)
                        ssim_t = ssim(ref_img, n_img_search)
                        mse, psnr, img_std, msre = img_mse(ref_img, None, meas_search, mode='probability')
                        ssim_search.append(ssim_t)
                        mse_search.append(mse)
                    idx_ssim = np.argmin(np.abs(np.array(ssim_search) - res['ssim'][-1]))
                    idx_mse = np.argmin(np.abs(np.array(mse_search) - res['mse'][-1]))

                    # calculate new detections_per_pixel 
                    d_factor_ssim = det_factor[idx_ssim]
                    d_per_pix, m_per_pix = det_per_pixel(ref_img, meas_uniform*(1+d_factor_ssim/100))
                    res['ssim_det_perc'].append(((res['det_per_pix'][-1] - d_per_pix)/res['det_per_pix'][-1])*100)   
                    d_factor_mse = det_factor[idx_mse]
                    d_per_pix, m_per_pix = det_per_pixel(ref_img, meas_uniform*(1+d_factor_mse/100))
                    res['mse_det_perc'].append( ((res['det_per_pix'][-1] - d_per_pix)/res['det_per_pix'][-1])*100)

                # calculate metrics on the intensity image. But first need to transform. To do so, find the maximum 
                # non-saturated value and replace saturated values with that. 
                max_below1 = np.max(n_img[n_img<1])
                n_img_tmp = n_img
                n_img_tmp[n_img==1] = max_below1
                # ref_img_scale is the maximum value in the intensity image and is used to normalize the intensity image to [0-1] above
                n_img_intensity = -np.log(1-n_img_tmp)/ref_img_scale 
                res['mse_intensity'].append(np.average( (n_img_intensity-ref_img_int)**2)) 
                res['ssim_intensity'].append(ssim(n_img_intensity, ref_img_int)) 
                res['msre_intensity'].append(np.average( (n_img_intensity-ref_img_int)**2/ref_img_int**2)) 
                if PLT:
                    ax[axs].imshow((n_img_intensity)**(1/2.2), cmap='gray')
                    ax[axs].title.set_text(f'Int. gamma 2.2: {distribution}')
                    axp[axs].imshow(n_img, cmap='gray')
                    axp[axs].title.set_text(f'Prob: {distribution}')

    df = pd.DataFrame(res)
    pd.set_option("display.precision", 3)
    pd.options.display.float_format = '{:.2E}'.format
    # df = df.drop('img', axis=1)
    print(df[(df.distribution!='msre_meas')])

    # create a summary LaTex table 
    df_sub = df[['detections', 'distribution', 'ssim', 'mse', 'ssim_det_perc', 'mse_det_perc']]
    df_sub = df_sub[(df_sub['distribution'] == 'uniform') | (df_sub['distribution'] == 'mse_det') | (df_sub['distribution'] == 'msre_det')]

    idx = df_sub['detections'] == 2
    for d in [5,12,30,100]:
        idx = idx | (df_sub['detections'] == d)
    df_sub = df_sub[idx]
    print(df_sub)

    def format_float(val):
        try:
            if abs(val) < 1e-2:
                return "{:.3e}".format(val)
            else:
                return "{:.3f}".format(val)
        except:
            return val

    def format_det(val):
        try:
            return "{:.1f}".format(val)
        except Exception as e:
            print(f'{e} Format_det error')
            return val

    def rename_dist(val):
        if val=='mse_det':
            return 'MSE'
        elif val == 'msre_det':
            return 'MSRE'
        else:
            return val

    # Apply the custom formatting function to the DataFrame
    # first to the individual columns 
    df_sub['detections'] = df_sub['detections'].apply(format_det)
    df_formatted = df_sub.applymap(format_float)
    df_formatted = df_formatted.rename({'mse_det': 'MSE', 'ssim': 'SSIM', 'msre_det': 'MSRE'})
    df_formatted['distribution'] = df_formatted['distribution'].apply(rename_dist)

    print(df_formatted.to_latex(escape=False, header=True, index=False))
    print('---'*50)

    # df_formatted = df_formatted.drop(columns='msre')
    pivot = pd.pivot_table(df_sub, columns=['distribution','detections'], aggfunc=['mean','std'])
    print(pivot.to_latex(escape=False, header=True, index=False))

    print('Pivot table: MSE detections')
    print(pivot['mean', 'mse_det'])

    print('Pivot table: uniform')
    print(pivot['mean', 'uniform'])

bsds_ssim = False
if bsds_ssim:
    res = {'ssim': [], 'mse': [], 'n_meas': [], 'det_pix': [], 'avg_prob': [], 
           'ssim_intensity': [], 'mse_intensity':[],
           'improve_factor':[], 'ssim_opt': [], 'mse_opt': [], 'det_pix_opt': [],
           'img_name': [], 'threshold': []}
    from bernoulli_inhibit import mse, msre, to_intensity
    from create_binary_images import cfg
    from create_binary_images import main as binary_images_main
    # extract IQ versus number of photons and average probability 

    # load image 
    full_dir = os.path.join(config['bsds_dir'], 'test')
    file_list = img_files_in_dir(full_dir, '.jpg')

    cfg['max_frames'] = 1000
    th = 1

    # TODO - 
    # 2) allow for specifying a file name to create binary images 
    # 3) incorporate optimal measurement distribution 
    # 4) make it a function to run on supercomputer 
    # 5) save results, including image name and threshold to a CSV in a results dir 

    # cycle through binary images in the BSDS directory
    for img_name in file_list:

        img, roi_x, roi_y = open_img(os.path.split(img_name)[1], full_dir, 
            roi_x=(128,232), roi_y=(128,400), Thumbnail=False)

        # check if the Numpy file already exists
        numpy_file = img_name.replace('.jpg', f'_bernoulli_thresh(None,_{th})_maxframes{cfg["max_frames"]}.npy')
        if os.path.isfile(numpy_file):
            print(f'binary file: {numpy_file} already exists. Wont recreate')
            binary_img_filename = numpy_file
        else: # create the binary file 
            saved_image, saved_image_names = binary_images_main(figure_dir=full_dir, img_file_range=img_name, 
                thresholds=[(None, th)], max_imgs=1)
            binary_img_filename = os.path.join(full_dir, saved_image_names[0] + '.npy')
        binary_img_full = np.load(binary_img_filename)

        # modify the reference to match the generated threshold 
        img = img/np.mean(img)/th
        # img is proportional to intensity
        y_ref = 1-np.exp(-img) # probability image 

        # vary number of photons 
        for N in np.concatenate( (np.arange(2,100), np.arange(100,1000,10))):
            binary_img = np.average(binary_img_full[:,:,0:N], 2)

            # convert to intensity 
            intensity_img = -np.log(1-binary_img)
            intensity_img[binary_img==0] = 0
            intensity_img[binary_img==1] = -np.log(1-(N-1)/N)

            # get IQ metrics on binary rate images
            res['ssim'].append(ssim(y_ref, binary_img))
            res['mse'].append(mse(y_ref, binary_img))
            res['n_meas'].append(N)
            res['det_pix'].append(np.average(binary_img)*N)
            res['avg_prob'].append(np.average(y_ref))

            # get IQ metrics on intensity images
            res['ssim_intensity'].append(ssim(img, intensity_img))
            res['mse_intensity'].append(mse(img, intensity_img))

            # find the optimal measurement distribtion 
            o_mse, nt, improve_factor = optimal_mse(y_ref, res['det_pix'][-1])
            nt_round = np.rint(nt).astype(int)
            nt_round[nt_round<1] = 1
            optimal_binary_img = np.zeros(np.shape(binary_img))
            optimal_detections = np.zeros(np.shape(binary_img))

            # would like to do this without a loop but not sure how 
            for i in range(np.shape(binary_img_full)[0]):
                for j in range(np.shape(binary_img_full)[1]):
                    optimal_binary_img[i,j] = np.average(binary_img_full[i,j,0:nt_round[i,j]])
                    optimal_detections[i,j] = np.sum(binary_img_full[i,j,0:nt_round[i,j]])

            res['ssim_opt'].append(ssim(y_ref, optimal_binary_img))
            res['mse_opt'].append(mse(y_ref, optimal_binary_img))
            res['det_pix_opt'].append(np.average(optimal_detections))

            res['img_name'].append(img_name)
            res['threshold'].append(th)
            res['improve_factor'].append(improve_factor)

        df = pd.DataFrame.from_dict(res)
        csv_file_name = os.path.split(img_name)[1].replace('.jpg','') # to_csv automatically appends .csv
        df.to_csv(os.path.join(results_dir, f'output_{csv_file_name}.csv'))

        fig,ax=plt.subplots()
        ax.plot(res['det_pix'], res['ssim'], linestyle='none', marker='o', label='Binary rate')
        ax.plot(res['det_pix_opt'], res['ssim_opt'], linestyle='none', marker='.', label='Binary Rate: Optimal')
        ax.plot(res['det_pix'], res['ssim_intensity'], linestyle='none', marker='x', label='Intensity')
        ax.legend()

        fig,ax=plt.subplots()
        ax.plot(res['det_pix'], res['mse'], linestyle='none', marker='o', label='Binary rate')
        ax.plot(res['det_pix_opt'], res['mse_opt'], linestyle='none', marker='.', label='Binary Rate: Optimal')
        ax.plot(res['det_pix'], res['mse_intensity'], linestyle='none', marker='x', label='Intensity')
        ax.legend()

        plt.close('all')
