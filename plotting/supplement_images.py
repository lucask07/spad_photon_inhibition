"""
Lucas Koerner
2023/11/21 

Create images, inhibition patterns, and iq vs. detections for the supplement
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pickle as pkl
import pandas as pd 
import matplotlib.ticker as mticker
import itertools 
import imageio as iio

import inhibition_captures # required to load the pickles 
from plotting.plot_tools import bracket_measures, load_irs

from utils import my_savefig

np.seterr(divide='ignore') # prevent warnings when calculating divide by zero and log(0)
np.seterr(invalid='ignore')
fig_size = (8,6)

plt.ion()

mpl.rcParams['lines.markersize']=1

SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def find_closest(metrics, metric, value):
    mask_idx = np.argmin(np.abs(metrics['mask'][metric]-value))
    nomask_idx = np.argmin(np.abs(metrics['no_mask'][metric]-value))

    return mask_idx, nomask_idx


def fstr(template):
    return eval(f"f'{template}'")


def read_bdry(fname):
    df = pd.read_csv(fname, delimiter='\s+', header=None, index_col=False)
    return df 

gamma = 0.4
df = pd.read_csv('bracket_folders.csv') 
img_names = np.unique(df['img_ids'])

pps = [2, 5, 12, 30] 
pps = [1,2,5,8,12,30]
# pps = [1,2,5,12,30,80,200]
kernel = 'flip_laplacian'
t = '12'
# il = 16 # was 
il = 32 

file_name = 'img_{}phpp_{}'
# output directory 
figure_dir = '/home/lkoerner/lkoerner/bernoulli_inhibit/summary_figures/supplement'
masks = ['nomask', 'mask']
res = {'img_name':[], 'pps':[], 'mask_nomask':[], 'ssim':[], 'msre':[], 'mse':[], 'actual_pp':[]}

img_name = [393035, 179084, 145079, 130066, 223004]
# img_name = ['workshop_4k_crop']
# img_names = list(img_names)
img_names = img_name # Temporary - just one name for debug 

fig_sz = (3,3)

for img_name in img_names:
    plt.close('all')
    data_dir = f'/home/lkoerner/lkoerner/bernoulli_inhibit/tests_probability_images/tests_output_bracket/{img_name}/{kernel}/thresh{t}/length{il}/'
    # data_dir = f'/home/lkoerner/lkoerner/bernoulli_inhibit/tests_probability_images/tests_output_hdr_bugfix/{img_name}/{kernel}/thresh{t}/length{il}/'
    irs = load_irs(row=None, kernel=kernel,length=il, thresh=t, img_name=img_name, data_dir=data_dir)

    #plot SSIM versus detections 
    fig,ax = plt.subplots(figsize=(2.2,1.8))
    m = 'mask'
    ax.semilogx(irs.metrics[m]['photons_pp'], irs.metrics[m]['ssim'], linestyle='-',marker='o', label='proposed')
    m = 'no_mask'
    ax.semilogx(irs.metrics[m]['photons_pp'], irs.metrics[m]['ssim'], linestyle='--',marker='x', label='no inhibition')
    ax.set_xlabel('D/pix.')
    ax.set_ylabel('SSIM')
    ax.legend()
    ax.set_xlim([5, 500])
    ax.set_ylim([0.3, 1]) 
    plt.tight_layout()
    my_savefig(fig, figure_dir, f'ssim_{img_name}_il{il}')

    #plot MSE versus detections 
    fig,ax = plt.subplots(figsize=(2.2, 1.8))
    m = 'mask'
    ax.loglog(irs.metrics[m]['photons_pp'], irs.metrics[m]['mse'], linestyle='-',marker='o', label='proposed')
    m = 'no_mask'
    ax.loglog(irs.metrics[m]['photons_pp'], irs.metrics[m]['mse'], linestyle='--',marker='x', label='no inhibition')
    ax.set_xlabel('D/pix.')
    ax.set_ylabel('MSE')
    ax.legend()
    ax.set_xlim([5, 500])
    ax.set_ylim([1e-4, 0.1]) 
    plt.tight_layout()
    my_savefig(fig, figure_dir, f'mse_{img_name}_il{il}')

    # plot measures for exposure level 
    fig, ax = plt.subplots(figsize=(2.4,1.6))

    with open(os.path.join(data_dir, f'measure_dist_data_{img_name}.pkl'), 'rb') as f:
        meas = pkl.load(f)
    if type(img_name) == int:
        x,yl = bracket_measures(ax, meas, subsample=1, percent=True) # smaller images so subsample of 1
    else:
        x,yl = bracket_measures(ax, meas, subsample=20, percent=True) # large image so subsample otherwise the Lowess fit takes too long

    plt.tight_layout()
    my_savefig(fig, figure_dir, f'measures_{img_name}_small_il{il}')

    ref_img = irs[0].load_img()
    img_size = np.shape(ref_img)
    # create images of the inhibition patterns %measurements 
    for idx,m in enumerate(meas):
        fig,ax=plt.subplots(figsize=fig_sz) #height_ratios needs mpl 3.6.0
        # cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        imshow_img = ax.imshow(np.reshape((1000-m['unmasked'])/1000*100, img_size)) # ploting the inhibition fraction 
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(imshow_img, cax=cax, orientation='vertical')
        cax.xaxis.set_visible(False)

        #plt.tight_layout()
        my_savefig(fig, figure_dir, f'inhibition_fraction_img{img_name}_ppp{idx}_il{il}', tight=False)

    for pp in pps:
        closest_idx = find_closest(irs.metrics, 'photons_pp', pp)
        for idx,mask in enumerate(masks):
            fig,ax=plt.subplots(figsize=fig_sz) #height_ratios needs mpl 3.6.0
            t_fname = os.path.join(data_dir, file_name.format(pp, mask))
            print(t_fname)

            img = np.load(t_fname, allow_pickle=True)

            ax.imshow(img**gamma, cmap='gray')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            # log metrics 
            res['img_name'].append(img_name)
            res['pps'].append(pp)
            res['mask_nomask'].append(mask)

            if mask=='mask':
                cidx = 0
                mask_key = 'mask'
            else:
                cidx = 1
                mask_key = 'no_mask'
            res['ssim'].append(irs.metrics[mask_key]['ssim'][closest_idx[cidx]])
            res['msre'].append(irs.metrics[mask_key]['msre'][closest_idx[cidx]])
            res['mse'].append(irs.metrics[mask_key]['mse'][closest_idx[cidx]])
            res['actual_pp'].append(irs.metrics[mask_key]['photons_pp'][closest_idx[cidx]])
            plt.tight_layout()
            my_savefig(fig, figure_dir, f'new_teaser_img{img_name}_{pp}ppp_{mask}_il{il}')

    fig,ax=plt.subplots(figsize=fig_sz) #height_ratios needs mpl 3.6.0
    ax.imshow(ref_img**gamma, cmap='gray')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plt.tight_layout()
    my_savefig(fig, figure_dir, f'new_teaser_img{img_name}_reference_image')

df = pd.DataFrame(res)
df.to_csv('supplement_image_metrics.csv')


EDGE = False
if EDGE:
    # now edge map 
    kernel = 'laplacian_and_avg_or_avg'
    pps = [2, 5, 12, 30] 
    t = '[[-12,12],4,16]'
    il = 16
    data_dir = f'/home/lkoerner/lkoerner/bernoulli_inhibit/tests_probability_images/tests_output/{img_name}/{kernel}/thresh{t}/length{il}/hed/png/'
    file_name = 'img_{}phpp_{}.png'

    for pp in pps:
        fig,ax=plt.subplots(2,1, figsize=(6,6)) #height_ratios needs mpl 3.6.0
        for idx,mask in enumerate(masks):
            t_fname = os.path.join(data_dir, file_name.format(pp, mask))
            print(t_fname)
            img = iio.imread(t_fname)

            # invert black and white to save printer toner
            ax[idx].imshow(255-img, cmap='gray')
            ax[idx].xaxis.set_visible(False)
            ax[idx].yaxis.set_visible(False)
            
        plt.tight_layout()
        my_savefig(fig, figure_dir, f'teaser_edge{img_name}_{pp}ppp')
