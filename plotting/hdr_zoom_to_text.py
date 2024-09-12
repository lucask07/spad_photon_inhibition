"""
Lucas Koerner
2023/10/17 

Create images for the teaser figure (both HDR with exposure bracket and edges) 
load the binary rate numpy and the HED edge png 
(invert the edge map so that edges are dark and no edges are white)

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.ticker as mticker
import itertools 
import imageio as iio
import pickle as pkl
import matplotlib as mpl
from utils import my_savefig
from mpl_toolkits.axes_grid.inset_locator import inset_axes

from plotting.plot_tools import bracket_measures, load_irs

np.seterr(divide='ignore') # prevent warnings when calculating divide by zero and log(0)
np.seterr(invalid='ignore')

plt.ion()

mpl.rcParams['lines.markersize']=3

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


def fstr(template):
    return eval(f"f'{template}'")


def read_bdry(fname):
    df = pd.read_csv(fname, delimiter='\s+', header=None, index_col=False)
    return df 


img_name = 'vatican_road_8k_smallcrop'
zx = [680, 740]
zy = [290, 350]
zloc = 4
h = 35
w = 35

if 0:
    img_name = 'vulture_hide_8k_crop3'
    zx = [180, 380]
    zy = [10, 150]
    zloc = 8 # lower center 
    h = 45
    w = 45

img_name = 'workshop_4k_crop'
zx = [70, 200]
zy = [10, 150]
zloc = 3 # lower center 
h = 55
w = 55

pps = [1,2, 5, 8, 12, 16, 20, 24, 30, 80, 200] 
# pps = [1,2,5,12,30,80,200]
kernel = 'flip_laplacian'
t = '12'
il = 32
irs = load_irs(df=None, row=None, kernel=kernel, length=il, thresh=t, img_name=img_name)

ref = irs[1].load_img()
ref = 1-np.exp(-ref) # convert to binary rate 

data_dir = f'/home/lkoerner/lkoerner/bernoulli_inhibit/tests_probability_images/tests_output_bracket/{img_name}/{kernel}/thresh{t}/length{il}/'
file_name = 'img_{}phpp_{}'
# output directory 
figure_dir = '/home/lkoerner/lkoerner/bernoulli_inhibit/summary_figures'

masks = ['orig', 'nomask', 'mask']
titles = {'orig': 'Reference', 'nomask':'Conventional', 'mask': 'Proposed'}
gamma = 0.4


fig_ref,ax_ref=plt.subplots(figsize=(1.08, 1.4)) #height_ratios needs mpl 3.6.0
ref_inset=inset_axes(ax_ref, height=f"{h}%", width=f"{w}%", loc=zloc)
ax_ref.imshow(ref**gamma, cmap='gray')
ref_inset.imshow(ref[zx[0]:zx[1],zy[0]:zy[1]]**gamma, cmap='gray')

ref_inset.xaxis.set_visible(False)
ref_inset.yaxis.set_visible(False)
ref_inset.patch.set_edgecolor('white')
ref_inset.patch.set_linewidth(0.5)
ax_ref.axis('off')
ax_ref.xaxis.set_visible(False)
ax_ref.yaxis.set_visible(False)
ax_ref.patch.set_visible(False)
ax_ref.axis('off')
        
plt.tight_layout(pad=1.00)
my_savefig(fig_ref, figure_dir, f'hdr_img{img_name}_ref')

for pp in pps:
    fig,ax=plt.subplots(1,3, figsize=(3.25, 1.4)) #height_ratios needs mpl 3.6.0
    axinset = []
    for axi in ax:
        axinset.append(inset_axes(axi, height=f"{h}%", width=f"{w}%", loc=zloc)) 
    
    for idx,mask in enumerate(masks):
        if mask == 'orig':
            ax[idx].imshow(ref**gamma, cmap='gray')
            # ax[idx].set_title(titles[mask], fontsize=7)
            axinset[idx].imshow(ref[zx[0]:zx[1],zy[0]:zy[1]]**gamma, cmap='gray')
        else: 
            t_fname = os.path.join(data_dir, file_name.format(pp, mask))
            print(t_fname)
            img = np.load(t_fname, allow_pickle=True)

            ax[idx].imshow(img**gamma, cmap='gray')
            # ax[idx].set_title(titles[mask], fontsize=7)
            axinset[idx].imshow(img[zx[0]:zx[1],zy[0]:zy[1]]**gamma, cmap='gray')
        
        axinset[idx].xaxis.set_visible(False)
        axinset[idx].yaxis.set_visible(False)
        axinset[idx].patch.set_edgecolor('white')
        axinset[idx].patch.set_linewidth(0.5)
        ax[idx].axis('off')
        ax[idx].xaxis.set_visible(False)
        ax[idx].yaxis.set_visible(False)
        ax[idx].patch.set_visible(False)
        ax[idx].axis('off')
        
    plt.tight_layout(pad=1.00)
    my_savefig(fig, figure_dir, f'hdr_img{img_name}_{pp}ppp_3x1')


# plot SSIM versus detections 
fig,ax = plt.subplots(figsize=(2.5,2))

m = 'mask'
ax.semilogx(irs.metrics[m]['photons_pp'], irs.metrics[m]['ssim'], linestyle='-',marker='o', label='proposed')
m = 'no_mask'
ax.semilogx(irs.metrics[m]['photons_pp'], irs.metrics[m]['ssim'], linestyle='--',marker='x', label='no inhibition')
ax.set_xlabel('D/pix.')
ax.set_ylabel('SSIM')
ax.legend(loc=4)
ax.set_xlim([3, 50])
ax.set_ylim([0.4, 1])
plt.tight_layout()
my_savefig(fig, figure_dir, 'ssim_workshop_small')

fig,ax = plt.subplots(figsize=(2.5,1.6))

with open(os.path.join(data_dir, f'measure_dist_data_{img_name}.pkl'), 'rb') as f:
    meas = pkl.load(f)
x,yl = bracket_measures(ax, meas, subsample=20)
plt.tight_layout()
my_savefig(fig, figure_dir, 'measures_workshop_small')
