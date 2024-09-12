"""
Lucas Koerner


read eval-bdry.txt
th ? ? f-ods ? ? f-ois ap 


"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.ticker as mticker
import itertools 
import matplotlib as mpl 

from utils import my_savefig

np.seterr(divide='ignore') # prevent warnings when calculating divide by zero and log(0)
np.seterr(invalid='ignore')

plt.ion()

SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
mpl.rcParams['lines.markersize']=4 # default is 6
mpl.rcParams['lines.linewidth']=1 

PER_PIX = True

if PER_PIX:
    xlabel = {'photons_pp':'detections/pix', 'measurements_pp': 'measure/pix'}
else:
    xlabel = {'photons':'detections', 'measurements': 'measurements'}

def find_closest(metrics, metric, value):
    mask_idx = np.argmin(np.abs(metrics['mask'][metric]-value))
    nomask_idx = np.argmin(np.abs(metrics['no_mask'][metric]-value))

    return mask_idx, nomask_idx


def fstr(template):
    return eval(f"f'{template}'")


def read_bdry(fname):
    df = pd.read_csv(fname, delimiter='\s+', header=None, index_col=False)
    return df 


data_dir = '/home/lkoerner/lkoerner/hed/hed_inhibit/k{k}_t{t}_il{il}_phpp{pp}_mask{mask}/nms-eval/'
fname = 'eval_bdry.txt'
figure_dir = '/home/lkoerner/lkoerner/bernoulli_inhibit/summary_figures'

length = 20
kernels = ['flip_laplacian', 'single_pix_bright', 'neighbor8']
kernels = ['flip_laplacian',  'neighbor8', 'laplacian']
x_axis = 'photons_pp'

fig,ax=plt.subplots(2,1, figsize=(6,6), height_ratios = [2,1]) #height_ratios needs mpl 3.6.0
pps = [1,2,5,12,30,80,200]
t = 16
il = 16
masks = ['mask', 'nomask']
res = {'fois': [], 'phmask':[], 'pps':[], 'ks':[]}

for mask in masks:
    for k in kernels:
        for pp in pps:
            t_fname = os.path.join(fstr(data_dir), fname)
            print(t_fname)
            df = read_bdry(t_fname)
            res['fois'].append(df[6].iloc[0])
            res['pps'].append(pp)
            res['ks'].append(k)
            if mask == 'mask':
                mt = 1 
            else:
                mt = 0
            res['phmask'].append(mt)

mask='mask'
k = 'laplacian_and_avg_or_avg'
t = '12-12-4-16'
il = 16

for pp in pps:
    t_fname = os.path.join(fstr(data_dir), fname)
    print(t_fname)
    df = read_bdry(t_fname)
    res['fois'].append(df[6].iloc[0])
    res['pps'].append(pp)
    res['ks'].append(k)
    if mask == 'mask':
        mt = 1 
    else:
        mt = 0
    res['phmask'].append(1)

kernels = ['flip_laplacian',  'neighbor8', 'laplacian', 'laplacian_and_avg_or_avg']

kernels = ['laplacian', 'laplacian_and_avg_or_avg']

dft = pd.DataFrame(res)
# k_label = {'flip_laplacian': 'Center (8x) + surround', 'neighbor8': '3x3 average', 'single_pix_bright': 'Center pixel', 'laplacian': 'Laplacian', 'laplacian_and_avg_or_avg': 'laplacian_and_avg'}

k_label = {'flip_laplacian': 'Center(8x)+surround', 'neighbor8': '3x3 Avg.', 'single_pix_bright': 'Center pix.', 'laplacian': 'Laplacian', 'laplacian_and_avg_or_avg': 'Proposed'}

mks = itertools.cycle(('v', 's', 'x', 'o', '*')) # cycle through markers 
k_colors  = {}  # save the color for each kernel 

for m in [0,1]:
    for k in kernels:
        dfs = dft[(dft.ks==k) & (dft.phmask==m)]
        if m == 1:
            ml = 'Inhibit'
        else:
            ml = 'No inhibition'
        
        if m == 1:
            pline, = ax[0].semilogx(dfs['pps'], dfs['fois'], marker=next(mks), label= ml + ': ' + k_label[k])
            k_colors[f'{k}{m}'] = pline.get_color()
            print(f'Max {k}: {np.max(dfs["fois"])}') 
        elif m == 0 and k ==kernels[0]:
            pline, = ax[0].semilogx(dfs['pps'], dfs['fois'], marker=next(mks), label= ml)
            k_colors[f'{k}{m}'] = pline.get_color()
            print(f'Max no inhibition {k}: {np.max(dfs["fois"])}') 
            no_inhibit = dfs

ax[0].legend()
ax[0].set_xlabel('detections/pix.')
ax[0].set_ylabel('OIS F-score')
ax[0].set_xlim([0.8, 15]) # up to 12ppp 
ax[0].set_ylim([0.65, 0.76]) # up to 12ppp 
fscore_lims = ax[0].get_ylim() # to set the xlim range of the power at equal metric plot

# interpolate the no inhibition curve to assess power savings at equal IQ or edge metric  
from scipy.interpolate import PchipInterpolator
f_no = PchipInterpolator(no_inhibit['fois'], no_inhibit['pps']) # input the fscore and determine the photons per pixel 
fig2,ax2=plt.subplots(1,1)
ax2.plot(no_inhibit['fois'], no_inhibit['pps'], marker='*', linestyle='none')
f_eval = np.linspace(no_inhibit['fois'].iloc[0], no_inhibit['fois'].iloc[-1], 100)
ax2.plot(f_eval, f_no(f_eval))

# plot power savings at equal fscore 
# k_label = {'flip_laplacian': 'Center (8x) + surround', 'neighbor8': '3x3 average', 'single_pix_bright': 'Center pixel', 'laplacian': 'Laplacian', 'laplacian_and_avg_or_avg': 'laplacian_and_avg'}

mks = itertools.cycle(('v', 's', 'x', 'o', '*')) # cycle through markers 
m = next(mks) # skip the no inhibtion for the power savings plot

for m in [0,1]:
    for k in kernels:
        dfs = dft[(dft.ks==k) & (dft.phmask==m)]
        if m == 1:
            ml = 'Inhibit'
        else:
            ml = 'No inhibition'
        
        if m == 1:
            ni = f_no(dfs['fois'])
            ax[1].semilogx(dfs['fois'], 100*(dfs['pps']-ni)/ni, color=k_colors[f'{k}{m}'], linestyle='none', marker=next(mks), label= ml + ': ' + k_label[k])
        if m == 0 and k=='neighbor8':
            ax[1].semilogx([],[], marker=next(mks))

ax[1].set_ylim([-40, 30])
ax[1].set_ylabel('Power [%]')
ax[1].set_xlabel('OIS F-score')
ax[1].set_xlim(fscore_lims)
# ticks_loc = ax[1].get_xticks(minor).tolist()
# ax[1].xaxis.set_minor_locator(mticker.FixedLocator(ticks_loc))
# ax[1].set_xticklabels([f'{x}' for x in ticks_loc])
ax[1].set_xticklabels(ax[1].get_xticks(minor=True), minor=True)

plt.tight_layout()
my_savefig(fig, figure_dir, f'fscore_pp')

# just F-score, will anotate power savings in SVG 
mks = itertools.cycle(('v', 's', 'x', 'o', '*')) # cycle through markers 
k_colors  = {}  # save the color for each kernel 

fig,ax=plt.subplots(figsize=(3.25, 2.0)) #height_ratios needs mpl 3.6.0

for m in [0,1]:
    for k in kernels:
        dfs = dft[(dft.ks==k) & (dft.phmask==m)]
        if m == 1:
            ml = 'Inhibit'
        else:
            ml = 'No inhibition'
        
        if m == 1:
            pline, = ax.semilogx(dfs['pps'], dfs['fois'], marker=next(mks), label= ml + ': ' + k_label[k])
            k_colors[f'{k}{m}'] = pline.get_color()
            print(f'Max {k}: {np.max(dfs["fois"])}') 
        elif m == 0 and k ==kernels[0]:
            pline, = ax.semilogx(dfs['pps'], dfs['fois'], marker=next(mks), label= ml)
            k_colors[f'{k}{m}'] = pline.get_color()
            print(f'Max no inhibition {k}: {np.max(dfs["fois"])}') 
            no_inhibit = dfs

ax.legend()
ax.set_yticks([0.65, 0.70, 0.75])
ax.set_yticklabels(['0.65', '0.70', '0.75'])
ax.set_xlabel('detections/pix.')
ax.set_ylabel('OIS F-score')
ax.set_xlim([0.8, 15]) # up to 12ppp 
ax.set_ylim([0.65, 0.76]) # up to 12ppp

plt.tight_layout()
my_savefig(fig, figure_dir, 'fscore_no_detections')
