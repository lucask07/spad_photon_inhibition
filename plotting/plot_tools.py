"""
Lucas Koerner

"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import my_savefig
from scipy.interpolate import griddata
import pandas as pd 
import seaborn as sns 
import pickle as pkl
import statsmodels.api as sm
import inhibition_captures 

np.seterr(divide='ignore') # prevent warnings when calculating divide by zero and log(0)
np.seterr(invalid='ignore')
fig_size = (8,6)

plt.ion()

SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

PER_PIX = True

if PER_PIX:
    xlabel = {'photons_pp':'detections/pix', 'measurements_pp': 'measure/pix'}
else:
    xlabel = {'photons':'detections', 'measurements': 'measurements'}

def load_irs(row=None, kernel=None, length=None, thresh=None, img_name=None, data_dir=None):
    # get an irs from a given data-frame column  
    # an irs is a bracket result and has a list of the inhibition result for each exposure time  
    if row is not None:
        kernel = row['kernels']
        length = row['inhibit_lens']
        thresh =  row['inhibit_threshs']
        img_name = row['img_ids']
    if data_dir is None:    
        data_dir = f'tests_probability_images/tests_output_bracket/{img_name}/{kernel}/thresh{thresh}/length{length}'
    irs = inhibition_captures.load_pickle(os.path.join(data_dir, 'irs_pickle'))
    return irs 


def metric_plots(metrics, figure_dir, prob_img, unmasked, prob_img_xlabel='p', fig_name='', metrics_to_plot=['mse', 'msre', 'ssim'], fig_ax=None, lbl=None):
    """
    plot metrics in a 2x2 grid.

    One plot is MSE, MSRE, SSIM, and GMSD
    The second plot is MSE, MSRE, SSIM and measurements vs. probability

    """   
    if lbl==None:
        lbl = ['conv.', 'inhibit']
    figs = [] # to return for further processing
    axs = []
    if PER_PIX:
        vs_list = ['photons_pp', 'measurements_pp'] 
    else:
        vs_list = ['photons', 'measurements']

    for vs in vs_list: 

        fig,ax=plt.subplots(2,2, figsize=fig_size)
        ax[0,0].loglog(metrics['no_mask'][vs],metrics['no_mask'][metrics_to_plot[0]], marker='o', label=lbl[0])
        ax[0,0].loglog(metrics['mask'][vs],metrics['mask'][metrics_to_plot[0]], marker='*', label=lbl[1])
        ax[0,0].set_ylabel(metrics_to_plot[0].upper())
        ax[0,0].set_xlabel(xlabel[vs])
        ax[0,0].legend()

        ax[0,1].loglog(metrics['no_mask'][vs],metrics['no_mask'][metrics_to_plot[1]], marker='o', label=lbl[0])
        ax[0,1].loglog(metrics['mask'][vs],metrics['mask'][metrics_to_plot[1]], marker='*', label=lbl[1])
        ax[0,1].set_ylabel(metrics_to_plot[1].upper())
        ax[0,1].set_xlabel(xlabel[vs])
        ax[0,1].legend()

        ax[1,0].loglog(metrics['no_mask'][vs], metrics['no_mask'][metrics_to_plot[2]], marker='o', label=lbl[0])
        ax[1,0].loglog(metrics['mask'][vs], metrics['mask'][metrics_to_plot[2]], marker='*', label=lbl[1])
        ax[1,0].set_ylabel(metrics_to_plot[2].upper())
        ax[1,0].set_xlabel(xlabel[vs])
        ax[1,0].legend()
        if metrics_to_plot[2]=='ssim':
            ax[1,0].set_ylim([0.4, 1]) # low level of 0.4 since SSIM is a bit meaningless when very small. 

        ax[1,1].loglog(metrics['no_mask'][vs], metrics['no_mask']['gmsd'], marker='o', label=lbl[0])
        ax[1,1].loglog(metrics['mask'][vs], metrics['mask']['gmsd'], marker='*', label=lbl[1])
        ax[1,1].set_ylabel('GMSD')
        ax[1,1].set_xlabel(xlabel[vs])
        ax[1,1].legend()
        
        my_savefig(fig, figure_dir, f'summary_tests_wgmsd_vs{vs}_{fig_name}')
        figs.append(fig)
        axs.append(ax)

        fig,ax=plt.subplots(2,2, figsize=fig_size)
        ax[0,0].loglog(metrics['no_mask'][vs],metrics['no_mask'][metrics_to_plot[0]], marker='o', label=lbl[0])
        ax[0,0].loglog(metrics['mask'][vs],metrics['mask'][metrics_to_plot[0]], marker='*', label=lbl[1])
        ax[0,0].set_ylabel(metrics_to_plot[0].upper())
        ax[0,0].set_xlabel(xlabel[vs])
        ax[0,0].legend()

        ax[0,1].loglog(metrics['no_mask'][vs],metrics['no_mask'][metrics_to_plot[1]], marker='o', label=lbl[0])
        ax[0,1].loglog(metrics['mask'][vs],metrics['mask'][metrics_to_plot[1]], marker='*', label=lbl[1])
        ax[0,1].set_ylabel(metrics_to_plot[1].upper())
        ax[0,1].set_xlabel(xlabel[vs])
        ax[0,1].legend()

        ax[1,0].loglog(metrics['no_mask'][vs], metrics['no_mask'][metrics_to_plot[2]], marker='o', label=lbl[0])
        ax[1,0].loglog(metrics['mask'][vs], metrics['mask'][metrics_to_plot[2]], marker='*', label=lbl[1])
        ax[1,0].set_ylabel(metrics_to_plot[2].upper())
        ax[1,0].set_xlabel(xlabel[vs])
        ax[1,0].legend()
        if metrics_to_plot[2]=='ssim':
            ax[1,0].set_ylim([0.4, 1]) # low level of 0.4 since SSIM is a bit meaningless when very small. 

        ax[1,1].plot(prob_img.ravel(), unmasked.ravel(), linestyle = 'None', marker='*')
        ax[1,1].set_ylabel('Meas.')
        ax[1,1].set_xlabel(prob_img_xlabel)

        my_savefig(fig, figure_dir, f'summary_tests_vs{vs}_{fig_name}')
        figs.append(fig)
        axs.append(ax)

    return figs, axs

def bracket_measures(ax, meas, subsample=1, max_frame=1000, ppp = [0.1, 1, 10], percent=False,logx=False):

    ls = {0: 'dashed', 1: 'solid', 2: 'dotted', 3: 'dashdot', 4: (5,(10,3))}

    mm_y = [0,0]
    if logx:
        mm_x = [1e-4, 1e1]
    else:
        mm_x = [0, 2.5]

    for idx, m in enumerate(meas):
        x = m['x'][::subsample]
        ms = m['unmasked'][::subsample]
        print(len(x))
        print(len(ms))
        yl = sm.nonparametric.lowess(ms, x, frac=1/5, it=3)
        ax.set_xlim([np.min(x), np.max(x)*1.0])
        mm_y[0] = np.min([np.min(yl[:,1]), mm_y[0]])
        mm_y[1] = np.max([np.max(yl[:,1]*1.05), mm_y[1]])
        if not logx:
            mm_x[0] = np.min([np.min(yl[:,0]), mm_x[0]])
        # mm_x[1] = np.max([np.max(yl[:,0]*1.05), mm_x[1]])
        if percent:
            if logx: 
                ax.semilogx(yl[:,0], (1-yl[:,1]/max_frame)*100, label='ppp={0}'.format(ppp[idx]), linestyle=ls[idx])
            else:
                ax.plot(yl[:,0], (1-yl[:,1]/max_frame)*100, label='ppp={0}'.format(ppp[idx]), linestyle=ls[idx])
        else:
            if logx: 
                ax.semilogx(yl[:,0], yl[:,1], label='ppp={0}'.format(ppp[idx]), linestyle=ls[idx])
            else:
                ax.plot(yl[:,0], yl[:,1], label='ppp={0}'.format(ppp[idx]), linestyle=ls[idx])
    ax.set_ylabel('Measures')
    if percent:
        ax.set_ylabel('% Inhibtion')
    ax.set_xlabel('$\phi/T_1$')
    ax.legend(prop={'size':8})
    ax.set_xlim(mm_x)
    ax.set_ylim(mm_y)
    if percent:
        ax.set_ylim([-5,100])

    return x,yl

def metric_plots_bracket(metrics, figure_dir, flux_img, irs, fig_name='', max_frame=None, metrics_to_plot=['mse', 'msre', 'ssim'], lbl=None):
    """
    plot metrics in a 2x2 grid for an exposure bracket from an InhibitionResult list.
    The difference is that the lower-left m vs. probability plot will have as many curves as 
    exposure brackets 

    Parameters:
        metrics: dict (with keys of no_mask and mask)

    """
    if lbl==None:
        lbl = ['clk. recharge', 'inhibit']
    figs = [] # to return for further processing
    axs = [] 

    for vs in ['photons_pp', 'measurements_pp']:

        fig,ax=plt.subplots(2,2, figsize=fig_size)
        ax[0,0].loglog(metrics['no_mask'][vs],metrics['no_mask'][metrics_to_plot[0]], marker='o', label=lbl[0])
        ax[0,0].loglog(metrics['mask'][vs],metrics['mask'][metrics_to_plot[0]], marker='*', label=lbl[1])
        ax[0,0].set_ylabel(metrics_to_plot[0].upper())
        ax[0,0].set_xlabel(xlabel[vs])
        ax[0,0].legend()

        ax[0,1].loglog(metrics['no_mask'][vs],metrics['no_mask'][metrics_to_plot[1]], marker='o', label=lbl[0])
        ax[0,1].loglog(metrics['mask'][vs],metrics['mask'][metrics_to_plot[1]], marker='*', label=lbl[1])
        ax[0,1].set_ylabel(metrics_to_plot[1].upper())
        ax[0,1].set_xlabel(xlabel[vs])
        ax[0,1].legend()

        ax[1,0].loglog(metrics['no_mask'][vs], metrics['no_mask'][metrics_to_plot[2]], marker='o', label=lbl[0])
        ax[1,0].loglog(metrics['mask'][vs], metrics['mask'][metrics_to_plot[2]], marker='*', label=lbl[1])
        ax[1,0].set_ylabel(metrics_to_plot[2].upper())
        ax[1,0].set_xlabel(xlabel[vs])
        ax[1,0].legend()
        if metrics_to_plot[2]=='ssim':
            ax[1,0].set_ylim([0.4, 1]) # low level of 0.4 since SSIM seems meaningless when very small. 

        for ir in irs:
            max_frame = np.max(list(ir.captures.keys()))
            unmasked = ir.captures[max_frame]['unmasked']
            ax[1,1].plot(flux_img.ravel(), unmasked.ravel(), linestyle = 'None', marker='*', label=f'ppp={ir.ppp}')
            ax[1,1].set_ylabel('Meas.')
            ax[1,1].set_xlabel('$\lambda$')
        ax[1,1].legend()

        my_savefig(fig, figure_dir, f'summary_tests_vs{vs}_{fig_name}')

        figs.append(fig)
        axs.append(ax)

    return figs, axs


def build_title(name, metrics, label_metrics, frame_num, num_pixels=None, num_brackets=1):

    if name is not None:
        title_str = name + '\n'
    else:
        title_str = ''

    for l in label_metrics:
        if l=='photons_per_pix':
            title_str = title_str + f'ph/pix = {metrics["photons"][frame_num]/num_pixels:.1f} \n' + f'N={frame_num*num_brackets}' + '\n'
        else:
            if l=='SSIM':
                title_str = title_str + f'{l} = {metrics[l.lower()][frame_num]:.2f}\n'
            else:
                title_str = title_str + f'{l} = {metrics[l.lower()][frame_num]:.2f}\n'

    return title_str.strip() # remove an ending newline

def plot_ref(irs, ax):
    cmap='gray'
    try:
        ref_img = irs.find_ppp(ppp=1)[0].load_img() # load image with ppp=1; if multiple images 
    except:
        ref_img = irs.load_img() # if single exposure 
    ref_img = 1-np.exp(-ref_img)

    ax.imshow(ref_img, vmin=0, vmax=1,cmap=cmap)
    ax.set_axis_off()
    ax.set_title(build_title('Reference', None, [], None))

    return ref_img 

def plot_nomask_mask(irs, ax, metric='photons', metric_val=1e6, bracket=True, to_intensity=False):

    cmap='gray'
    label_metrics = ['SSIM','MSRE', 'photons_per_pix']

    mask_num = np.argmin(np.abs(irs.metrics['mask'][metric] - metric_val))  
    no_mask_num = np.argmin(np.abs(irs.metrics['no_mask'][metric] - metric_val))  

    if bracket:
        p_mask, flux_wtd_mask, wts = irs.combine_exposures(mask_num, masked='captures')
        p_no_mask, flux_wtd_mask, wts = irs.combine_exposures(no_mask_num, masked='captures_nomask')
        num_brackets = len(irs)
    else:
        print('warning - intensity plotting is not yet setup')

    num_pixels = np.prod(np.asarray(np.shape(p_no_mask)))
    
   # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5) 
    props = dict(boxstyle='round', facecolor='white', alpha=0.5) 
    
    ax[0].imshow(p_no_mask, vmin=0, vmax=1,cmap=cmap)
    ax[0].set_axis_off()
    ax[0].text(1.05, 0.68, build_title(None, irs.metrics['no_mask'], label_metrics, no_mask_num, num_pixels, num_brackets=num_brackets), fontsize=8, transform=ax[0].transAxes, bbox=props)
    ax[1].imshow(p_mask, vmin=0, vmax=1, cmap=cmap)
    ax[1].set_axis_off()
    ax[1].text(1.05,0.68, build_title(None, irs.metrics['mask'], label_metrics, mask_num, num_pixels, num_brackets=num_brackets), fontsize=8, transform=ax[1].transAxes, bbox=props)


# 3x1 with ground truth, conventional, and SSIM 
def plot_img_compare(irs, metric='photons_pp', metric_vals=[4, 6, 30], bracket=True, to_intensity=False, figure_dir='', metrics=None):

    fig,ax=plt.subplots(3,3, figsize=(10,7)) 

    plot_ref(irs, ax[0,0])
    ax[0,1].set_title('Clocked Recharge')
    ax[0,2].set_title('Inhibition')
    for idx,mv in enumerate(metric_vals):
        plot_nomask_mask(irs, [ax[idx,1],ax[idx,2]], metric_val=mv, metric='photons_pp')
   
    #ax[2,0].set_axis_off() # adding a metric here 
    flux_img = irs.find_ppp(1)[0].load_img()

    mm_y = [0,0]
    mm_x = [0,0]
    meas_h_list = []
    
    for idx, ir in enumerate(irs):
        max_frame = np.max(list(ir.captures.keys()))
        unmasked = ir.captures[max_frame]['unmasked']
        x = flux_img.ravel()
        yl = sm.nonparametric.lowess(unmasked.ravel(), x, frac=1/5, it=3)
        irs[idx].h = x
        irs[idx].smooth_measures = yl
        ax[1,0].set_xlim([np.min(x), np.max(x)*1.0])
        mm_y[0] = np.min([np.min(yl[:,1]), mm_y[0]])
        mm_y[1] = np.max([np.max(yl[:,1]*1.05), mm_y[1]])

        mm_x[0] = np.min([np.min(yl[:,0]), mm_x[0]])
        mm_x[1] = np.max([np.max(yl[:,0]*1.05), mm_x[1]])
        ax[1,0].plot(yl[:,0],yl[:,1], label='$\overline{{(ph/pix)}}={0}$'.format(ir.ppp))
        ax[1,0].set_ylabel('Measures')
        ax[1,0].set_xlabel('$\phi$')
        ax[1,0].set_xlim([0,3])
        meas_h_list.append({'x':x, 'unmasked': unmasked.ravel(), 'yl': yl})

    ax[1,0].legend(prop={'size':10})
    ax[1,0].set_xlim(mm_x)
    ax[1,0].set_ylim(mm_y)

    metrics_to_plot = ['ssim']
    vs = 'photons_pp'
    lbl = ['clk. recharge', 'inhibit']

    ax[2,0].semilogx(irs.metrics['no_mask'][vs], irs.metrics['no_mask'][metrics_to_plot[0]], marker='o', label=lbl[0])
    ax[2,0].semilogx(irs.metrics['mask'][vs], irs.metrics['mask'][metrics_to_plot[0]], marker='*', label=lbl[1])
    ax[2,0].set_ylabel('SSIM')
    ax[2,0].set_xlabel('Det./pix')
    ax[2,0].legend(prop={'size':6}, loc=5)
    ax[2,0].set_ylim([0.4, 1])
    ax[2,0].set_xlim([1, 200])
   
    # find measurements/detections at equal metric 
    test_at = [0.5, 0.6, 0.8]
    metric_nomask = irs.metrics['no_mask'][metrics_to_plot[0]]
    metric_mask = irs.metrics['mask'][metrics_to_plot[0]]
    d_nomask = irs.metrics['no_mask'][vs]
    d_mask = irs.metrics['mask'][vs]

    for t in test_at:
        idx = np.argmin(np.abs(metric_nomask - t))
        print(f'No inhibition SSIM={t} with d={d_nomask[idx]} \n')
        idx = np.argmin(np.abs(metric_mask - t))
        print(f'inhibition SSIM={t} with d={d_mask[idx]} \n')

    plt.tight_layout()
    fig_name = os.path.splitext(os.path.split(irs.img_path)[1])[0]
    my_savefig(fig, figure_dir, f'image_comparison_{fig_name}')

    with open(os.path.join(figure_dir, f'measure_dist_data_{fig_name}.pkl'), 'wb') as f:
        pkl.dump(meas_h_list, f)

    return fig, ax, irs

def meas_heat_map(x, y, meas_or_detections):

    """ 
    create a heat map of measurements or detections versus 2 variables (e.g. Gradient-magnitude and flux)
    """

    grid_x, grid_y = np.mgrid[np.min(x):np.max(x):100j, np.min(y):np.max(y):200j]
    grid_z1 = griddata( (x.ravel(), y.ravel()), meas_or_detections, (grid_x, grid_y), method='linear')

