"""
Lucas J. Koerner, koerner.lucas@stthomas.edu
Jan 2023 

Summary single pixel inhibition results 
and study exposure bracketing 

"""
import os
import sys
import copy
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from plotting.plot_tools import metric_plots, metric_plots_bracket, plot_img_compare
from kernels import spatial_kernels as spatial_kernels
from load_config import config

plt.ion()

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from inhibition_captures import load_pickle
from utils import my_savefig, GMSD, GradientMag
import tests.inhibit_result_tests as ir_test

home = os.path.expanduser('~')
script_dir = os.path.join(home, 'bernoulli_inhibit/tests_probability_images/')
print(f'script directory: {script_dir}')


def main(inhibit_length, img_name, inhibit_thresh, kernel_name='neighbor8', exp_thresholds=[0.1,1.0,10.0]):

#    irs = ir_test.main(img_dir='bernoulli_images/BSDS500/data/images/test/', img_name=img_name, extension='.jpg', kernel_name=kernel_name, thresholds=exp_thresholds, 
#        inhibit_thresh=inhibit_thresh, inhibit_length=inhibit_length, thumbnail=False, roi_x=(16,100), roi_y=(16,100), max_frames=None)

    irs = ir_test.main(img_dir=config['img_dir'], img_name=img_name, extension=config['img_ext'], kernel_name=kernel_name, thresholds=exp_thresholds, 
        inhibit_thresh=inhibit_thresh, inhibit_length=inhibit_length, thumbnail=False, roi_x=(16,100), roi_y=(16,100), max_frames=None)

    img_name = os.path.splitext(os.path.split(irs.img_path)[1])[0]

    kernel_name = 'none'
    for k in spatial_kernels:
        try:
            if np.all(np.isclose(irs[0].spatial_kernel, spatial_kernels[k][0])):
                kernel_name = k
        except:
            pass # sizes don't match so skip 

    figure_dir = os.path.join(script_dir, 'tests_output_bracket', img_name, kernel_name, f'thresh{inhibit_thresh}', f'length{inhibit_length}')
    figure_dir = figure_dir.replace(' ', '') # remove spaces that may be caused by the Laplacian_and_avg Kernels 
    print(f'Weighted_summary. Figure dir {figure_dir}')
    irs.fig_dir = figure_dir

    try:
        os.makedirs(figure_dir)
    except FileExistsError:
        pass

    # plot the meausrements vs. probability histogram using the ground truth image 
    max_frame = 999
    fig,ax=plt.subplots()
    for ir in irs:
        int_img = ir.load_img()
        ax.plot(1-np.exp(-int_img.ravel()), ir.captures[max_frame]['unmasked'].ravel(),
                linestyle='None', marker='*', label=f'ppp={ir.ppp}')
    ax.legend()
    my_savefig(fig, figure_dir, 'p_vs_m_vs_ppp')

    ir = irs.find_ppp(1)[0] # get the inhibition result at ppp=1 
    flux_img = ir.load_img()
    p_img = 1 - np.exp(-ir.load_img()) # calculate the corresponding probability frame 
    max_frame = 999
    figs, axs = metric_plots_bracket(irs.metrics, figure_dir, flux_img=flux_img, irs=irs, fig_name='test_bracket', metrics_to_plot=['msre', 'gms_avg','ssim'])
    for fig in figs:
        fig.suptitle('Exposure brackets')

    # plot the gradient magnitude for the 1st ppp=1 capture  
    p_img_gradmag = GradientMag(p_img)
    figs, axs = metric_plots(ir.metrics, figure_dir, unmasked=ir.captures[max_frame]['unmasked'], prob_img=p_img_gradmag, prob_img_xlabel='|G|', fig_name='gradient_mag')
    for fig in figs:
        fig.suptitle(f'Grad. Mag ppp={ir.ppp}')

    for ir in irs:
        p = 1 - np.exp(-ir.load_img()) # calculate the corresponding probability frame 
        max_frame = 999
        figs, axs = metric_plots(ir.metrics, figure_dir, fig_name=f'ppp_{ir.ppp}', 
                prob_img=p, unmasked=ir.captures[max_frame]['unmasked'], metrics_to_plot=['msre', 'mse', 'ssim'])
        for fig in figs:
            fig.suptitle(f'Single exposure with ppp={ir.ppp}')

    _, _, irs = plot_img_compare(irs, figure_dir=figure_dir)

    filename_list = []
    for ph_pp in [1,2,5,8,12,16,20,24,30,50,80,200]:
        f1n, f2n, m1, m2 = irs.save_nomask_mask(filename=f'img_{ph_pp}phpp', metric_val=ph_pp)
        filename_list.append(f1n)
        filename_list.append(f2n)

    # create the test.lst for HED edge detection processing
    with open(os.path.join(figure_dir, 'test.lst'), 'w+') as f:
        for line in filename_list:
            f.write(f'{line}\n')

    # not training so this file is not used 
    with open(os.path.join(figure_dir, 'train_pair.lst'), 'w+') as f:
        f.write(f'train/aug_data/0.0_1_0/1000075.jpg train/aug_gt/0.0_1_0/100075.png\n')

    plt.close('all')
    irs.pickle('irs_pickle') 
    return irs 


if __name__ == '__main__':
    inhibit_thresh = 6
    inhibit_length=8
    #for inhibit_length in [8,20,40]:
    #    for img_name in ['148026', '167083', '170057', '220075', '223061', '227092', '241004', '306005', '351093', '361010', '41033']:

    for inhibit_length in [80]:
        for img_name in ['148026']:
            irs = main(inhibit_length, img_name, inhibit_thresh, kernel_name='large13x13')


