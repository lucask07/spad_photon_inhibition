"""
Reads a CSV file to create summary plot of MSE vs. photons detected 

"""
import os
import sys
import pdb
import copy
from datetime import datetime  # Current date time in local system
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
from matplotlib.lines import Line2D
from run_images import open_img
from utils import disp_img, create_img, my_savefig


def non_uniform_savgol(x, y, window, polynom):
    """

    See: https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data

    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x

    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length
        as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))     # Matrix
    tA = np.empty((polynom, window))    # Transposed matrix
    t = np.empty(window)                # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.pinv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed


plt.ion()
TYPE = 'fscore'

data_dir = '/Users/koer2434/My Drive/UST/research/bernoulli_imaging/bernoulli_inhibit/data/fscore/multiple_job_runs/20221030/'
img_dir = '/Users/koer2434/Documents/hdr_images/BSDS/BSDS300/images/test/'
# make directories if needed
for d in [data_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# an input data name that is read and used to make summary plots
csv_name = 'total_out.csv' # make this composite file with combine_csvs.py 
df = pd.read_csv(os.path.join(data_dir, csv_name))
unq_imgs = np.unique(df['img_name'])
unq_kernels = np.unique(df['kernel_name'])
print(np.unique(df['threshold']))

data_dir_num_frames = '/Users/koer2434/My Drive/UST/research/bernoulli_imaging/bernoulli_inhibit/data/fscore/multiple_job_runs/20221015_num_frames/'
data_dir_num_frames = '/Users/koer2434/My Drive/UST/research/bernoulli_imaging/bernoulli_inhibit/data/fscore/multiple_job_runs/20221029_num_frames/'

df2 = pd.read_csv(os.path.join(data_dir_num_frames, csv_name))

markers = ['o', 'v', '8', 's', 'p', '*', '+', 'x', 'D', '1','^', '<',] # label markers

unq_thresholds = np.unique(df['threshold'])
n = len(unq_thresholds)
color = iter(cm.rainbow(np.linspace(0, 1, n)))
colors = {}
for t in unq_thresholds:
    colors[t] = next(color)

PLT_TYPE = 'linear'
PLT_TYPE = 'semilogx'
max_frames = 1000

for to_plot, to_plot_vs_frames, norm_to_max, plot_lbl in [('mse_mask', 'mse_nomask', True, 'F-score'), 
                                                          ('int_mse', 'int_mse_nomask', False, 'MSE')]:

    norm_data = {} 

    for img_name in unq_imgs:
        norm_data[img_name] = {}
        min_ph = np.inf
        max_ph = 0
        fig,ax = plt.subplots(1,2, figsize=(14,8))

        fig.suptitle(img_name)
    #    y = open_img((img_name, 'DIV2K_train_HR/', (128,600), (128,600), 'o'))
        y = open_img((img_name, img_dir), False)
        im = ax[1].imshow(y) # scaled from 0 to 1 
        fig.colorbar(im)
       
        for uk_idx, uk in enumerate(df['kernel_name'].unique()):
            # print(f'Kernel {uk} uses marker {markers[uk_idx]}')
            
            idx_kernel = (df['img_name'] == img_name) & (df['kernel_name'] == uk) & (df['max_frames'] == max_frames)# create a line that goes through all
            if sum(idx_kernel) > 0:
                norm_data[img_name][uk] ={}
                line_ph = (df['total_photons'][idx_kernel] - df['masked_photons'][idx_kernel])
                line_y = df[to_plot][idx_kernel]
                # need to sort these so the lines aren't jaggy 
                line_y = np.asarray(line_y)
                line_ph = np.asarray(line_ph)

                idx_sort = np.argsort(line_ph)
                line_y = line_y[idx_sort]
                line_ph = line_ph[idx_sort]

                # normalize to the maximum of the no-mask data 'to_plot_vs_frames'
                if norm_to_max:
                    norm_data[img_name][uk]['metric'] = line_y/np.max(df[to_plot_vs_frames][idx_kernel]) # TODO: would it be better to place back into data-frame?
                else:
                    norm_data[img_name][uk]['metric'] = line_y/np.min(df[to_plot_vs_frames][idx_kernel]) 

                # TODO is this normalization consistent with the versus frames method?
                norm_data[img_name][uk]['photons'] = line_ph/np.max(df['total_photons'][idx_kernel])

            # store all normalized data for each kernel to create a smooth plot by kernel 
            # norm_data[uk] np.vertcat(line_y/np.max(line_y))

            for t in unq_thresholds:
                idx = (df['img_name'] == img_name) & (df['kernel_name'] == uk) & (df['threshold'] == t) & (df['max_frames'] == max_frames)
                if sum(idx) > 0:
                    ph = (df['total_photons'][idx] - df['masked_photons'][idx])
                    total_ph = df['total_photons'][idx].iloc[0]
                    min_ph = np.min([np.min(ph), min_ph])
                    max_ph = np.max([np.max(ph), max_ph])

                    if t == unq_thresholds[0]:
                        legend_label = uk
                    else:
                        legend_label = None

                    if PLT_TYPE == 'linear':
                        ax[0].plot(ph, 
                            df[to_plot][idx], marker=markers[uk_idx], linestyle='None',
                            color=colors[t], label=legend_label )
                    elif PLT_TYPE == 'semilogx':
                        ax[0].semilogx(ph, 
                            df[to_plot][idx], marker=markers[uk_idx], linestyle='None',
                            color=colors[t], label=legend_label  )
                    elif PLT_TYPE == 'log':
                        ax[0].loglog(ph, 
                            df[to_plot][idx], marker=markers[uk_idx], linestyle='None',
                            color=colors[t], label=legend_label )
            if PLT_TYPE == 'linear':
                ax[0].plot(line_ph, line_y, linestyle='--')
            elif PLT_TYPE == 'semilogx':
                ax[0].semilogx(line_ph, line_y, linestyle='--')
            elif PLT_TYPE == 'log':
                ax[0].loglog(line_ph, line_y, linestyle='--')

        # no inhibition, fscore vs. frames 
        idx2 = (df2['img_name'] == img_name)
        frames = df2['max_frames'][idx2]
        metric = df2[to_plot_vs_frames][idx2]
        x = np.arange(min_ph, max_ph)

        norm_data[img_name]['no_mask'] = {}

        if norm_to_max: # normalize to the maximum value (F-score -- higher is good) or the minimum value (MSE--lower is good)
            max_idx = np.argmax(frames)
            norm_data[img_name]['no_mask']['metric'] = metric/metric.iloc[max_idx]  # no_mask is a kernel name 
        else:
            norm_data[img_name]['no_mask']['metric'] = metric/np.min(metric) 
        norm_data[img_name]['no_mask']['photons'] = frames/np.max(frames)

        if PLT_TYPE == 'linear':
            ax[0].plot(frames/np.max(frames)*max_ph, metric, linestyle='--')
            ax[0].set_xlim((0,  ax[0].get_xlim()[1]))
        elif PLT_TYPE == 'log':
            pass
            # ax[0].loglog(x, total_ph/x, linestyle='--')
            # ax[0].set_xlim((min_ph,  ax[0].get_xlim()[1]))
        elif PLT_TYPE == 'semilogx':
            ax[0].plot(frames/np.max(frames)*max_ph, metric, linestyle='--', marker='.')
            #ax[0].set_xscale('log', base=2)
            #ax[0].set_yscale('log', base=2)
            ax[0].set_xlim(min_ph,  ax[0].get_xlim()[1])
        ax[0].set_xlabel('photons detected')
        if TYPE != 'fscore':
            ax[0].set_ylabel('$MSE_{mask}/MSE_{no-mask}$')
        else:
            ax[0].set_ylabel('$F-score_{mask}$')
            ax[0].set_ylim([0.3, ax[0].get_ylim()[1]])

        ax[0].legend()
        leg = ax[0].get_legend()
        for handle in leg.legendHandles:
            handle.set_color('k')

        my_savefig(fig, data_dir, '{}_mse_vs_photons'.format(img_name))
        plt.close(fig)

    # create a plot for all images 
    # using norm_data 

    img_names = list(norm_data.keys())
    kernel_names = list(norm_data[img_names[0]].keys())
    kernel_data = {}
    for img in img_names:
        for k in kernel_names:
            if kernel_data.get(k) is None:
                kernel_data[k] = {'metric': np.array([]), 'photons': np.array([])}
            if norm_data[img].get(k) is not None:
                kernel_data[k]['metric'] = np.append(kernel_data[k]['metric'], norm_data[img][k]['metric'])
                kernel_data[k]['photons'] = np.append(kernel_data[k]['photons'], norm_data[img][k]['photons'])

    from scipy.signal import savgol_filter
    fig_total,ax = plt.subplots(figsize=(9,4.5))
    fig_total2,ax2 = plt.subplots()
    for k in kernel_names:
        coeffs = np.polyfit(kernel_data[k]['photons'], kernel_data[k]['metric'], 9) #TODO better way to smooth data

        kernel_data[k]['fit'] = coeffs        
        idx_sort = np.argsort(kernel_data[k]['photons'])
        sort_ph = kernel_data[k]['photons'][idx_sort]
        sort_metric = kernel_data[k]['metric'][idx_sort]

        if k == 'no_mask':
            sort_metric = sort_metric
            print(f'Length of photons for no mask: {len(kernel_data[k]["photons"])}' )

        poly_eval = np.polyval(coeffs, sort_ph)
        savgol_eval = non_uniform_savgol(sort_ph, sort_metric, 45, 21)  # window and polynomial order 
        #yhat = savgol_filter(y, 51, 3)
        if k == 'no_mask':
            leg_label = 'vs_frames'
            marker = '*'
        else:
            leg_label = k
            marker = '.'
        ax.loglog(sort_ph, poly_eval, label=leg_label, marker=marker)
        ax2.loglog(sort_ph, savgol_eval, label=leg_label, marker=marker)
        # ax.plot(kernel_data[k]['photons'], kernel_data[k]['metric'], marker='.', linestyle='None')

    ax.set_ylabel(plot_lbl)
    ax.set_xlabel('Photon fraction')
    if plot_lbl == 'F-score':
        ax.set_ylim([0.65, 1.1])
        ax.set_xlim([1e-2,1.1])

    elif plot_lbl == 'MSE':
        ax.set_ylim([0.9, 100])
        ax.set_xlim([1e-2,1.1])

    fig_total.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    ax2.set_ylabel(plot_lbl)
    ax2.set_xlabel('Photon fraction')
    if plot_lbl == 'F-score':
        ax2.set_ylim([0.65, 1.1])
        ax2.set_xlim([1e-2,1.1])
    elif plot_lbl == 'MSE':
        ax2.set_ylim([0.9, 100])
        ax2.set_xlim([1e-2,1.1])

    fig_total2.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))

    my_savefig(fig_total, data_dir, f'composite_fit_{to_plot}_vs_photons_all_kernels')
    my_savefig(fig_total2, data_dir, f'composite_fit_{to_plot}_vs_photons_all_kernels_savgol')

    # plt.close('all')
max_thresh = np.max(df['threshold'])
min_length = np.min(df['inhibition_length'])

for img_name in unq_imgs:
# for img_name in ['109053.jpg']:
    idx = (df['img_name'] == img_name) & (df['max_frames'] == max_frames) # there will be many, just print the first no mask
    print(f'Image name {img_name} no mask, inhibit study, F-score = {df[idx]["mse_nomask"].iloc[0]}')

    idx = (df['img_name'] == img_name) & (df['threshold'] == max_thresh) & (df['inhibition_length'] == min_length) & (df['masked_photons'] == 0) # there will be many, just print the first no mask
    if np.sum(idx) > 0:
        print(f'Image name {img_name} 0 masking, inhibit study, F-score = {df[idx]["mse_mask"].iloc[0]}')

    idx2 = (df2['img_name'] == img_name) & (df2['max_frames'] == 1000)
    print(f'Image name {img_name} no mask versus frames F-score = {df2[idx2]["mse_nomask"].iloc[0]}')
    print('-'*30)