"""
Reads a CSV file to create summary plot of MSE or fscore vs. photons detected 

Lucas Koerner
koerner.lucas@stthomas.edu

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
PLT_TYPE = 'LOG'
data_dir = '/Users/koer2434/My Drive/UST/research/bernoulli_imaging/bernoulli_inhibit/data/fscore/multiple_job_runs/20211117_fscore_tointensity'

to_plot = 'mse_mask'
to_plot_vs_frames ='mse_nomask'
norm_to_max = True
plot_lbl = 'F-score'

figall, axall = plt.subplots(figsize=(9,4.5))

for csv_name in ['false_total_out.csv', 'true_total_out.csv']:
    df = pd.read_csv(os.path.join(data_dir, csv_name))

    unq_imgs = np.unique(df['img_name'])
    unq_kernels = np.unique(df['kernel_name'])
    norm_data = {}

    img_cnt = 0
    axall.set_prop_cycle(None)
    for img_name in unq_imgs:
        norm_data[img_name] = {}
        # no inhibition, fscore vs. frames 
        idx2 = (df['img_name'] == img_name)
        frames = df['max_frames'][idx2]
        metric = df[to_plot_vs_frames][idx2]

        norm_data[img_name]['no_mask'] = {}
        # TODO: normalize to the maximum of use intensity is True and False 

        if norm_to_max: # normalize to the maximum value (F-score -- higher is good) or the minimum value (MSE--lower is good)
            max_idx = np.argmax(frames)
            norm_data[img_name]['no_mask']['metric'] = metric/metric.iloc[max_idx]  # no_mask is a kernel name 
        else:
            norm_data[img_name]['no_mask']['metric'] = metric/np.min(metric) 
        norm_data[img_name]['no_mask']['photons'] = frames/np.max(frames)

    # create a plot for all images 
        if 'false' in csv_name:
            marker = 'o'
            lbl = f'{img_name.replace(".jpg","")}: probability'
        else:
            marker = '*'
            lbl = f'{img_name.replace(".jpg","")}: intensity'

        if img_cnt < 6:
            if img_cnt==0:
                axall.semilogx(frames, metric, marker=marker, linestyle='none',
                                label=lbl)
            else:
                axall.semilogx(frames, metric, marker=marker, linestyle='none')

            axall.set_ylabel('F-score')
            axall.set_xlabel('Frames')            
            img_cnt +=1

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
    fig_total2,ax2 = plt.subplots(figsize=(9,4.5))

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
        savgol_eval = non_uniform_savgol(sort_ph, sort_metric, 149, 21)  # window and polynomial order 
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

    fig_total.legend(loc='upper left', bbox_to_anchor=(0.15, 0.75))
    ax2.set_ylabel(plot_lbl)
    ax2.set_xlabel('Photon fraction')
    if plot_lbl == 'F-score':
        ax2.set_ylim([0.65, 1.1])
        ax2.set_xlim([1e-2,1.1])
    elif plot_lbl == 'MSE':
        ax2.set_ylim([0.9, 100])
        ax2.set_xlim([1e-2,1.1])

    fig_total2.legend(loc='upper left', bbox_to_anchor=(0.15, 0.72))

    my_savefig(fig_total, data_dir, f'composite_fit_{to_plot}_vs_photons_all_kernels')
    my_savefig(fig_total2, data_dir, f'composite_fit_{to_plot}_vs_photons_all_kernels_savgol')

    axall.legend()
    my_savefig(figall, data_dir, f'fscore_vs_frames_intensity_v_prob')
    # plt.close('all')