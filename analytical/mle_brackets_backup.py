# Analytical calculations 
# 
# photon inhibition: study SNR efficiency
# plot snr detection efficiency metric 
# and measurement efficiency metric
# 
# 
# 2023/8/6, Lucas Koerner 

from scipy.stats import poisson 
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import ticker

import matplotlib.pyplot as plt
from utils import disp_img, create_img, my_savefig, GMSD, GradientMag
from bernoulli_inhibit import to_intensity, mse, msre
from matplotlib.transforms import Transform
from matplotlib.ticker import (
    AutoLocator, AutoMinorLocator)


import copy
figure_dir = '../manuscript/figures/'
plt.ion()
#matplotlib.rcParams['lines.linewidth'] = 1 # default is 1.5

atul_scale = 6.75/8.5 # 
atul_scale = 1
SMALL_SIZE = 7*atul_scale 
MEDIUM_SIZE = 8*atul_scale
BIGGER_SIZE = 10*atul_scale

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=6)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

mpl.rcParams['lines.markersize'] = 3
mpl.rcParams['lines.linewidth'] = 0.75

# https://stackoverflow.com/questions/17165435/matplotlib-show-labels-for-minor-ticks-also
def ticks_format(value, index):
    """
    get the value and returns the value as:
       integer: [0,99]
       1 digit float: [0.1, 0.99]
       n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = np.floor(np.log10(value))
    base = value/10**exp
    if exp == 0 or exp == 1:   
        return '${0:d}$'.format(int(value))
    if exp == -1:
        return '${0:.1f}$'.format(value)
    else:
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))


def avalanche_pwr(N, H):
    # number of detections for 
    # N: number of measurements 
    # H: exposure level (photons per measurement) 
    return N*(1 - np.exp(-H)) # = N*p

def snr_exp(N, H):
    # exposure referred SNR from Fossum 
    return np.sqrt(N)*H*1/(np.sqrt(np.exp(H) - 1))


if __name__ == "__main__":
    plt.ion()

    for eff in ['detections', 'measures', 'both']:
        #fig, ax1 = plt.subplots(layout='constrained', figsize=(3.25,2.36))
        fig, ax1 = plt.subplots(layout='constrained', figsize=(2.5,1.8))
        # fig, ax1 = plt.subplots(layout='constrained', figsize=(1.6, 1.8))
        
        plt.autoscale(enable=True, axis='x', tight=True)
        #ax1 = fig.add_axes((0.1,0.25,0.8,0.65)) # create an Axes with some room below

        H = np.logspace(-2,1,100000)
        N = 100
        snr_exp_vec = np.vectorize(snr_exp)

        snr_db = 20*np.log10(snr_exp_vec(N,H))
        ax1.semilogx(H, snr_db, color='black')
        
        msnr = np.max(snr_db)
        h_max = np.argmax(snr_db)
        print(f'Max SNR = {msnr} at H={H[h_max]}')

        h3db_down = np.argmin(np.abs(snr_db[h_max:]-msnr+3)) + h_max
        print(f'SNR at 3db down = {snr_db[h3db_down]} at H={H[h3db_down]}')
        print(f'Y at 3db down = {1-np.exp(-H[h3db_down])}')

        ax1.set_xlabel(r'H = $\phi T$ [photons]')
        ax1.set_ylabel('SNR [dB]')
        ax1.set_ylim([-6, 20])
        ax1.axvline(H[h3db_down], linestyle='dotted', color='tab:orange')
        if eff == 'detections':
            ax1.text(3e-3, 17, '(a)')
        elif eff == 'measures':
            ax1.text(3e-3, 17, '(b)')
        # ax1.tick_params(axis='y')

        def h2y(x):
            return 1-np.exp(-x)
        def y2h(x):
            return -np.log(1-x)

        secax = ax1.secondary_xaxis('top', functions=(h2y, y2h))
        secax.set_xlabel(r"$Y$", fontsize=MEDIUM_SIZE)
        plt.draw()

        labels = [w.get_text() for w in secax.get_xticklabels()]
        locs=list(secax.get_xticks())
        labels+=[r'$0.5$']
        locs+=[0.5]
        #labels+=[r'$0.7$']
        #locs+=[0.7]
        labels+=[r'$0.8$']
        locs+=[0.8]
        labels+=[r'$0.99$']
        locs+=[0.99]
        #labels+=[r'$0.9999$']
        #locs+=[0.9999]           
        labels+=[r'$10^{-2}$']
        locs+=[0.01]        
        secax.set_xticklabels(labels)
        secax.set_xticks(locs)
        secax.tick_params(axis='x', which='major', labelsize=SMALL_SIZE)
        secax.tick_params(axis='x', which='minor', labelsize=SMALL_SIZE)
        secax.grid()

        # label all minor ticks 
        # secax.xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))  #add the custom ticks

        # Adding Twin Axes to plot using dataset_2
        ax2 = ax1.twinx()
             
        color = 'tab:red'
        
        if eff=='detections':
            ax2.set_ylabel(r'$SNR^2/D$', color = color)
            ax2.semilogx(H, snr_exp_vec(N,H)**2/(N*(1-np.exp(-H))), color = color)
        elif eff=='measures':
            ax2.set_ylabel(r'$SNR^2/W$', color = color)
            ax2.semilogx(H, snr_exp_vec(N,H)**2/(N), color = color)
        elif eff == 'both':
            # ax2.set_ylabel(r'$SNR^2/D$', color = color, fontsize=fz)
            ax2.semilogx(H, snr_exp_vec(N,H)**2/(N*(1-np.exp(-H))), color = color, linestyle='dashed')
            color = 'tab:blue'                        
            # ax2.set_ylabel(r'$SNR^2/W$', color = color, fontsize=fz)
            ax2.semilogx(H, snr_exp_vec(N,H)**2/(N), color = color, linestyle='-.')

            # Create multiple y-axis labels
            labels = [r'$SNR^2_{H/D}$', r'$SNR^2_{H/W}$']
            colors = ['tab:red', 'tab:blue']

            # Position for each label
            positions = [0.07, 0.18]

            for label, color, position in zip(labels, colors, positions):
                # Create a label at the specified position
                ax2.text(1.15 + position, 0.5, label,
                        transform=ax2.transAxes,
                        verticalalignment='center',
                        color=color,
                        fontsize=MEDIUM_SIZE, rotation=90)
        
        #reject_est = np.max([flux-1, np.zeros((np.shape(flux)[0],))],axis=0)
        #ax2.semilogx(flux, reject_est, color = color, linestyle='--')
        #ax2.tick_params(axis ='y', labelcolor = 'k')
        
        plt.tight_layout()

        my_savefig(fig, figure_dir, f'snr_{eff}_eff')
