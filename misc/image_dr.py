"""
Utility 

Parse images and get image statistics: determine dynamic range, percent of saturated blocks, etc. 

Lucas J. Koerner, koerner.lucas@stthomas.edu
Nov 2022
Aug 2023

"""
import os
import sys
import copy
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import imageio
import platform
import cv2

matplotlib.use('QtAgg') # must run %matplotlib qt 
plt.ion()

from run_images import open_img
from bernoulli_inhibit import blockshaped
from utils import my_savefig 

hdr_img_dir = 'images/'
hdr_img_dir = '/Users/koer2434/Documents/hdr_images/'
bsds_img_dir = '/Users/koer2434/Documents/hdr_images/BSDS/BSDS500/data/images/test/'

if platform.system() == 'Linux':
    figure_dir = 'home/lkoerner/lkoerner/bernoulli_data/figures/'
else:
    figure_dir = '/Users/koer2434/My Drive/UST/research/bernoulli_imaging/manuscript/figures/inhibit_policies_hysteresis2/'

def dr_metrics(y):

    cnt_zeros = np.sum(y==0)
    print(f'Removing {cnt_zeros} zeros')
    y = y[y>0]

    dr = np.max(y)/np.min(y)
    dr_log = 20*np.log10(dr)
    dr_bits = np.log2(dr)

    print(f'{dr}, {dr_log} dB, {dr_bits} bits')

    perc_nums = np.asarray([99.9, 99, 95, 50, 5, 1, 0.1])
    perc_s = np.percentile(y, perc_nums)
    
    print(f'Percentile ratio [99.9/0.1]: {perc_s[0]/perc_s[-1]}')
    print(f'Percentile ratio [99/1]: {perc_s[1]/perc_s[-2]}')
    print(f'Percentile ratio [95/5]: {perc_s[2]/perc_s[-3]}')
    
    return dr, perc_s, perc_nums

def img_mse(y, texp, N):
    # given the inensity/flux as y and 
    # the exposure time and number of frames 
    # determine the image MSE if the image is sampled by a Bernoulli imager 

    img_std = 1/np.sqrt(N)*np.sqrt(1-np.exp(-y*texp))/np.sqrt(np.exp(-y*texp))

    img_var = img_std**2
    mse = np.sum(img_var, axis=None) 
    psnr = np.max(y*texp)*N/np.sqrt(mse)

    return mse, psnr, img_std

def blk_saturate(y, sz, texp):
    
    y_prob = 1 - np.exp(-y*texp) # probability for a 1 
    # expectation value is just y*exp
    
    blks = blockshaped(y_prob[0:(np.shape(y_prob)[0]//sz)*sz,0:(np.shape(y_prob)[1]//sz)*sz],
                       sz,sz)
    cnt_sat = np.sum(np.prod(blks, axis=(1,2))>0.9)
    return cnt_sat, blks

def plot_mse_psnr(y, N=100):
    mse_arr = np.array([])
    psnr_arr = np.array([])
    texp_arr = np.linspace(1/np.max(y)/100, 1/np.average(y)*10, 100)

    for texp in texp_arr:
        mse, psnr, img_std = img_mse(y, texp, N=100)
        mse_arr = np.append(mse_arr, mse)
        psnr_arr = np.append(psnr_arr, psnr)

    fig,ax=plt.subplots()
    ax.loglog(texp_arr/np.average(y), psnr_arr)
    ax.set_xlabel(r'$T/\overline{y}')
    ax.set_ylabel('PSNR')
    ax.set_title(f'Max: {np.max(y)}, Min: {np.min(y)}, Avg: {np.average(y)}')

    return texp_arr, mse_arr, psnr_arr

# SNR 
def SNR(Y):
    N = 1
    snr = -np.sqrt(N)*np.log(1-Y)*np.sqrt((1-Y)/Y)
    snr[np.isnan(snr)] = 0
    return snr

def image_stats(y):
    stats = {}
    stats['D'] = np.sum(y)
    stats['min'] = np.min(y)
    stats['mu'] = np.average(y)
    stats['mu+2s'] = stats['mu'] + 2*np.std(y)
    stats['mu-2s'] = stats['mu'] - 2*np.std(y)
    stats['max'] = np.max(y)
    stats['$Y>Y_{1/2}$ [\%]'] = np.sum(y>Y_sat)/np.size(y)*100
    stats['$D(Y>Y_{1/2})$ [\%]'] = np.sum(y>Y_sat)/np.sum(y)*100

    return stats 

if __name__ == '__main__':

    #h = open_img( ('3_gt.hdr', ''), Thumbnail=False, hdr_dir_override=hdr_img_dir )
    hist_bins = 100

    h = open_img( ('33044.jpg', ''), Thumbnail=False, hdr_dir_override=bsds_img_dir )
    hist_bins = 30

    dr_metrics(h)
    cnt_sat,blks = blk_saturate(h, sz=3, texp=1/np.average(h)/2)
    print(f'saturated blocks {cnt_sat}, total blocks {np.shape(blks)[0]}')
    plot_mse_psnr(h)

    # Notability: Inhibition Panel 
    T_arr = np.array([0.1, 0.3, 1, 3, 10])
    T_colors = ['blue', 'magenta', 'black', 'purple', 'red']

    # make 4x2 subplot of histograms 
    Y_sat = 0.9925 # point at which SNR drops by x2 

    h = h/np.average(h) # set average H = 1 

    fz=12
    fig,ax=plt.subplots(2, 4, figsize=(20,6))

    # histogram of intensity 
    h_hist, bin_edges = np.histogram(h.flatten(), bins=hist_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    ax[0,0].semilogx(bin_centers, h_hist)
    ax[0,0].set_xlabel('H', fontsize=fz)
    ax[0,0].set_ylabel('N', fontsize=fz)


    df = pd.DataFrame()
    # histogram of binary rate 
    # and cumulative sum of number of detections up to given binary rate 
    for T, clr in zip(T_arr, T_colors):
        h_new = h/np.average(h)*T # set average H = 1 
        y = 1-np.exp(-h_new.flatten())
        y_hist, bin_edges = np.histogram(y, bins=hist_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        ax[0,1].plot(bin_centers, y_hist, color=clr, label=f'T={T} ppp')

        # here could plot y vs cumsum(y) but that is massive
        #  use the histogram to reduce the size of the arrays
        ax[0,2].plot(bin_centers, np.cumsum(y_hist*bin_centers), color=clr,
            label=f'T={T} ppp')
        snr = SNR(y)
        snr_hist, bin_edges = np.histogram(snr, bins=hist_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        ax[0,3].plot(bin_centers, snr_hist, color=clr, label=f'T={T} ppp')
        stats = image_stats(y)
        stats['H'] = T
        #df = df.concat(stats, ignore_index=True)
        df = pd.concat([df, pd.DataFrame.from_records([stats])], ignore_index=True)

    ax[0,1].set_xlabel('Y', fontsize=fz)
    ax[0,1].set_ylabel('N', fontsize=fz)
    ax[0,1].legend()

    ax[0,2].set_xlabel('Y', fontsize=fz)
    ax[0,2].set_ylabel(r'$\sum D$', fontsize=fz)
    #ax[0,2].legend()

    ax[0,3].set_xlabel('SNR', fontsize=fz)
    ax[0,3].set_ylabel('N', fontsize=fz)
    #ax[0,3].legend()

    for i in range(4):
        ax[1,i].axis('off')

    plt.tight_layout() 
    my_savefig(fig, '../manuscript/figures/', 'inhibition_storyboard')

    print(df)

    df_new = df.reindex(columns=['H', 'min', 'mu', 'max', 'D', '$Y>Y_{1/2}$ [\%]', '$D(Y>Y_{1/2})$ [\%]'])
    # df_new.style.format(decimal=',', thousands='.', precision=2)

    table_str = df_new.to_latex(index=False, float_format=f'{{:0.3f}}'.format, escape=False)
    print(table_str)

    # https://stackoverflow.com/questions/30281485/using-latex-in-matplotlib
    # plt.rc('text', usetex=True)
    # ax[1,2].text(0.3,0.2,table_str,ha="left",va="bottom",transform=ax[1,2].transAxes,size=10)

    LAVAL = True
    if LAVAL:
        img_dir = '9C4A2269-3107b3bf85.exr'
        img = '9C4A2269-3107b3bf85.exr'
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

        img = cv.imread(os.path.join(img_dir, img))


    if 0:
        y2 = open_img( ('89072.jpg', ''), Thumbnail=False)
        dr_metrics(y2)
        cnt_sat, blks = blk_saturate(y2, sz=3, texp=1/np.average(y2)/2)
        print(f'saturated blocks {cnt_sat}, total blocks {np.shape(blks)[0]}')

        plot_mse_psnr(y2)
        # plt.hist(img_std.ravel(), 100)

        y3 = open_img( ('89072.jpg', ''), Thumbnail=False, hdr_dir_override=None, rev_g=False)
        dr_metrics(y3)
        cnt_sat, blks = blk_saturate(y3, sz=3, texp=1/np.average(y3)/2)
        print(f'saturated blocks {cnt_sat}, total blocks {np.shape(blks)[0]}')
