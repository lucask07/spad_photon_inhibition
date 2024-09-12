"""
Lucas J. Koerner, koerner.lucas@stthomas.edu
Nov 2022

assess measures of image dynamic range

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

matplotlib.use('QtAgg')

from run_images import open_img
from bernoulli_inhibit import blockshaped
from python_fscore import eval_fscore_hdr, eval_fscore

hdr_img_dir = 'images/'

if platform.system() == 'Linux':
    figure_dir = 'home/lkoerner/lkoerner/bernoulli_data/figures/'
else:
    figure_dir = '/Users/koer2434/My Drive/UST/research/bernoulli_imaging/manuscript/figures/inhibit_policies_hysteresis2/'
# location for output data 

if platform.system() == 'Linux':
    data_dir = 'home/lkoerner/lkoerner/bernoulli_data/data/' 
else: 
    data_dir = '/Users/koer2434/My Drive/UST/research/bernoulli_imaging/data/inhibit_policies/'


# make directories if needed 
for d in [figure_dir, data_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

def dr_metrics(y):
    dr = np.max(y)/np.min(y)
    dr_log = 20*np.log10(dr)
    dr_bits = np.log2(dr)

    print(f'{dr}, {dr_log} dB, {dr_bits} bits')

    perc_s = np.percentile(y,[99.9,99,95,5,1,0.1])
    
    print(f'Percentile 99.9: {perc_s[0]/perc_s[-1]}')
    print(f'Percentile 99: {perc_s[1]/perc_s[-2]}')
    print(f'Percentile 95: {perc_s[2]/perc_s[-3]}')
    
    return dr

def blk_saturate(y, sz, exposure):
    
    y_prob = 1 - np.exp(-y*exposure) # probability for a 1 
    # expectation value is just y*exposure 
    
    blks = blockshaped(y_prob[0:(np.shape(y_prob)[0]//sz)*sz,0:(np.shape(y_prob)[1]//sz)*sz],
                       sz,sz)
    cnt_sat = np.sum(np.prod(blks, axis=(1,2))>0.9)
    return cnt_sat, blks

y = open_img( ('3_gt.hdr', ''), Thumbnail=False, hdr_dir_override=hdr_img_dir )
dr_metrics(y)
cnt_sat,blks = blk_saturate(y,3,1/np.average(y)/2)
print(f'saturated blocks {cnt_sat}, total blocks {np.shape(blks)[0]}')
roi = np.array([1000, 1600, 1000, 1400])
y = y[(roi[0]-1):roi[1], (roi[2]-1):roi[3]] # -1 to match with MATLAB indexing
fscore, edge_map = eval_fscore_hdr(y, '3_gt.mat', roi)
print(fscore)

if 0:
    y2 = open_img( ('89072.jpg', ''), Thumbnail=False)
    dr_metrics(y2)
    cnt_sat,blks = blk_saturate(y2,3,1/np.average(y2)/2)
    print(f'saturated blocks {cnt_sat}, total blocks {np.shape(blks)[0]}')

    y3 = open_img( ('89072.jpg', ''), Thumbnail=False, hdr_dir_override=None, rev_g=False)
    dr_metrics(y3)

    cnt_sat,blks = blk_saturate(y3,3,1/np.average(y3)/2)
    print(f'saturated blocks {cnt_sat}, total blocks {np.shape(blks)[0]}')
