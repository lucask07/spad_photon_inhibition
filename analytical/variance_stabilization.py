
"""
Does a probability image better communicate confidence in differences? 
corresponds to section processing_on_binaryrate.text 

2023
Lucas Koerner
koerner.lucas@stthomas.edu

"""
import os
import sys
import copy
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.ion() 
import pandas as pd
import platform
from scipy.stats import bernoulli
from utils import my_savefig

if platform.system() == 'Linux':
    figure_dir = '~/bernoulli_data/figures/'
else:
    figure_dir = '/Users/koer2434/My Drive/UST/research/bernoulli_imaging/manuscript/overleaf/Adaptive Photon Rejection/figures/'


prob_arr = np.linspace(0,1,1000) # shuffle to get various gradient values
np.random.shuffle(prob_arr)
prob_std = np.array([])
H_std = np.array([])

gp_arr = np.array([])
gp_true_arr = np.array([])
gp_std = np.array([])

gH_arr = np.array([])
gH_true_arr = np.array([])
gH_std = np.array([])


N = 10000  # number of frames 
trials = 1000
p_prior = None 

for p in prob_arr:
	res = np.sum(bernoulli.rvs(p, size=(N, trials)),axis=0)
	H = -np.log(1-res/N)
	H_true = -np.log(1-p)

	prob_std = np.append(prob_std, np.std(res/N))
	H_std = np.append(H_std, np.std(H))

	if p_prior is not None:
		gp = res - p_prior
		gH = H - H_prior 
		gp_true = p - p_true_prior
		gH_true = H_true - H_true_prior

		gp_arr = np.append(gp_arr, np.mean(gp/N))
		gH_arr = np.append(gH_arr, np.mean(gH))

		gp_true_arr = np.append(gp_true_arr, gp_true)
		gH_true_arr = np.append(gH_true_arr, gH_true)

		gp_std = np.append(gp_std, np.std(gp/N))
		gH_std = np.append(gH_std, np.std(gH))

	p_prior = res
	H_prior = H
	p_true_prior = p
	H_true_prior = H_true

fig, ax = plt.subplots(2,3, figsize=(16,7))

ax[0,0].plot(prob_arr, prob_std, linestyle='none', marker='*')
ax[0,0].set_xlabel('p')
ax[0,0].set_ylabel('$\sigma_p$')

p_arr_sort = np.sort(prob_arr)
ax[0,0].plot(p_arr_sort, np.sqrt(p_arr_sort*(1-p_arr_sort)/N), linestyle='-')

h_arr = -np.log(1-prob_arr)
ax[1,0].plot(h_arr, H_std, linestyle='none', marker='*')
ax[1,0].set_xlabel('H')
ax[1,0].set_ylabel('$\sigma_H$')
ax[1,0].plot(np.sort(h_arr), np.sqrt(np.divide(p_arr_sort,(1-p_arr_sort))/N), linestyle='-')


ax[0,1].plot(gp_arr, gp_std, linestyle='none', marker='*')
ax[0,1].set_xlabel('$\Delta p$')
ax[0,1].set_ylabel('$\sigma_{\Delta p}$')

ax[1,1].plot(gH_arr, gH_std, linestyle='none', marker='*')
ax[1,1].set_xlabel('$\Delta H$')
ax[1,1].set_ylabel('$\sigma_{\Delta H}$')

gp_snr = np.divide(np.abs(gp_arr), gp_std)
ax[0,2].plot(gp_arr, gp_snr, linestyle='none', marker='*')
ax[0,2].set_xlabel('$\Delta p$')
ax[0,2].set_ylabel('$|\Delta p|/\sigma_{\Delta p}$')

gH_snr = np.divide(np.abs(gH_arr), gH_std)
ax[1,2].plot(gH_arr, gH_snr, linestyle='none', marker='*')
ax[1,2].set_xlabel('$\Delta H$')
ax[1,2].set_ylabel('$|\Delta H|/\sigma_{\Delta H}$')

my_savefig(fig, figure_dir, 'h_p_noise_gradient_noise')

fig, ax = plt.subplots(1,2, figsize=(8,5))
ax[0].hist(gp_snr, 100)
ax[0].set_xlim([0, 150])
ax[0].set_xlabel('$|\Delta p|/\sigma_{\Delta p}$')
ax[0].set_ylabel('$N$')

ax[1].hist(gH_snr, 100)
ax[1].set_xlim([0, 150])
ax[1].set_xlabel('$|\Delta H|/\sigma_{\Delta H}$')
ax[1].set_ylabel('$N$')

my_savefig(fig, figure_dir, 'h_p_histogram')

def gen_roc(meas_arr, meas_std, true_arr, thresh_arr):
	meas_noise = meas_arr + np.random.normal(0, meas_std)

	tp = np.asarray([np.sum( (meas_noise > x) & (true_arr>x)) for x in thresh_arr])
	fp = np.asarray([np.sum( (meas_noise > x) & (true_arr<x)) for x in thresh_arr])
	tn = np.asarray([np.sum( (meas_noise < x) & (true_arr<x)) for x in thresh_arr])
	fn = np.asarray([np.sum( (meas_noise < x) & (true_arr>x)) for x in thresh_arr])

	return tp,fp,tn,fn

fig,ax = plt.subplots(1,2)
thresh_arr = np.linspace(np.min(gp_true_arr), np.max(gp_true_arr), 100)
tp,fp,tn,fn = gen_roc(gp_arr, gp_std, gp_true_arr, thresh_arr)
ax[0].plot(np.divide(fp, fp+tn), np.divide(tp, (tp+fn)), linestyle='none', marker='*')

# thresh_arr_H = np.linspace(np.min(gH_true_arr[np.isfinite(gH_true_arr)]), np.max(gH_true_arr[np.isfinite(gH_true_arr)]), 100)
# conf_H = [np.sum(gH_arr > (x - np.random.normal(0, gH_std))) for x in thresh_arr_H]
# ax[1].plot(conf_H)

thresh_arr_H = np.linspace(np.min(gH_true_arr[np.isfinite(gH_true_arr)]), np.max(gH_true_arr[np.isfinite(gH_true_arr)]), 100)
tp,fp,tn,fn = gen_roc(gH_arr, gH_std, gH_true_arr, thresh_arr_H)
ax[1].plot(np.divide(fp, fp+tn), np.divide(tp, (tp+fn)), linestyle='none', marker='*')
