"""
Q1: optimal weighting to combine images when the goal is a binary rate image? 

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

def evaluate_wts(wts, lest, name, prints=True):
	# normalize the weights 
	wts = wts/np.sum(wts)
	wts_t = np.tile(wts, (trials,1))
	lest_wt = lest*wts_t.T
	l_wt = np.sum(lest_wt, axis=0)
	wt_std = np.std(l_wt)
	wt_mean = np.average(l_wt)
	if prints:
		print(wts)
		print(f'For weighting {name}. pest: Mean {wt_mean}, Standard deviation {wt_std}')
	return wt_std, wt_mean

trials = 10000

lam = 0.5
Texp = np.array([0.1, 1, 5])
N = np.array([1000, 10000, 1000])  # number of measurements 

lest = np.zeros((len(Texp), trials))
pt = np.array([])

for idx, T in enumerate(Texp):
	H = lam*T
	p = 1-np.exp(-H) # ground truth probability 
	res = np.sum(bernoulli.rvs(p, size=(N[idx], trials)),axis=0)
	pt = np.append(pt, p)
	lest[idx,:] = -np.log(1-res/N[idx])/T

# wts = (pt*N)**2/((pt)*(1-pt))
# evaluate_wts(wts, lest, 'prob. weights')

# wts = wts**0.5
# evaluate_wts(wts, lest, 'prob. sqrt weights')

H_arr = lam*Texp
# This is the winner with the smallest standard deviation! 
snr_h = lam*Texp*np.sqrt(N)*np.sqrt(np.divide(np.exp(-H_arr), 1-np.exp(-H_arr)))
snr_wts = snr_h**2
evaluate_wts(snr_wts, lest, 'SNR')

# evaluate_wts(snr_wts**0.5, lest, 'SNR sqrt weights')

# snr_h = lam*Texp*N*np.sqrt(np.divide(np.exp(-H_arr), 1-np.exp(-H_arr)))
# snr_wts = snr_h**2
# evaluate_wts(snr_wts, lest, 'SNR*N')

# snr_h = lam*Texp*np.sqrt(np.divide(np.exp(-H_arr), 1-np.exp(-H_arr)))
# snr_wts = snr_h**2
# evaluate_wts(snr_wts, lest, 'SNR-ignore frames')

trials = 1000
lam = 1
Texp = np.logspace(-2,2,100)
N = 20

lest = np.zeros((len(Texp), trials))
mse = np.zeros(len(Texp))
snr = lam*Texp*np.sqrt(N)*np.sqrt(np.divide(np.exp(-lam*Texp), 1-np.exp(-lam*Texp)))

for idx,T in enumerate(Texp):
	H = lam*T
	p = 1-np.exp(-H) # ground truth probability 
	res = np.sum(bernoulli.rvs(p, size=(N, trials)),axis=0)
	# res[res==N] = N-1
	lest[idx,:] = -np.log(1-res/N)/T
	mse[idx] = np.average( (lest[idx,:]-1)**2) 

fig,ax = plt.subplots()
ax.loglog(Texp, 1/mse, marker='*')

fig,ax = plt.subplots()
ax.loglog(Texp, np.sqrt(1/mse), marker='*')
ax.loglog(Texp, snr, marker='o')
