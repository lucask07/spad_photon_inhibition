# Determine the maximum likelihood 
# of an intensity estimate given measured values with exposure brackets 

# 2023/08/15, Lucas Koerner 

from scipy.stats import poisson 
import numpy as np
import matplotlib.pyplot as plt
from inhibition_exposure import snr_exp
from scipy.optimize import fsolve
from math import isclose
from utils import my_savefig # my_savefig(fig, figure_dir, figname)
import pandas as pd 

figure_dir = '../manuscript/figures/dynamic_range/'
plt.ion()

phi = np.linspace(0.001, 100, 100000)

# for SPAD images rate is rather low
# will want to try greater exposure times 
Texp = np.array([1, 5, 10])
measurements = np.array([0,10,0])
detections = np.array([0,7,0])

for idx, T in enumerate(Texp):
	Y = detections[idx]/measurements[idx]
	H = -np.log(1-Y)
	try:
		print(f'H = {H}; Y={Y}')
		print(f'Look-ahead Binary rate at {Texp[idx+1]} = {1-np.exp(-H*Texp[idx+1]/T)}')
	except:
		pass

prob = 1
for (d,m,T) in zip(detections, measurements, Texp):
	prob = prob*(np.exp(-phi*T))**(m-d)*(1-np.exp(-phi*T))**d

idx = np.argmax(prob)
mle_phi = phi[idx]
print(f'MLE: {mle_phi}')
print(f'Binary rate at {Texp[0]} for MLE: {1-np.exp(-mle_phi*Texp[0])}')
max_counts = np.sum(Texp*measurements)
print(f'Max counts: {max_counts}')
print(f'Total counts: {np.sum(Texp*measurements)*(1-np.exp(-mle_phi*Texp[0]))}')

# fig, ax = plt.subplots()
# for N in [100, 1000]:
# 	p = np.linspace(1,N-1, num_p-1)/N
# 	p_qtz = np.round(p*N)/N
# 	H_qtz = -np.log(1-p_qtz)
# 	H = -np.log(1-p)
# 	snr_qtz = snr_exp(N, H_qtz)
# 	ax.semilogx(H, 20*np.log10(snr_qtz), linestyle = 'None', marker='*',label=f'N = {N}')
# 	ax.set_xlabel('$H$')
# 	ax.set_ylabel('$SNR\;[dB]$')
# ax.legend()
# my_savefig(fig, figure_dir, 'qntz_snr')