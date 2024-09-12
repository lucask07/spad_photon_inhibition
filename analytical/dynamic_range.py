# Question:
#  does the dynamic range defined by Chan 2022 match 
#   the limit set by quantization determined by the number of frames?
# 2022/11/23, Lucas Koerner 

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


total_res = {}

names = ['N', 'H-', 'H+', 'H-qtz', 'H+qtz', 'DR', 'DRqtz', 'Sat-50']
for n in names:
	total_res[n] = np.array([])

# chan eq 26

def target_snr(H, snr_val, N):
	"""
	set this equation to zero to determine H for a target SNR given a number of frames (N)
	
	H will have two roots 
	
	Args: 
	  H: exposure (often what is solved for)
	  snr_val: target SNR (often 1 to determine DR as per S. Chan)
	  N: number of frames 

	Returns:
	  evaluation of an expression that is set to zero to find roots of H

	"""

	return snr_val**2 * (np.exp(H) - 1) - N*H**2  

def print_dr(roots):
	
	"""
	determine and print the dynamic range given a list or tuble (of length 2)
	of minimum and maximum exposure

	Args: 
	  roots: list or tuple of 2 H values 

	Returns:
	  dynamic range in dB 
	"""

	roots = np.sort(roots)
	print(roots)
	dr = 20*np.log10(roots[1]/roots[0])
	print(f'Dynamic range {roots[1]/roots[0]}; dynamic range [dB] {dr}')
	return dr

def min_max_h(N):

	"""
	determine the minimum and maximum measurable exposure (H) given a number of measurements (N)

	Args: 
	  N: number of measurements (or frames)

	Returns:
	  tuple of min exposure, max exposure

	"""

	p_min = 1/N
	h_min = -1*np.log(1-p_min)
	p_max = (N-1)/N
	h_max = -1*np.log(1-p_max)
	return (h_min, h_max)

# for N in [10, 100, 1000, 10000, 1e5]:
N_arr = np.round(np.logspace(0.5,8,100))
for N in N_arr:
	print('--'*20)
	print(f'{N} frames:')
	snr_target = 1
	roots = fsolve(target_snr, [np.sqrt(N), 1/N], (snr_target, N), xtol=1.e-10)
	if isclose(roots[0], roots[1], rel_tol=1e-4, abs_tol=0.0):
		print('Try fsolve again')
		roots = fsolve(target_snr, [np.sqrt(N) + N, 1/N], (snr_target, N), xtol=1.e-10)
	dr_snr1 = print_dr(roots)
	m_m_h = min_max_h(N=N)
	dr_qtz = print_dr(m_m_h)
	print('--'*20)

	# determine the exposure at which 50% of the time the probability is 1
	p_sat = 0.5**(1/N)
	h_sat = -np.log(1-p_sat)
	print(f'With {N} frames. p of {p_sat} and H of {h_sat} has a 50% of saturation')
	p_max = (N-1)/N
	print(f'The maximum quantized value (p={p_max}) has a {(p_max**N)*100} % chance of saturation')
	p_snr_max = 1-np.exp(-np.max(roots))
	print(f'The high exposure SNR of {snr_target} with (p={p_snr_max}) has a {(p_snr_max**N)*100} % chance of saturation')
	print('--'*20)

	# names = ['N', 'H-', 'H+', 'H+qtz', 'DR', 'DRqtz', 'Sat-50']
	total_res['N'] = np.append(total_res['N'], N)
	total_res['H-'] = np.append(total_res['H-'], roots[1])
	total_res['H+'] = np.append(total_res['H+'], roots[0])
	total_res['H-qtz'] = np.append(total_res['H-qtz'], m_m_h[0])
	total_res['H+qtz'] = np.append(total_res['H+qtz'], m_m_h[1])
	total_res['DR'] = np.append(total_res['DR'], dr_snr1)
	total_res['DRqtz'] = np.append(total_res['DRqtz'], dr_qtz)
	total_res['Sat-50'] = np.append(total_res['Sat-50'], h_sat)

def f1(x):
	return '%d' % x

def f2(x):
	return '%1.3f' % x

def f3(x):
	return '%1.2e' % x

df = pd.DataFrame.from_dict(total_res)
print(df.to_latex(index=False,formatters=[f1,f3,f2,f3,f2,f2,f2,f2]))

fig,ax = plt.subplots()
ax.loglog(N_arr, 20*np.log10(total_res['H+qtz']), label='$H_{+qtz}$', marker='*', linestyle='none')
ax.loglog(N_arr, 20*np.log10(total_res['H+']), label='$H_+$', marker='.', linestyle='none')
ax.legend()
ax.set_xlabel('N')
ax.set_ylabel('$H$')
my_savefig(fig, figure_dir, 'H+_Hqtz_vs_frames')


fig,ax = plt.subplots()
ax.loglog(N_arr, 20*np.log10(total_res['H+']/total_res['H+qtz']), label='H+/H+qtz', marker='*', linestyle='none')
ax.legend()
ax.set_xlabel('N')
ax.set_ylabel('$H_{+}/H_{+qtz}$ [dB]')
my_savefig(fig, figure_dir, 'H+_Hqtz_vs_frames_ratio')


def qtz_error_var(p, N):
	"""
	Calculate the variance of a Bernoulli measurement 
	  due to quantization noise set by a limited number of frames.
	  calculate this quantization error around the given probability
	
	Args: 
	  p: probability (from 0 - 1.0)
	  N: number of measurements (or frames)

	Returns:
	  the variance (in units of exposure, float)
	  the size of the LSB (in units of exposure)

	std^2 = var = int_{-lsb/2}^{lsb/2} 1/lsb *e^2 de 
	
	analytically with a linear transfer this evaluates to lsb^2/12

	"""
	p_middle = np.round(p*N)/N
	p_min = p_middle - 1/N/2 
	p_max = p_middle + 1/N/2 

	# calculate error of the H estimate 
	h_min = -1*np.log(1-p_min)
	h_max = -1*np.log(1-p_max)
	h_middle = -1*np.log(1-p_middle)
	lsb = h_max-h_min

	integral_steps = 1000
	h = np.linspace(h_min, h_max, integral_steps)
	h_step = np.diff(h)[0]
	var = 1/(h_max-h_min) * np.sum( (h-h_middle)**2 * h_step)

	return var, lsb 

def bernoulli_var(p, N):
	"""
	Calculate the variance of a Bernoulli measurement 
	  given a probability and number of samples 
	
	Args: 
	  p: probability (from 0 - 1.0)
	  N: number of measurements (or frames)

	Returns:
	  the variance (float) [of the probability estimate]

	"""
	return 1/N*p*(1-p)


def qtz_error_var_approx(p, N):

	p_middle = np.round(p*N)/N
	p_min = p_middle - 1/N/2 
	p_max = p_middle + 1/N/2 

	# calculate error of the H estimate 
	h_min = -1*np.log(1-p_min)
	h_max = -1*np.log(1-p_max)
	lsb = h_max-h_min

	return lsb**2/12, lsb


# plot quantized SNR (by using only N steps)
fig, ax = plt.subplots()
N = 100
p = np.linspace(1,N-1,N-1)/N
H = -np.log(1-p)
snr_qtz = snr_exp(N, H)
ax.semilogx(H, 20*np.log10(snr_qtz), linestyle = 'None', marker='*')
ax.set_xlabel('$H$')
ax.set_ylabel('$SNR\;[dB]$')

N = 100
fig, ax = plt.subplots()
qtz_error_vec = np.vectorize(qtz_error_var)
qtz_error_approx_vec = np.vectorize(qtz_error_var_approx)
qtz_var, lsb = qtz_error_vec(p, N)
ax.semilogx(H, qtz_var, linestyle = 'None', marker='*')
ax.set_xlabel('$H$')
ax.set_ylabel('$\sigma^2_{qtz}$')

for N in [4, 10, 20, 50, 100, 1000]:
	fig, ax = plt.subplots()
	p = np.linspace(1,N-1,N-1)/N
	H = -np.log(1-p)
	bernoulli_vec = np.vectorize(bernoulli_var)
	bern_var = bernoulli_vec(p, N) # TODO: check units (one is in p)
	qtz_var, lsb = qtz_error_vec(p, N)
	qtz_var_approx, lsb = qtz_error_approx_vec(p, N)
	# H_max = -np.log(1 - (N-1)/N) -- this is simply the last point so not that helpful to plot 
	ax.semilogy(H, qtz_var, linestyle = 'None', marker='*', label='qtz.')
	ax.semilogy(H, -np.log(1-bern_var), linestyle = 'None', marker='*', label='bernoulli')
	ax.semilogy(H, qtz_var_approx, linestyle = 'None', marker='*', label='qtz. approx')
	
	noise_diff = qtz_var + np.log(1-bern_var) 
	# need to interpolate this to find the 
	cross_idx = np.argmin(np.abs(qtz_var + np.log(1-bern_var)))

	print(f'(N={N} Equal noise at H = {H[cross_idx]}, p={1-np.exp(-H[cross_idx])}')

	# ax.vlines(x=[H_max], ymin=1e-7, ymax=np.max(qtz_var))
	ax.semilogx(H, qtz_var -np.log(1-bern_var), linestyle = 'None', marker='*', label='Sum')
	ax.set_xlabel('$H$')
	ax.set_ylabel('$\sigma^2$')
	ax.legend()
	fig.suptitle(f'N = {N}')


# plot stair steps 
fig, ax = plt.subplots()
num_p = 1000000
for N in [100, 1000]:
	p = np.linspace(1,N-1, num_p-1)/N
	p_qtz = np.round(p*N)/N
	H_qtz = -np.log(1-p_qtz)
	H = -np.log(1-p)
	ax.plot(H, H_qtz, linestyle = '-', label=f'N = {N}')
ax.legend()
ax.set_xlabel('$H$')
ax.set_ylabel('$\widehat{H}$')
my_savefig(fig, figure_dir, 'quantized_transfer')


fig, ax = plt.subplots()
for N in [100, 1000]:
	p = np.linspace(1,N-1, num_p-1)/N
	p_qtz = np.round(p*N)/N
	H_qtz = -np.log(1-p_qtz)
	H = -np.log(1-p)
	snr_qtz = snr_exp(N, H_qtz)
	ax.semilogx(H, 20*np.log10(snr_qtz), linestyle = 'None', marker='*',label=f'N = {N}')
	ax.set_xlabel('$H$')
	ax.set_ylabel('$SNR\;[dB]$')
ax.legend()
my_savefig(fig, figure_dir, 'qntz_snr')