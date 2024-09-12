# photon inhibition with exposure brackets
#
# 2023/10/02, Lucas Koerner 
# koerner.lucas@stthomas.edu 

# At a constant sensing latency determine: 
# measurements, detections, inhibitions, and SNR weights 
# for a given exposure bracket policy 
# do this for both dim and bright pixels 
# consider a single exposure time and also an exposure bracket 

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import glob
import os 
import pandas as pd
import matplotlib as mpl

from utils import my_savefig
from create_binary_images import open_img 

from snr_efficiency import snr_exp

figure_dir = '../manuscript/figures/'
plt.ion()

SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=6)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['lines.linewidth'] = 0.75

def d_i(h, w):
    '''
    determine detections and inhibitions 
    given an h (photons in exposure time)
    and a number of measurements

    This is for a clocked recharge scenario 
    '''

    d = (1-np.exp(-h))*w 
    i = 0
    for ph in np.arange(2,50):
        i += (h**ph*np.exp(-h)/np.math.factorial(ph))*(ph-1)
    i = i*w

    return d,i

# SNR-based weighting 
# inhibition 

pix_fluxes = [(1,'bright'), (0.1,'mid'), (0.01,'dim')] 
texps = [10,1,0.1,0.01]
t_latency = 111 # allows for 100 of each for an exposure bracket

# policy 1, just a single shortest exposure time 
res = {'policy':[], 'phi':[], 
       'texp':[], 'd':[], 'i':[],'w':[], 
       'snr_h':[], 'snr_h2_overd':[], 'snr_h2_overw':[],
       'weight':[], 'snr_hdr': []}

policy = '1'
for texp in texps:
    w = t_latency/texp
    for pf in pix_fluxes:
        h = texp*pf[0]
        d,i=d_i(h=h, w=w)
        snr_h = snr_exp(w,h)
        snr_h_2_overd = snr_exp(w,h)**2/d
        res['policy'].append(policy)
        res['phi'].append(pf[0])
        res['texp'].append(texp)
        res['w'].append(w)
        res['d'].append(d)
        res['i'].append(i)        
        res['snr_h'].append(snr_h)
        res['snr_h2_overd'].append(snr_h_2_overd)
        res['snr_h2_overw'].append(snr_h**2/w)
        res['weight'].append(1)
        res['snr_hdr'].append(snr_h)


policy = 'bracket'
texps = [10,1,0.1]
w = t_latency/np.sum(texps) # weight for each exposure in the bracket, equal distribution
for pf in pix_fluxes:
    snr_h_array = []
    for texp in texps:
        h = texp*pf[0]
        d,i=d_i(h=h, w=w)
        snr_h = snr_exp(w,h)
        snr_h_array.append(snr_h)
        snr_h_2_overd = snr_exp(w,h)**2/d
        res['policy'].append(policy)
        res['phi'].append(pf[0])
        res['texp'].append(texp)
        res['w'].append(w)
        res['d'].append(d)
        res['i'].append(i)        
        res['snr_h'].append(snr_h)
        res['snr_h2_overd'].append(snr_h_2_overd)
        res['snr_h2_overw'].append(snr_h**2/w)

    wts_sum_array = []
    for idx, texp in enumerate(texps):
        # weight calculation matches Gnanasambandam, 2020 "HDR Imaging with Quanta Image Sensors: Theoretical Limits and Optimal Reconstruction"
        wts_sum = np.sum(np.array(snr_h_array)**2)
        wts_sum_array.append(wts_sum)
        res['weight'].append(snr_h_array[idx]**2/wts_sum)

    # calculate the HDR SNR, this assumes that each exposure time gets the same number of frames #
    numerator = w * pf[0]
    denominator_sq = 0
    for i in range(len(texps)):
        denominator_sq += (res['weight'][(i - len(texps))]/texps[i])**2*w*(np.exp(pf[0]*texps[i])-1)
    for i in range(len(texps)):
        res['snr_hdr'].append(numerator/np.sqrt(denominator_sq))

df = pd.DataFrame(res)

# create an SNR at equal detections column 
bracket_d = np.sum(df[df.policy=='bracket']['d'])
texp1_d = np.sum(df[(df.policy=='1') & (df.texp==0.1)]['d'])

df['snr_h_equal_det'] = df['snr_hdr']*np.sqrt(texp1_d/bracket_d)

print(f'Power consumption {texp1_d/bracket_d}')

# create a new dataframe from just the bracket policy 
for idx,texp in enumerate(texps[::-1]):
    df2 = df[(df.policy=='bracket') & (df.texp==texp)]
    dp = pd.pivot_table(df2, values=['snr_h', 'weight', 'd', 'i'], index='phi')
    cols = dp.columns.tolist()
    cols = cols[2:4] + cols[0:2]
    dp = dp[cols]
    if idx==0:
        dt = dp
    else:
        dt = pd.concat([dt,dp], axis=1)
dp = pd.pivot_table(df2, values=['snr_hdr','snr_h_equal_det'], index='phi')
dp = dp[['snr_hdr','snr_h_equal_det']] # change ordering 
dt = pd.concat([dt,dp], axis=1)

dt.to_csv(os.path.join('analytical', 'bracket_table.csv'), float_format="%.2f")
print(dt.to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.2f}".format,
)) 


# write up how snr-weighting is done 

# SNR vs Texp for 3 flux levels (in legend) and detections, inhibitions on y
# place stars for a bracketing approach and indicate SNR weights 

# calculate composite SNR once brackets are combined 

matplotlib.rcParams['legend.handlelength'] = 0
matplotlib.rcParams['legend.numpoints'] = 1

#fig,ax = plt.subplots(figsize=(3.32,2.36))

fig,ax = plt.subplots(figsize=(3.25,1.7))

ax2 = ax.twinx()
texp = np.logspace(np.log10(0.02) ,1, 10)
d_i_vect = np.vectorize(d_i)
snr_exp_vect = np.vectorize(snr_exp)

w = t_latency/texp
markers = ['o', 's', '^']
for idx,pf in enumerate(pix_fluxes[::-1]):
    h = texp*pf[0]
    d,i=d_i_vect(h=h, w=w)
    snr_h = snr_exp_vect(w,h)

    if pf[0] == 1:
        lbl = f'$\phi_{idx}$'
    else:
        lbl = f'$\phi_{idx}={pf[0]}\phi_2$'

    ax.semilogx(texp, snr_h, label=lbl, marker=markers[idx], color='k')
    ax2.semilogx(texp, d, label=lbl, marker=markers[idx], color='r', linestyle='--')
    ax2.semilogx(texp, i, label=lbl, marker=markers[idx], color='b', linestyle='-.')

ax.set_xlabel('$T \phi_2$')
ax.set_ylabel('$SNR_H$')

# Create multiple y-axis labels

if 0:
    labels = ['detections', 'inhibitions']
    colors = ['red', 'blue']

    # Position for each label
    positions = [0.03, 0.10]

    for label, color, position in zip(labels, colors, positions):
        # Create a label at the specified position
        ax2.text(1.10+position, 0.5, label,
                transform=ax.transAxes,
                verticalalignment='center',
                color=color,
                fontsize=SMALL_SIZE, rotation=90)

labels = ['detections,', 'inhibitions']
colors = ['red', 'blue']

# Position for each label
positions = [0.0, 0.62]

for label, color, position in zip(labels, colors, positions):
    # Create a label at the specified position
    ax2.text(1.15, 0.1+position, label,
            transform=ax.transAxes,
            verticalalignment='center',
            color=color,
            fontsize=9, rotation=90)


ax.legend(loc=(0.05, 0.4)) # left center 
plt.tight_layout(pad=1.0)
my_savefig(fig, figure_dir, 'SNR_D_I_vsTexp')