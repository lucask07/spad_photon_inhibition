# Determine the inhibition fraction for a give saturation lookahead policy 

# 2024/06/30, Lucas Koerner 

import glob
import os
from scipy.stats import poisson, binom 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import my_savefig # my_savefig(fig, figure_dir, figname)
import pandas as pd 
from create_binary_images import open_img # open_img(file_name, dir_name, roi_x=(128,232), roi_y=(128,400), Thumbnail=True)

figure_dir = '../manuscript/figures/saturation_la_design/'
plt.ion()

fig_size = (4.2, 2.5)

SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=6)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

mpl.rcParams['lines.markersize'] = 3 
mpl.rcParams['lines.linewidth'] = 0.75
mpl.rcParams['axes.formatter.limits']=(-3,3)

class Policy():

    def __init__(self, t, m, d):
        
        if (len(m) == len(t)) and (len(m) == (len(d) + 1)):
            self.t = t # array of exposure times 
            self.m = m # measurements per exposure time  
            self.d = d # thresholds for inhibition 
        else:
            print('Error in length of t, measurements or d') 

    def __repr__(self):
        return f't:{self.t}, m:{self.m}, d:{self.d}'

    def detections(self, phi):
        # without lookahead saturation 
        dets = []        
        for texp, meas in zip(self.t, self.m):
            Y = 1 - np.exp(-phi*texp)
            dets.append(binom.mean(meas, Y))
        return np.asarray(dets)

    def detections_noinh(self, phi):
        # without lookahead saturation 
        # total number of detections
        return np.sum(self.detections(phi))

    def detections_unity_texp(self, phi):
        # t = 1 is minimum exposure time 
        # determine number of detections 
        meas = np.sum(self.t * self.m) 
        return meas*(1 - np.exp(-phi))

    def la_probs(self, phi):
        probs = []
        for idx in range(len(self.d)):
            probs.append(self.la_prob(phi, idx))
        return np.cumprod(np.asarray(probs))
    
    def detections_la(self, phi):
        # probability of the exposure time occuring (not being inhibited)
        no_inh = self.la_probs(phi) 
        # the first exposure time is not inhibited so preappend 1  
        no_inh = np.append(1, no_inh)
        return np.sum(self.detections(phi)*no_inh)

    def detections_vsY(self, phi, la=True):
        # probability of the exposure time occuring (not being inhibited)
        no_inh = self.la_probs(phi) 
        # the first exposure time is not inhibited so preappend 1  
        no_inh = np.append(1, no_inh)
        y_arr = (1-np.exp(-phi*self.t))
        if la:
            return self.detections(phi)*no_inh, y_arr
        else:
            return self.detections(phi), y_arr

    def la_prob(self, phi, idx):
        # return both the expected number of detections and the 
        # probability that that measurement occurs.
        Y = 1 - np.exp(-phi*self.t[idx])
        no_inh = binom.cdf(self.d[idx]-1, self.m[idx], p=Y)
        # check that 1 - no_inh matches inh 
        inh = 0
        for k in np.arange(self.d[idx], self.m[idx]+1):
            inh += binom.pmf(k, self.m[idx], p=Y)
        return no_inh 

texp = np.array([0.1, 1, 10])
measurements = np.array([3,3,3])
detections = np.array([2,2])
p = Policy(texp, measurements, detections)

print(p.detections(1))
print(p.detections(0.1))
print(p.detections(10))

no_inh = p.la_prob(1, 0)
no_inh = p.la_probs(1)

dets_ni =[]
dets_i = []
dets_unity = []
phi_arr = np.logspace(-4, 2, 100)
for phi in phi_arr:
    dets_ni.append(p.detections_noinh(phi))
    dets_i.append(p.detections_la(phi))
    dets_unity.append(p.detections_unity_texp(phi))

dets_ni = np.asarray(dets_ni)
dets_i = np.asarray(dets_i)
dets_unity = np.asarray(dets_unity)

fig,ax=plt.subplots(figsize=fig_size)
ax.loglog(phi_arr, dets_ni, label='Bracket')
ax.loglog(phi_arr, dets_i, label='Bracket+LA')
ax.loglog(phi_arr, dets_unity, label='Min Texp')
ax.set_xlabel('$\Phi$')
ax.set_ylabel('#detections')
ax.legend()
my_savefig(fig, figure_dir, 'dets_vs_Phi')

fig,ax=plt.subplots(figsize=fig_size)
ax.semilogx(phi_arr, dets_ni/dets_unity, label='Bracket')
ax.semilogx(phi_arr, dets_i/dets_unity, label='Bracket+LA')
ax.set_xlabel('$\Phi$')
ax.set_ylabel('% detections')
my_savefig(fig, figure_dir, 'percent_dets_vs_Phi')

# apply .detections and .detections_la to arrays 
# and also calculate detections for unity
d_la = list(map(p.detections_la, [0.3, 12]))
d_noinh = list(map(p.detections_noinh, [0.3, 12]))

def detections_vs_rate(p, img=None, xlim=[1e-1, 1]):
    # separate detections for each Texp so it can be plotted vs. Y
    d_ni = {'d': [ ], 'y': [ ]}
    d_i =  {'d': [ ], 'y': [ ]}

    for k in ['d', 'y']:
        for i in range(len(p.m)):
            d_ni[k].append([]) # create a list of empty lists 
            d_i[k].append([])
    if img is None:
        phi_arr = np.logspace(-4, 4, 2000)
    else:
        phi_arr = np.sort(img.ravel())
    for phi in phi_arr:
        for inh in [True, False]: 
            d, y = p.detections_vsY(phi, inh)
            for idx,t in enumerate(p.t):
                if inh:
                    d_i['d'][idx].append(d[idx])
                    d_i['y'][idx].append(y[idx])
                else:
                    d_ni['d'][idx].append(d[idx])
                    d_ni['y'][idx].append(y[idx])

    fig,ax=plt.subplots(figsize=fig_size)
    for inh in [True, False]: 
        for idx,t in enumerate(p.t):
            if inh:
                ax.semilogx(d_i['y'][idx], d_i['d'][idx], label=f'Bracket+LA: t = {t:.2f}', linestyle ='-', marker='.')
            else:
                ax.semilogx(d_ni['y'][idx], d_ni['d'][idx], label=f'Bracket: t = {t:.2f}', linestyle='--')

    ax.set_xlabel('$Y$')
    ax.set_ylabel('# detections')
    ax.set_xlim(xlim)
    ax.legend()
    return fig, d_i, d_ni 
# can also represent the x-axis in the rate domain 
# plt.ticklabel_format(scilimits=(-5, 8))

if 0:
    texp = np.array([0.1, 1, 10])
    measurements = np.array([3,3,3])
    detections = np.array([2,2])
    p = Policy(texp, measurements, detections)
    fig, d_i, d_ni = detections_vs_rate(p)
    my_savefig(fig, figure_dir, 'dets_vs_rate_p1')

    texp = np.array([0.1, 1, 10])
    measurements = np.array([3,3,3])
    detections = np.array([1,1])
    p2 = Policy(texp, measurements, detections)
    fig, d_i, d_ni = detections_vs_rate(p2)
    my_savefig(fig, figure_dir, 'dets_vs_rate_p2')

    texp = np.array([0.1, 1, 10])
    measurements = np.array([3,3,3])
    detections = np.array([3,3])
    p3 = Policy(texp, measurements, detections)
    fig, d_i, d_ni  = detections_vs_rate(p3)
    my_savefig(fig, figure_dir, 'dets_vs_rate_p3')

    texp = np.array([1/4, 1/2, 1, 2, 4])
    measurements = np.array([10,10,10,10,10])
    detections = np.array([9,9,9,9])
    p4 = Policy(texp, measurements, detections)
    fig, d_i, d_ni = detections_vs_rate(p4)
    my_savefig(fig, figure_dir, 'dets_vs_rate_p4')

    texp = np.array([1/4, 1/2, 1, 2, 4])
    measurements = np.array([10,10,10,10,10])
    detections = np.array([10,10,10,10])
    p5 = Policy(texp, measurements, detections)
    fig, d_i, d_ni  = detections_vs_rate(p5)
    my_savefig(fig, figure_dir, 'dets_vs_rate_p5')

    t_factor = 6.667
    texp = np.array([1/t_factor, 1, t_factor])
    measurements = np.array([10,10,10])
    detections = np.array([5,5])
    p6 = Policy(texp, measurements, detections)
    fig, d_i, d_ni  = detections_vs_rate(p6)
    my_savefig(fig, figure_dir, 'dets_vs_rate_p6')

    texp = np.array([1,2,3,5,8,13,21])
    measurements = np.array([2,1,1,1,1,1,1])
    detections =   np.array([2,1,1,1,1,1])
    p5 = Policy(texp, measurements, detections)
    fig, d_i, d_ni  = detections_vs_rate(p5)
    my_savefig(fig, figure_dir, 'dets_vs_rate_fibonacci')

# load an (HDR) image, normalize to 0 to 1 
# from the image determine a range of exposure times 

res = {}
for k in ['img_name', 'perc_zero', 'dr', 'texp_min', 'mult_fact', 'm','d','d_i', 'd_ni', 'd_unity_texp']:
    res[k] = []

img_dir = '/Users/koer2434/Documents/hdr_images/100samplesDataset/wideDR10'
imgs = glob.glob(os.path.join(img_dir, '*.exr'))
imgs = [os.path.split(i)[1] for i in imgs] # get list of just the filenames

for img_name in imgs:
    img = open_img(img_name, 
                       img_dir, Thumbnail=False)[0]

    img1d = img.ravel()

    # what percent are zero? 
    perc_zero = np.sum(img1d==0)/np.size(img1d)
    print(f'Percent zero = {perc_zero}')

    nozero_min = np.min(img1d[img1d>0])
    nozero_max = np.max(img1d)
    dr = 20*np.log10(nozero_max/nozero_min)
    print(f'Dynamic range = {dr} [dB]')

    perc_vals = np.percentile(img1d[img1d>0], [0.1,1,10,90,99,99.9])

    texp_min = 1/perc_vals[-1]*1.6
    mult_fact = 5
    texp = np.array([texp_min, texp_min*5, texp_min*5**2, texp_min*5**3, texp_min*5**4])
    measurements = np.array([10,10,10,10,10])
    detections =   np.array([6,6,6,6])
    p5 = Policy(texp, measurements, detections)
    fig, d_i, d_ni  = detections_vs_rate(p5, img=img1d[::100], xlim=[1e-3,1])
    d_ut = p5.detections_unity_texp(img1d[::100]) 

    print(f'Bracket detections {np.sum(d_ni["d"])}; LA detections {np.sum(d_i["d"])}')
    res['img_name'].append(img_name)
    res['dr'].append(dr)
    res['perc_zero'].append(perc_zero)
    res['texp_min'].append(texp_min)
    res['mult_fact'].append(mult_fact)
    res['m'].append(measurements[0]) # policies -- assume the same over all brackets 
    res['d'].append(detections[0])
    res['d_i'].append(np.sum(d_i['d']))
    res['d_ni'].append(np.sum(d_ni['d']))
    res['d_unity_texp'].append(np.sum(d_ut))
df = pd.DataFrame(res)
df.to_csv(os.path.join(figure_dir, 'wide_dr_lookahead.csv'), index=False)

# plot a histogram of H vs. exposure time

# fig,ax=plt.subplots()
# for t in p.t:
#     hist,bin_edges = np.histogram((img1d*t),bins=100)
#     ax.plot(bin_edges[:-1], hist)
# show the image by sampling using the number of measurements? ... tricky because of SNR weighted reconstruction 
