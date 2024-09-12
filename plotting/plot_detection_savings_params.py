"""
Lucas Koerner

"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import glob 
import re
import seaborn as sns # for boxplot / jitter plot

from utils import my_savefig
import inhibition_captures # required to load the pickles 
from plotting.plot_tools import load_irs

np.seterr(divide='ignore') # prevent warnings when calculating divide by zero and log(0)
np.seterr(invalid='ignore')
fig_size = (8,6)

plt.ion()

TINY_SIZE = 8
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TINY_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TINY_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# directory to search for output results 
base_dir = '/home/lkoerner/lkoerner/bernoulli_inhibit/tests_probability_images/tests_output_bracket/'

file_org = {0:'tests_output', 1:'img_id', 2:'kernel', 3:'thresh',4:'length'}

# using the HED output mat files as an indicator of the file names 
files = glob.glob(os.path.join(base_dir, '*/*/*/*/hed/mat/*.mat'))

# if the HED outputs are not available instead use the image comparison file  
files = glob.glob(os.path.join(base_dir, '*/*/*/*/image_comparison*.png'))
res = {'img_ids':[],
        'kernels':[],
        'inhibit_lens':[],
        'inhibit_threshs':[],
        # 'phpps':[],
        'mask': []}

for fs in files:
    fs_sp = fs.split('/')
    idx = fs_sp.index('tests_output_bracket') # find index 0
    img_id = fs_sp[idx+1]
    kernel = fs_sp[idx+2]
    thresh_tmp = re.findall(r'\d+', fs_sp[idx+3])
    if len(thresh_tmp) > 1:
        thresh = '-'.join(thresh_tmp)
    else:
        thresh = thresh_tmp[0]

    inhibit_len_tmp = re.findall(r'\d+', fs_sp[idx+4])
    if len(inhibit_len_tmp) > 1:
        inhibit_len = '-'.join(inhibit_len_tmp)
    else:
        inhibit_len = inhibit_len_tmp[0]
    # fname = fs_sp[idx+7]
    # phpp = re.findall(r'\d+', fname.split('_')[1])[0] 
    # mask_nomask = fname.split('_')[2].replace('.mat', '')

    for mask_nomask in ['mask', 'nomask']:
        res['img_ids'].append(img_id)
        res['kernels'].append(kernel)
        res['inhibit_lens'].append(inhibit_len)
        res['inhibit_threshs'].append(thresh)
        res['mask'].append(mask_nomask)

df = pd.DataFrame(res)
df.to_csv('bracket_folders.csv')

df.inhibit_lens = df.inhibit_lens.astype(int)
df.inhibit_threshs = df.inhibit_threshs.astype(int)

# directory for figure results 
figure_dir = '/home/lkoerner/lkoerner/bernoulli_inhibit/summary_figures/supplement/'

PER_PIX = True # plot detections per pixel (True) or detections.
if PER_PIX:
    xlabel = {'photons_pp':'detections/pix', 'measurements_pp': 'measure/pix'}
else:
    xlabel = {'photons':'detections', 'measurements': 'measurements'}


def find_closest(metrics, metric, value):
    # locate the index of the nearest value of a certain metric in a metrics dict. 
    # the metrics dictionary should have both a mask and no_mask key 
    mask_idx = np.argmin(np.abs(metrics['mask'][metric]-value))
    nomask_idx = np.argmin(np.abs(metrics['no_mask'][metric]-value))

    return mask_idx, nomask_idx


class MetricEqual:
    def __init__(self, metric, vals):
        self.metric = metric
        self.vals = vals # e.g. vals = [0.7, 0.8, 0.85]

    def get_col_names(self, idx):
        # names of the pandas DataFrame columns
        # mask and no mask
        col_names = [f'{self.metric}_{idx}_target']
        for m in ['mask', 'nomask']:
            col_names.extend([f'{self.metric}_{idx}_val_{m}', f'{self.metric}_{idx}_det_{m}', f'{self.metric}_{idx}_meas_{m}']) 

        col_names.extend([f'{self.metric}_{idx}_det_savings', f'{self.metric}_{idx}_meas_savings'])

        return col_names 

    def get_equal(self, val, metric_dict):
        equal_vals = [val]
        for m in ['mask', 'no_mask']:
            idx = np.argmin(np.abs(metric_dict[m][self.metric]-val))
            equal_vals.append(metric_dict[m][self.metric][idx])
            equal_vals.append(metric_dict[m]['photons_pp'][idx])
            equal_vals.append(metric_dict[m]['measurements_pp'][idx])
        # append the savings metrics in % 
        # detections 
        equal_vals.append( 100*(equal_vals[2]-equal_vals[5])/equal_vals[5])
        # measurements
        equal_vals.append( 100*(equal_vals[3]-equal_vals[6])/equal_vals[6])
        # this will have the same length as the col_names 
        return equal_vals 
    
    def get_equals(self, metric_dict):
        
        r_list = []
        for idx,val in enumerate(self.vals):
            r_list.append({'cols': [], 'vals':[]})
            r_list[idx]['vals'].extend(self.get_equal(val, metric_dict))
            r_list[idx]['cols'].extend(self.get_col_names(idx))

        return r_list 

# remove duplicates at the same policy configuration created by phpps  
# df = df[df.phpps==1]
df.reset_index(drop=True)

pps = ['bracket', 1] # exposure times and bracket

for pp_idx, pp in enumerate(pps):
    print(f'PP / bracket {pp}')
    print('-'*40)
    eq_metric = MetricEqual(metric='ssim', vals=[0.7, 0.75, 0.8])

    # for each row in the data frame load the inhibition result 
    for ridx,row in df.iterrows():
        #print(f'At row {ridx}')

        irs = load_irs(row=row)
        if pp == 'bracket':
            vs = eq_metric.get_equals(irs.metrics)
            
        elif pp==1:
            ir = irs.find_ppp(pp)[0]
            vs = eq_metric.get_equals(ir.metrics)

        for v in vs:
            df.loc[ridx, v['cols']] = v['vals'] 

    ils = np.unique(df['inhibit_lens'])
    its = np.unique(df['inhibit_threshs'])
    kernels = np.unique(df['kernels'])

    for il in ils:
        for it in its:
            for k in kernels:
                idx = (df.inhibit_lens==il) & (df.inhibit_threshs==it) & (df.kernels==k) & (df['mask']=='nomask')
                #print(np.sum(idx)) # this is the number of unique images (19 in total) 
                dfs = df[idx] # get mask values (.mask doesn't work) 
                for det_or_meas in ['det', 'meas']:
                    for val_idx, val in enumerate(eq_metric.vals):
                        tmp = np.average(dfs[f'{eq_metric.metric}_{val_idx}_{det_or_meas}_savings'])
                        df.loc[idx, f'{eq_metric.metric}_{val_idx}_{det_or_meas}_savings_avg'] = tmp # this is the average over all images and is replicated into the row of each image  
    det_or_meas = 'det'

    metric1 = 'ssim_0_det_savings_avg'
    metric2 = 'ssim_0_meas_savings_avg'
    fig,ax = plt.subplots(2,2,figsize=(3.5, 2.75))
    if pp == 'bracket':
        k = 'flip_laplacian'
        il = 32 
        it = 12
    else:
        k = 'flip_laplacian'
        il = 4 
        it = 12
        mask = 'nomask'

    mask = 'nomask'
    idx = (df.inhibit_lens==il) & (df.kernels==k) & (df['mask']=='nomask')
    ax[0,0].plot(df[idx]['inhibit_threshs'], df[idx][metric1], linestyle='none', marker='*')
    ax[1,0].plot(df[idx]['inhibit_threshs'], df[idx][metric2], linestyle='none', marker='*')

    ax[0,0].set_ylabel('D [%]')
    ax[1,0].set_ylabel('Meas [%]')

    idx = (df.inhibit_threshs==it) & (df.kernels==k) & (df['mask']=='nomask')
    ax[0,1].plot(df[idx]['inhibit_lens'], df[idx][metric1], linestyle='none', marker='*')
    ax[1,1].plot(df[idx]['inhibit_lens'], df[idx][metric2], linestyle='none', marker='*')

    ax[1,0].set_xlabel('$\eta$')
    ax[1,1].set_xlabel(r'$\tau_H$')
    
    plt.tight_layout()
    my_savefig(fig, figure_dir, f'eta_tau_{pp}') 
