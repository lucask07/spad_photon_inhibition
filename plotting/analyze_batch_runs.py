"""
Lucas Koerner

"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd 

from utils import my_savefig
import inhibition_captures # required to load the pickles 

np.seterr(divide='ignore') # prevent warnings when calculating divide by zero and log(0)
np.seterr(invalid='ignore')

plt.ion()

SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

mpl.rcParams['lines.markersize']=3
mpl.rcParams['lines.linewidth']=0.75

PER_PIX = True

if PER_PIX:
    xlabel = {'photons_pp':'detections/pix.', 'measurements_pp': 'measure/pix.'}
else:
    xlabel = {'photons':'detections', 'measurements': 'measurements'}

def find_closest(metrics, metric, value):
    mask_idx = np.argmin(np.abs(metrics['mask'][metric]-value))
    nomask_idx = np.argmin(np.abs(metrics['no_mask'][metric]-value))

    return mask_idx, nomask_idx

figure_dir = '/home/lkoerner/lkoerner/bernoulli_inhibit/summary_figures'

img_name = '140088'
length = 20
kernel = 'flip_laplacian'

x_axis = 'photons_pp'

for metric in ['ssim', 'msre', 'mse']:
    fig,ax=plt.subplots(2,1, figsize=(3.25, 2.5))
    # for thresh in [1,2,4,6,10,12,24]:
    for thresh in [1,4,6,12,24]:
        data_dir = f'tests_probability_images/tests_output/{img_name}/{kernel}/thresh{thresh}/length{length}'

        irs = inhibition_captures.load_pickle(os.path.join(data_dir, 'irs_pickle'))
        ax[0].plot(irs[0].smooth_measures[:,0], irs[0].smooth_measures[:,1], label=f'$\eta={thresh}$')
        if metric in ['msre', 'mse']:
            ax[1].loglog(irs.metrics['mask']['photons_pp'], irs.metrics['mask'][metric], linestyle='-', marker='.', label=f'$\eta={thresh}$')
        else:
            ax[1].semilogx(irs.metrics['mask']['photons_pp'], irs.metrics['mask'][metric], linestyle='-', marker='.', label=f'$\eta={thresh}$')

    hmin = 0
    hmax = 3
    measures = 1000
    ax[0].plot([hmin, hmax], [measures, measures], linestyle='--', marker='*', label=f'no ihibition')
    ax[0].set_ylabel('Measures')
    ax[0].set_xlabel('$H = \phi T$')
    ax[0].legend()
    ax[0].set_xlim([0,3])

    if metric in ['msre', 'mse']:
        ax[1].loglog(irs.metrics['no_mask']['photons_pp'], irs.metrics['no_mask'][metric], linestyle='--', marker='*', label=f'no inhibition')
    else:
        ax[1].semilogx(irs.metrics['no_mask']['photons_pp'], irs.metrics['no_mask'][metric], linestyle='--', marker='*', label=f'no inhibition')
    ax[1].set_ylabel(metric.upper()) # capitalize the metric for the label 
    ax[1].set_xlabel(xlabel[x_axis])
    ax[1].set_xlim([0.5,100])
    # ax[1].legend()
    my_savefig(fig, figure_dir, f'measurements_thresh_metric{metric}_kernel{kernel}')

fig,ax=plt.subplots(3,2, figsize=(3.25, 2))
#fig2,ax2=plt.subplots(1,2, figsize=(3.25, 1.75))
fig2,ax2=plt.subplots(1,2, figsize=(4.4, 1.5))
# sweep_var_label = {0:r'$\eta={}$' + '\n' + '$I_F={:.2f}$', 1:r'$\tau_D={}$' + '\n' + r'$I_F={:.2f}$'}

sweep_var_label = {0:r'$I_F={:.2f}$', 1:r'$I_F={:.2f}$'}

num_measures = 1000

for col in [0,1]:
    metric_list = ['ssim', 'mse']
    for idx, metric in enumerate(metric_list):
        # for thresh in [1,2,4,6,10,12,24]:
        if col == 0:
            #sweep_var_list = [1,2,4,12,24]
            sweep_var_list = [2,12,24]
        else:
            sweep_var_list = [2,8,20,32,64]
            sweep_var_list = [64,32,20,8,2]
            sweep_var_list = [64,20,2]
        for sweep_var in sweep_var_list:
            if col==0:
                thresh=sweep_var
                length=20
            else:
                length=sweep_var
                thresh=12
            data_dir = f'tests_probability_images/tests_output/{img_name}/{kernel}/thresh{thresh}/length{length}'
            irs = inhibition_captures.load_pickle(os.path.join(data_dir, 'irs_pickle'))
            

            inhibit_frac = 1-np.mean(irs.metrics['mask']['photons_pp']/irs.metrics['no_mask']['photons_pp'])

            if idx==0:
                ax[0,col].plot(irs[0].smooth_measures[:,0], irs[0].smooth_measures[:,1], label=sweep_var_label[col].format(sweep_var, inhibit_frac))
                # ax2[col].plot(irs[0].smooth_measures[:,0], irs[0].smooth_measures[:,1], label=sweep_var_label[col].format(inhibit_frac))
                ax2[col].semilogx(irs[0].smooth_measures[:,0], irs[0].smooth_measures[:,1]/num_measures, label=sweep_var_label[col].format(inhibit_frac))
            if metric in ['msre', 'mse']:
                ax[idx+1, col].loglog(irs.metrics['mask']['photons_pp'], irs.metrics['mask'][metric], linestyle='-', marker='.', label=sweep_var_label[col].format(sweep_var, inhibit_frac))
            else:
                ax[idx+1, col].semilogx(irs.metrics['mask']['photons_pp'], irs.metrics['mask'][metric], linestyle='-', marker='.', label=sweep_var_label[col].format(sweep_var, inhibit_frac))

    hmin = 0
    hmax = 3
    log_hmin = 1e-2
    log_hmax = 10

    measures = 1000
    ax[0,col].plot([hmin, hmax], [measures, measures], linestyle='--', label=f'no inhibition')
    ax[0,col].set_ylabel('Measurements')
    ax[0,col].set_xlabel('$H = \phi T$ [photons]')
    leg1 = ax[0,col].legend()
    ax[0,col].set_xlim([hmin, hmax])

    # ax2[col].plot([hmin, hmax], [measures, measures], linestyle='--', label=f'no inhibition')
    if col==0:
        ax2[col].set_ylabel('Measurement\nfraction')
    else:
        ax2[col].get_yaxis().set_ticks([])
    ax2[col].set_xlabel('$H = \phi T$ [photons]')
    #leg2 = ax2[col].legend(loc=(0.39,0.48))
    leg2 = ax2[col].legend(loc=(0.02,0.05))
    #leg2.set_bbox_to_anchor((0.5,0.3))
    ax2[col].set_xlim([log_hmin, log_hmax])
    
    for idx,metric in enumerate(metric_list):
        if metric in ['msre', 'mse']:
            ax[idx+1,col].loglog(irs.metrics['no_mask']['photons_pp'], irs.metrics['no_mask'][metric], linestyle='--', marker='*', label=f'no inhibition')
        else:
            ax[idx+1,col].semilogx(irs.metrics['no_mask']['photons_pp'], irs.metrics['no_mask'][metric], linestyle='--', marker='*', label=f'no inhibition')

    ax[1,col].set_ylabel(metric_list[0].upper()) # capitalize the metric for the label 
    ax[1,col].set_xlabel(xlabel[x_axis])
    ax[1,col].set_xlim([0.5,100])
    # ax[1].legend()

    ax[2,col].set_ylabel(metric_list[1].upper()) # capitalize the metric for the label 
    ax[2,col].set_xlabel(xlabel[x_axis])
    ax[2,col].set_xlim([0.5,100])

ax[0,0].annotate(r'$\eta \! \! \uparrow$', xy=(1,570), xytext=(0.4,300), fontsize=MEDIUM_SIZE, arrowprops=dict(facecolor='black', width=1))
ax[0,1].annotate(r'$\tau_H \! \! \uparrow$', xy=(2.5,0), xytext=(2.5,300), fontsize=MEDIUM_SIZE, arrowprops=dict(facecolor='black', width=1))

ax2[0].annotate(r'$\eta \! \! \uparrow$', xy=(1.6,500/num_measures), xytext=(0.4,150/num_measures), fontsize=MEDIUM_SIZE, arrowprops=dict(headwidth=6, headlength=6, facecolor='black', width=0.2))
ax2[1].annotate(r'$\tau_H \! \! \uparrow$', xy=(2.5,0), xytext=(2.5,400/num_measures), fontsize=MEDIUM_SIZE, arrowprops=dict(headwidth=6, headlength=6, facecolor='black', width=0.2))

ax_s = ax

fig.tight_layout()
my_savefig(fig, figure_dir, f'measurements_thresh_twometrics_kernel{kernel}')

fig2.tight_layout()
my_savefig(fig2, figure_dir, f'measurements_thresh_kernel{kernel}')

def create_df(m_test='ssim', vals = [0.6,0.7,0.8,0.9]):
    kernel_names = ['neighbor8', 'flip_laplacian', 'laplacian', 'single_pix_bright', 'single_pix_dark']
    length = 20
    thresh = 12

    res = {'kernel': [], 'length': [], 'thresh':[]}

    for sv in vals:
        res[f'pp_{m_test}={sv}'] = []
        res[f'{m_test}={sv}'] = []

    for length in [8,20,64]:
        for k in kernel_names:

            if k in ['single_pix_bright', 'single_pix_dark']:
                thresh = 2
            else:
                thresh = 12

            data_dir = f'tests_probability_images/tests_output/{img_name}/{k}/thresh{thresh}/length{length}'
            irs = inhibition_captures.load_pickle(os.path.join(data_dir, 'irs_pickle'))

            for sv in vals:
                mask_idx, nomask_idx = find_closest(irs.metrics, f'{m_test}', sv)
                res[f'pp_{m_test}={sv}'].append(irs.metrics['mask']['photons_pp'][mask_idx])
                res[f'{m_test}={sv}'].append(irs.metrics['mask'][f'{m_test}'][mask_idx])
                
                # pick one kernel to record the no mask IQ numbers 
                if k == 'neighbor8':
                    res[f'pp_{m_test}={sv}'].append(irs.metrics['no_mask']['photons_pp'][nomask_idx])
                    res[f'{m_test}={sv}'].append(irs.metrics['no_mask'][f'{m_test}'][nomask_idx])

            res['kernel'].append(k)
            res['length'].append(length)
            res['thresh'].append(thresh)

            if k == 'neighbor8':
                res['kernel'].append('nomask')
                res['length'].append(length)
                res['thresh'].append(thresh)

    df = pd.DataFrame(res)
    return df 

df = create_df(m_test='ssim')
# save some subsets of the summary data-frame 
# rename columns and kernels 
# setup formating to be 3 digits after the decimal place

# SSIM = 0.7, 0.8
l = 20
df_s = df[df.length==l]
df_s.rename(columns ={'kernel': '$K_s$', 'length':'$\tau_D$', 'thresh': '$\eta$'}, inplace=True)

#m_rep = {'pp_ssim': 'ppp', 'msre': 'MSRE', 'mse':'MSE'}
#for v in [0.6,0.7,0.8,0.9]:
#    for m in ["pp_ssim", "msre", "mse"]:
#        df_s.rename(columns ={f'{m}={v}': f'{m_rep[m]} @ SSIM={v}'}, inplace=True)

format_decimal = '%.3f'
df_s.to_csv(os.path.join(figure_dir, 'summary_iq_ssim.csv'), index=False, float_format=format_decimal)


df = create_df(m_test='msre', vals=[1,0.2,0.05])
# save some subsets of the summary data-frame 
# rename columns and kernels 
# setup formating to be 3 digits after the decimal place

# SSIM = 0.7, 0.8
l = 20
df_s = df[df.length==l]
df_s.rename(columns ={'kernel': '$K_s$', 'length':'$\tau_D$', 'thresh': '$\eta$'}, inplace=True)

format_decimal = '%.3f'
df_s.to_csv(os.path.join(figure_dir, 'summary_iq_msre.csv'), index=False, float_format=format_decimal)
