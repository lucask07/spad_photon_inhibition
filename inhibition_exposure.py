# photon inhibition by exposure time 
# And optimizing MSE by distributing frames
#
# 2022/10/11, Lucas Koerner 
# 
# Pareto plots of SNR vs. power 
# SNR vs. total frames 
# system-level limitations: 
# 1) sensing latency 
# 2) number of frames 
# 3) energy / power 
# 4) SNR limit

from scipy.stats import poisson 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import disp_img, create_img, my_savefig, GMSD, GradientMag
from bernoulli_inhibit import to_intensity, mse, msre

import copy
figure_dir = '../manuscript/figures/'
plt.ion()


def snr_weighting(probs, texps, meas, wts=None, weighting='snr'):

    ''' determine optimal SNR based weighting with an exposure time bracket 
        probs: (array) the probability (or binary rate)  [W x H x T]
        texps: (exp time) [T] 
        meas: (array) the number of measurements for each binary rate [W x H x T]
    '''
    snr_h_mat = np.zeros( np.shape(probs) )

    for idx in range(np.shape(probs)[2]):
        
        snr_h = np.multiply(-np.log(1-probs[:,:,idx]), 
                            np.multiply( np.sqrt(np.divide(1-probs[:,:,idx], probs[:,:,idx])), 
                            np.sqrt(meas[:,:,idx])))
        snr_h_mat[:,:,idx] = snr_h 
        
        # if p is 0 or 1 set SNR to 0
        snr_h_mat[probs[:,:,idx] == 0, idx]=0
        snr_h_mat[probs[:,:,idx] ==1,  idx]=0  # goal is to have p=1 if all exposures have p=1 (otherwise p=1 should not be weighted) 
        
        # print(f'{idx}: Inv var {inv_var[i,j,idx]}, probability {p[i,j]}')

    if isinstance(weighting, (list, np.ndarray)): # assume weighting is a vector 
        wts = weighting
    elif weighting == 'snr':
        wts = np.divide(snr_h_mat**2, np.nansum(snr_h_mat**2, axis=2)[:,:,None]) # normalize the weights 
        wts[np.isnan(wts)]=0 
    else:
        print('error')

    flux_wtd = np.zeros(np.shape(probs)[0:2])
    all_ones = np.ones(np.shape(probs), dtype=bool) # determine if every exposure time had a probability of 1
    ppp_arr = texps  #this is an input 
    ppp_min = np.min(ppp_arr)

    for idx in range(np.shape(probs)[2]):
        all_ones[:,:,idx] = all_ones[:,:,idx] & (probs[:,:,idx] ==1)
        h_flux = to_intensity(probs[:,:,idx])/texps[idx] # calculate h and convert to flux 
        # print(f'H flux at {idx} = {h_flux}')
        
        np.nan_to_num(h_flux, copy=False, posinf=0) # replace nan and inf with 0. These should be from a p==0 or p==1 
        # print(f'{idx}: wts {wts[i,j,idx]}, probability {p[i,j]}')
        
        if isinstance(weighting, (list, np.ndarray)): # assume weighting is a vector 
            flux_wtd+= np.multiply(h_flux, wts[idx])
        elif weighting == 'snr': 
            flux_wtd += np.multiply(h_flux, wts[:,:,idx])
        else:
            print('error')

    wtd_p = 1-np.exp(-flux_wtd)
    # replace p with 1 for any that are all 1s at the shortest exposure time 
    wtd_p[all_ones[:,:,0]]=1
    return wtd_p, flux_wtd, wts, snr_h_mat


def img_mse(y_or_p, texp, N, mode='flux'):
    # given the inensity/flux as y and 
    # the exposure time and number of frames 
    # determine the image MSE if the image is sampled by a Bernoulli imager 
    # N can be a matrix or scalar 
    if mode == 'flux':
        img_std = np.multiply(1/np.sqrt(N), np.sqrt(1-np.exp(-y_or_p*texp))/np.sqrt(np.exp(-y_or_p*texp)))
        max_sig = np.max(np.multiply(y_or_p*texp,N))
        min_sig = np.min(np.multiply(y_or_p*texp,N))
    elif mode == 'probability':
        # TODO: check this 
        img_std = np.multiply(1/np.sqrt(N), np.sqrt(y_or_p/(1-y_or_p)))
        img_rel_std = np.multiply(img_std, (1/y_or_p)) 
        max_sig = np.max(np.multiply(-1/np.log(y_or_p), N))
        min_sig = np.min(np.multiply(-1/np.log(y_or_p), N))
  
    img_var = img_std**2
    mse = np.mean(img_var, axis=None)
    msre = np.mean(img_rel_std**2, axis=None)
    psnr = (max_sig/min_sig)/mse  # definition from Aydm "Extending quality metrics to full luminance"

    return mse, psnr, img_std, msre


def avalanche_pwr(N, H):
    return N*(1 - np.exp(-H)) # = N*p

def snr_exp(N, H):
    # exposure referred SNR from Fossum 
    return np.sqrt(N)*H*1/(np.sqrt(np.exp(H) - 1))

def two_by_two(img, meas_mat, lbl=['uniform', 'weighted']):
    """
    img is a probability image 
    meas_mat: is a list or tuple of measurements matrices, 
        assumption is the first one is uniform

    """

    print('Plot two by two' + '-'*80)

    fig,ax=plt.subplots(2,2, figsize=(10,8))
    im = ax[0,0].imshow(img)
    plt.colorbar(im)

    ax[0,0].set_title('Probability image')

    ax[0,1].imshow(meas_mat[1])
    ax[0,1].set_title('Measurement matrix')
    
    for m, l in zip(meas_mat, lbl):
        ax[1,0].semilogy(img.ravel(), m.ravel(), label=l, 
                        linestyle='none', marker='*')
        ax[1,0].set_xlabel('p')
        ax[1,0].set_ylabel('Measurements')
        ax[1,0].legend()

    baseline_avalanches = np.sum(np.multiply(img, meas_mat[0]))
    for m, l in zip(meas_mat, lbl):
        snr = snr_exp_vec(m, -np.log(1-img))    
        print(f'{l}: Total number of measurements: {np.sum(m)}')
        avalanches = np.sum(np.multiply(img, m))
        print(f'{l}: Expected number of avalanches: {avalanches}')
        mse, psnr, img_std, msre = img_mse(p, None, m, mode='probability')
        print(f'{l}: MSE = {mse}, psnr = {psnr}')
        mse, psnr, img_std, msre = img_mse(p, None, m*baseline_avalanches/avalanches, 
            mode='probability')
        print(f'{l}: (scaled) MSE = {mse}, psnr = {psnr}')
        scaled_avalanches = np.sum(np.multiply(img, m*baseline_avalanches/avalanches))
        print(f'{l}: (scaled) Expected number of avalanches: {scaled_avalanches}')
        ax[1,1].hist(snr.ravel(), bins=100, histtype=u'step',label=l) # ravel to create 1D array
        print('-'*20)

    ax[1,1].legend()
    ax[1,1].set_xlabel('SNR')
    ax[1,1].set_ylabel('Occurences')
    ax[1,0].legend()

    print('-'*80)

    return fig 

'''
How do I get the SNR after combining multiple frames?  

'''

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))


def mult_pix_optimal(intensity = np.array([[0.1, 10]]), texp = np.array([0.1, 1, 10]), 
                     meas = np.array([[[100, 100, 100], [100,100, 100]]])):

    """
    intensity: of 2 different pixels 2D: [W x H]
    texp: exposure backets  1D: [T] 
    meas:   3D: [WxHxT]
    """

    probs = np.zeros(np.shape(intensity) + np.shape(texp)) # [W x H x T]

    for idx, t in enumerate(texp):
        probs[:,:,idx] = 1 - np.exp(-intensity*t)

    wtd_p, flux_wtd, wts, snr_h_mat = snr_weighting(probs, texp, meas, wts=None, weighting='snr')

    res = {}
    for n in ['Intensity', 'Y', 'H', 'Texp', 'Measures', 'Weight', 'Frames_w_av', 'SNR_H']:
        res[n] = np.array([])

    for (row_idx, col_idx), i in np.ndenumerate(intensity):
        for idx,t in enumerate(texp):
            res['Intensity'] = np.append(i, res['Intensity'])
            res['Y'] = np.append(probs[row_idx, col_idx, idx], res['Y'])
            res['H'] = np.append(intensity[row_idx, col_idx]*t, res['H'])
            res['Texp'] = np.append(t, res['Texp'])
            res['Weight'] = np.append(wts[row_idx, col_idx, idx], res['Weight'])
            res['Frames_w_av'] = np.append(probs[row_idx, col_idx, idx], res['Frames_w_av'])
            res['SNR_H'] = np.append(snr_h_mat[row_idx, col_idx, idx], res['SNR_H'])
            res['Measures'] = np.append(meas[row_idx, col_idx, idx], res['Measures'])

        res['Intensity'] = np.append(i, res['Intensity'])
        res['Y'] = np.append(None, res['Y'])
        res['H'] = np.append(None, res['H'])
        res['Texp'] = np.append(None, res['Texp'])
        res['Weight'] = np.append(None, res['Weight'])
        res['Frames_w_av'] = np.append(None, res['Frames_w_av'])
        # inverse SNR weighting 
        res['SNR_H'] = np.append(1/np.sqrt(np.nansum (np.multiply(wts[row_idx, col_idx, :], 1/snr_h_mat[row_idx, col_idx, :])**2)), res['SNR_H'])
        res['Measures'] = np.append(None, res['Measures'])

    df = pd.DataFrame(res)

    # readout energy 
    read_frames = 0
    sensing_latency = 0
    for t in texp:
        idx = (df['Texp']==t)
        add_read_frames = np.max(df['Measures'][idx])
        # print(f'Adding {add_read_frames} to read_frames')
        read_frames += add_read_frames
        sensing_latency += np.max(df['Measures'][idx])*t

    avalanche_energy = np.nansum(df['Frames_w_av']*df['Measures'])
    snr_g_mean = geo_mean(df['SNR_H'][df['Texp'].isnull()])

    return df, read_frames, avalanche_energy, snr_g_mean, sensing_latency

# maximize SNR with a power constraint
# start by holding Texp at 1.6* each intensity value then optimize the measurements for each Texp 
# the number of measurements doesn't map directly to the energy 
# do this versus the ea_er ratio 


df, read_frames, ava_nrg, snr_gm, sensing_latency = mult_pix_optimal(intensity = np.array([[0.1, 10]]), texp = np.array([0.16, 16]),
                                                                meas = np.array([[[100, 100], [100, 100]]]))
# nrg = ava_nrg*ea_er + read_frames
# find the SNR and then scale all the measurements to meet an SNR goal 
# at first ignore sensing latency since Avalanche power and 


if __name__ == "__main__":
    plt.ion()
    fz = 16

    if 1:
        flux = np.logspace(-2, 1, 100)
        prob_0 = np.array([])
        photons_rejected = np.array([])

        for lam in flux:
            # for k in [0,1]:
            #   print(f'Probability of {k} = {poisson.pmf(k,lam)}')
            prob_0 = np.append(prob_0, poisson.pmf(0,lam))

            inhibited_ph = 0
            for k in np.arange(2,1000):
                inhibited_ph += poisson.pmf(k, lam)*(k-1)

            # print(f'Expected inhibited photon {inhibited_ph}')
            photons_rejected = np.append(photons_rejected, inhibited_ph)
            # number of frames with a detection is 1-poisson.pmf(0,lam)

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('H [exposure]', fontsize=fz)
        ax1.set_ylabel('P', fontsize=fz)
        ax1.semilogx(flux, prob_0, color = color, label='$P_0$')
        ax1.semilogx(flux, 1-prob_0, color = 'b', label='$P_1$')
        ax1.legend(loc='center left', fontsize=fz)
        ax1.tick_params(axis ='y')
         
        # Adding Twin Axes to plot using dataset_2
        ax2 = ax1.twinx()
         
        color = 'tab:green'
        ax2.set_ylabel('photons rejected', color = color, fontsize=fz)
        ax2.semilogx(flux, photons_rejected, color = color)
        reject_est = np.max([flux-1, np.zeros((np.shape(flux)[0],))],axis=0)
        ax2.semilogx(flux, reject_est, color = color, linestyle='--')
        ax2.tick_params(axis ='y', labelcolor = 'k')
        plt.tight_layout()
        fig.savefig('probability_photons_rejected.png')

        # fraction of photons rejected 
        fig2, ax3 = plt.subplots()
        ax3.semilogx(flux, photons_rejected/(1 - prob_0 + photons_rejected))
        ax3.set_ylabel('Fraction rejected', fontsize=fz)
        ax3.set_xlabel('H [exposure]', fontsize=fz)
        plt.tight_layout()
        fig2.savefig('photons_rejected.png')


        E_read = 0.01 # compared to avalanche energy which is 1
        lam = 1 # photons/sec 

        # given a flux lambda plot SNR vs. Tint 
        # Fixed 1) sensing latency 

        Twin = 1000 # sec 
        Tint = np.logspace(-2,2,1000)
        N = Twin/Tint 
        H = lam*Tint

        snr_exp_vec = np.vectorize(snr_exp)
        av_pwr_vec = np.vectorize(avalanche_pwr)

        fig, ax = plt.subplots()
        ax.semilogx(Tint, snr_exp_vec(N,H))
        ax.set_xlabel('$T_{int} [s]$')
        ax.set_ylabel('SNR')
        ax.set_title('Fixed sensing latency')

        mx_idx = np.argmax(snr_exp_vec(N,H))
        print(f'Fixed latency. Exposure time {Tint[mx_idx]} and N frames {N[mx_idx]} at max')

        fig, ax = plt.subplots()
        for E_read in [0, 0.001, 0.01]:
            ax.semilogx(Tint, av_pwr_vec(N,H) + N*E_read, label='$E_{read} =$ ' + f'{E_read}')
        ax.set_xlabel('$T_{int} [s]$')
        ax.set_ylabel('$Power$')
        ax.set_title('Fixed sensing latency')
        ax.legend()

        fig, ax = plt.subplots()
        ax.semilogx(av_pwr_vec(N,H) + N*E_read, snr_exp_vec(N,H))
        ax.set_xlabel('$Power$')
        ax.set_ylabel('SNR')
        ax.set_title('Fixed sensing latency')

        # System level limit: Fixed number of frames 

        N = 1000
        H = lam*Tint 

        fig, ax = plt.subplots()
        ax.semilogx(Tint, snr_exp_vec(N,H))
        ax.set_xlabel('$T_{int} [s]$')
        ax.set_ylabel('SNR')
        ax.set_title('Fixed number of frames')

        mx_idx = np.argmax(snr_exp_vec(N,H))
        print(f'Fixed frames. Exposure time {Tint[mx_idx]} and N frames {N} at max')

        fig, ax = plt.subplots()
        ax.semilogx(Tint, av_pwr_vec(N,H) + N*E_read)
        ax.set_xlabel('$T_{int} [s]$')
        ax.set_ylabel('$Power$')
        ax.set_title('Fixed number of frames')

        fig, ax = plt.subplots()
        ax.semilogx(av_pwr_vec(N,H) + N*E_read, snr_exp_vec(N,H))
        ax.set_xlabel('$Power$')
        ax.set_ylabel('SNR')
        ax.set_title('Fixed number of frames')

        # System level limit: amount of power 

        # for a given power limit can determine the number of frames 
        #  at a given exposure time and flux-rate. 
        #  if there is zero readout power the SNR is maximized when the integration time is minimized. 
        # As the readout energy increases the optimal exposure time is pushed to 1.6 which is a probability of 0.8 
        #  avalanche energy * 0.8 * N + N*readout_energy so that the number of frames is 
        #   N=energy budget/(avalanche_energy*0.8 + readout_energy)
        #  then the sensing latency would be 1.6*N 

        figr, axr = plt.subplots()

        pwr = 300
        E_read = 0.01 
        for E_read in [0, 0.001, 0.01, 0.1,100]: 
            # determine optimal N and Tint that maximizes SNR 

            '''
            N*(1 - np.exp(-lam*Tint)) + N*E_read = pwr 
            N = pwr / (1 - np.exp(-lam*Tint) + E_read)

            # find optimal Tint given this N 
            https://www.wolframalpha.com/input?i=derivative+l*t%2F%28sqrt%28W*%281-exp%28-l*t%29%29+%2B+Y%29%29*%281%2F%28sqrt%28exp%28-l*t%29-1%29%29%29+with+respect+to+t
            '''

            # determine the number of frames for a given integration time and power limit 
            N = pwr / (1 - np.exp(-lam*Tint) + E_read)
            fig, ax = plt.subplots()
            ax.semilogx(Tint, snr_exp_vec(N,H))
            ax.set_xlabel('$T_{int} [s]$')
            ax.set_ylabel('SNR')
            ax.set_title('Fixed power')

            fig, ax = plt.subplots()
            ax.semilogx(Tint, av_pwr_vec(N,H) + N*E_read)
            ax.set_xlabel('$T_{int} [s]$')
            ax.set_ylabel('$Power$')
            ax.set_title('Fixed power')

            fig, ax = plt.subplots()
            ax.semilogx(Tint, N)
            ax.set_xlabel('$T_{int} [s]$')
            ax.set_ylabel('$Frames$')
            ax.set_title('Power limit')

            axr.semilogx(N, snr_exp_vec(N,H), label=f'read-out E = {E_read}')
            axr.set_xlabel('$Frames$')
            axr.set_ylabel('SNR')
            axr.set_title('Fixed power')

            fig, ax = plt.subplots()
            ax2 = ax.twinx()
            ax.semilogx(N*Tint, snr_exp_vec(N,H))
            ax2.semilogx(N*Tint, Tint, 'r')
            ax.set_xlabel('$T_{win}$')
            ax.set_ylabel('SNR')
            ax2.set_ylabel('$T_{int}$', color = 'r')
            ax.set_title('Fixed power')

            mx_snr = np.max(snr_exp_vec(N,H))
            mx_idx = np.argmax(snr_exp_vec(N,H))
            print(f'Fixed power of {pwr}. Exposure time {Tint[mx_idx]} and N frames {N[mx_idx]} at max. Readout energy of {E_read}')
            print(f'Latency of {N[mx_idx]*Tint[mx_idx]}')

        axr.legend()
        my_savefig(figr, '', f'SNR_vs_frames')
        # System level limit: SNR 
        #  determine the N, Tint that minimizes power

        # we already have the maximum SNR for a given power 

        snr_target = 30
        N_pwr_limit = N[mx_idx]
        T_pwr_limit = Tint[mx_idx]
        N_snr = (snr_target/mx_snr)**2*N_pwr_limit
        snr_exp_vec(N_snr, lam*T_pwr_limit)

    # designing of frame mask in order to minimize MSE 

    snr_exp_vec = np.vectorize(snr_exp) # snr_exp(N,H)

    N = 100

    for max_prob in [1.0, 1/8]:

        print('---'*40)
        print(f'Maximum probability of {max_prob}')

        p = np.random.uniform(size=(100,100))*max_prob

        Nt = N*np.ones(np.shape(p))
        num_pixels = np.prod(np.shape(p))
        
        print('Equal frame weighting')
        detections = np.sum(np.multiply(Nt, p))
        mse, psnr, img_std, msre = img_mse(p, None, Nt, mode='probability')
        print(f'MSE = {mse}, psnr = {psnr}, msre = {msre}, total frames {np.sum(Nt)}, measurements per pixel {np.sum(np.multiply(Nt, p))/num_pixels}')
        print(f'detections = {detections}')

        print('Optimal frame weighting')
        p_frac = np.divide(p, (1-p))

        # optimal weighting for MSE 
        N_wt = p_frac**0.5/np.sum(p_frac**0.5) # see Blue notebook page 2022/11/18 - optimal MSE 

        # optimal weighting for MSRE 
        N_wt = 1/((1-p)*p)**0.5/np.sum(1/((1-p)*p)**0.5) # see Blue notebook page 2023/2/23 - optimal MSRE 

        # optimal measurement weighting when detections are constrained 
        N_wt = 1/((1-p)**0.5*p)/np.sum(1/((1-p)**0.5*p)) 

        Nt_opt = N*N_wt*len(p.ravel())
        detections = np.sum(np.multiply(Nt_opt, p))
        mse, psnr, img_std, msre = img_mse(p, None, Nt_opt, mode='probability')
        print(f'MSE = {mse}, psnr = {psnr}, msre = {msre}, total frames {np.sum(Nt)}, measurements per pixel {np.sum(np.multiply(Nt_opt, p))/num_pixels}')
        print(f'detections = {detections}')
        # print(f'Optimal weighting \n {Nt_opt} \n for a probability matrix of \n {p}')

        snr_orig = snr_exp_vec(Nt, -np.log(1-p))    
        snr_wt_mat = snr_exp_vec(Nt_opt, -np.log(1-p))

        fig,ax=plt.subplots(2,1)
        ax[0].semilogy(p.ravel(), N_wt.ravel(), marker='*', linestyle='none')
        ax[1].semilogy(p.ravel(), p.ravel()*N_wt.ravel(), marker='*', linestyle='none')

        # MSE weighting BUT set a Hmin so that low probability pixels have a min SNR 
        p_min_val = 0.05
        p_min = copy.deepcopy(p)
        p_min[p<p_min_val] = p_min_val
        p_frac = np.divide(p_min, (1-p_min))
        N_wt =p_frac**0.5/np.sum(p_frac**0.5)
        Nt_opt_minp = N*N_wt*len(p.ravel())

        mse, psnr, img_std, msre = img_mse(p, None, Nt_opt_minp, mode='probability')
        print(f'MSE = {mse}, psnr = {psnr}, msre = {msre}, total frames {np.sum(Nt)}, measurements per pixel {np.sum(np.multiply(Nt_opt, p))/num_pixels}')
        snr_wt_min = snr_exp_vec(Nt_opt_minp, -np.log(1-p))    

        fig = two_by_two(p,[Nt, Nt_opt, Nt_opt_minp], lbl=['uniform', 'weighted', 'weighted-minp'])
        my_savefig(fig, figure_dir, f'optimize_MSE_maxp_{max_prob}')

        # MSE weighting BUT set a Hmin so that certain pixels don't get too many frames 
        p_max_val = 0.7
        p_max = copy.deepcopy(p)
        p_max[p>p_max_val] = p_max_val
        p_frac = np.divide(p_max, (1-p_max))
        N_wt = p_frac**0.5/np.sum(p_frac**0.5)
        Nt_opt_maxp = N*N_wt*len(p.ravel())

        mse, psnr, img_std, msre = img_mse(p, None, Nt_opt_maxp, mode='probability')
        print(f'MSE = {mse}, psnr = {psnr}, msre = {msre}, total frames {np.sum(Nt)}')
        snr_wt_max = snr_exp_vec(Nt_opt_maxp, -np.log(1-p))    

        fig, ax = plt.subplots()
        ax.hist(snr_orig.ravel(), bins=100, histtype=u'step',label='uniform frames') # ravel to create 1D array
        ax.hist(snr_wt_mat.ravel(), bins=100, histtype=u'step',label='MSE weighting') # ravel to create 1D array
        ax.hist(snr_wt_min.ravel(), bins=100, histtype=u'step',label='Min SNR')
        ax.hist(snr_wt_max.ravel(), bins=100, histtype=u'step',label='Min SNR')

        ax.set_xlabel('SNR')
        ax.set_ylabel('Occurences')    
        ax.legend()

        fig2, ax2 = plt.subplots()
        ax2.semilogy(p.ravel(), Nt_opt.ravel(), label='Best MSE', linestyle='none', marker='*')
        ax2.semilogy(p.ravel(), Nt_opt_maxp.ravel(), label='Best MSE w/ max-p', linestyle='none', marker='*')

        ax2.set_xlabel('p')
        ax2.set_ylabel('N')
        ax2.legend()

        print('-'*40)
        # Equalize SNR 
        mse, psnr, img_std, msre = img_mse(p, None, Nt, mode='probability')
        print(f'MSE = {mse}, psnr = {psnr}, total frames {np.sum(Nt)}')

        h = -np.log(1-p)
        numerator = (np.exp(h) - 1)/h**2
        N_wt = N*numerator/np.sum(numerator)*len(p.ravel())
        fig2, ax2 = plt.subplots()
        ax2.semilogy(p.ravel(), N_wt.ravel(), label='Equal SNR', linestyle='none', marker='*')

        mse, psnr, img_std, msre = img_mse(p, None, N_wt, mode='probability')
        print(f'MSE = {mse}, psnr = {psnr}, total frames {np.sum(N_wt)}')
        snr_wt = snr_exp_vec(N_wt, -np.log(1-p))    

        fig = two_by_two(p, [Nt, N_wt], lbl=['uniform', 'weighted'])
        my_savefig(fig, figure_dir, f'equalize_SNR_{max_prob}')

        # equalize SNR with a frame limit of 8*N
        factor = 8
        N_wt_limit = copy.deepcopy(N_wt)
        N_wt_limit[N_wt_limit > (N*factor)] = (N*factor)
        N_wt_limit = N_wt_limit * np.sum(N_wt)/np.sum(N_wt_limit) # due to this renormalization the limit is not actually 8! 
        ax2.semilogy(p.ravel(), N_wt_limit.ravel(), label='Equal SNR with limit', linestyle='none', marker='*')

        ax2.set_xlabel('p')
        ax2.set_ylabel('N')
        ax2.legend()

        mse, psnr, img_std, msre = img_mse(p, None, N_wt_limit, mode='probability')    
        print(f'MSE = {mse}, psnr = {psnr}, total frames {np.sum(N_wt_limit)}')
        snr_limit = snr_exp_vec(N_wt_limit, -np.log(1-p))    

        fig, ax = plt.subplots()
        ax.hist(snr_orig.ravel(), bins=100, histtype=u'step',label='uniform frames') # ravel to create 1D array
        ax.hist(snr_wt.ravel(), bins=100, histtype=u'step',label='SNR equal') # ravel to create 1D array
        ax.hist(snr_limit.ravel(), bins=100, histtype=u'step',label='frame limit') # ravel to create 1D array

        ax.set_xlabel('SNR')
        ax.set_ylabel('Occurences')    
        ax.legend()


"""
if 0:
    print('-'*40)
    print('Test random frame weightings')
    for i in range(10):
        p_frac =  np.random.uniform(size=(np.shape(p)))
        N_wt =p_frac/np.sum(p_frac)
        Nt_opt = N*N_wt*len(p.ravel())
        mse, psnr, img_std = img_mse(p, None, Nt_opt, mode='probability')
        print(f'MSE = {mse}, psnr = {psnr}, total frames {np.sum(Nt)}')
"""