from run_images import open_img 
from image_dr import dr_metrics
import glob 
import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
plt.ion()

hdr_dir = '/home/lkoerner/lkoerner/datasets/100samplesDataset/'
fname = '9C4A2185-da53a9297d.exr'

res = {'names':[], 'dr':[]}

for f in glob.glob(os.path.join(hdr_dir, '*.exr')):
    print(f)
    img = open_img([f, ''], Thumbnail=False, hdr_dir_override=hdr_dir)
    dr,perc_s,perc_nums = dr_metrics(img)
    res['names'].append(f)
    res['dr'].append(dr)

s_idx = np.argsort(np.asarray(res['dr']))[::-1]
plt.plot(np.sort(20*np.log10(np.asarray(res['dr']))))


exp_percs = np.array([1,5,50,95,99]) # must be small to large so that first is longest exposure time 
stats = {'image':[], 'dr':[], 'frames': [], 
        'Texp-perc.': [], 'Texp':[], 'total_dets':[], 
        'perc-det-99':[],'perc-det-95':[],'perc-det-50':[],
        'dets_lowSNRD':[], 'lowSNRD_perc':[]} 

for s in s_idx[0:10]:
    f = os.path.split(res['names'][s])[1]
    img = open_img([f, ''], Thumbnail=False, hdr_dir_override=hdr_dir)
    os.popen(f'cp {res["names"][s]} {os.path.join(hdr_dir, "wideDR10")}')
    print(res['names'][s])
    print(f'DR = {res["dr"][s]}')
    dr,perc_s,perc_nums = dr_metrics(img)
    img = img[img>0].flatten() # remove zeros and flatten image 
  
    # should exposure times have equal frames or equal latency  
    long_exp = None
    for exp_p in exp_percs: # determine the exposure time based on a specified percentile of image exposure 
        h = perc_s[np.argwhere(perc_nums==exp_p)]
        texp = (1.59/h).flatten()[0]

        if long_exp is None:
            long_exp = texp
        h_all = (img*texp).flatten()
        y = 1-np.exp(-img*texp).flatten()
        # calculate the detection efficiency 
        det_eff = (h_all**2*np.exp(-h_all))/(1-np.exp(-h_all))**2

        total_dets = np.sum(y[:]) # sum of Y for total detections 
        stats['dets_lowSNRD'].append(np.sum(y[det_eff<0.5]))

        stats['image'].append(f)
        stats['dr'].append(20*np.log10(res["dr"][s]))
        stats['frames'].append(long_exp/texp)
        stats['Texp-perc.'].append(exp_p)
        stats['Texp'].append(texp)
        stats['total_dets'].append(total_dets)
        stats['lowSNRD_perc'].append(stats['dets_lowSNRD'][-1]/total_dets)
        
        for idx,perc in enumerate(perc_s):
            dets = np.sum(y[img>=perc])
            print(f'At percentile of {perc_nums[idx]} with value of {perc} {dets/total_dets*100}% detections') 
            if perc_nums[idx] in [99,95,50]:
                stats[f'perc-det-{int(perc_nums[idx])}'].append(dets/total_dets*100)
        #if s == s_idx[0]:
        if 0: 
            plt.figure()
            plt.semilogx(h_all.flatten(), det_eff.flatten(), linestyle='none', marker='*')

df = pd.DataFrame(stats)
print(df)

# process this data frame to have one row of rapid and one row of exposure bracket 
# df2 = df[['image', 'dr', 'Texp-perc.', 'Texp']]
df['image'] = df['image'].apply(lambda x: x[:-6])
# df2 = df2[df2['Texp-perc.']==95]

# dft = df2[df2['Texp-perc.'].isin([5,50,95]) 

# add row as 'bracket'
unique_categories = df['image'].unique()
duplicates = pd.concat([df[df['image'] == category].iloc[:1] for category in unique_categories], ignore_index=True)
duplicates['Texp-perc.']='bracket'
dfd = pd.concat([df, duplicates], ignore_index=True)

# Goal: state fraction of detections with degraded detection eff. of 3 dB (or 6 dB) 
# now determine the number of frames for the bracket 
short_texp_frames = 1000
brackets = [5,50,95]

# loop through each image and determine the number of frames for each exposure time in a bracket 
img_uq = dfd['image'].unique()
for img_u in img_uq:
    frames_each = short_texp_frames*(dfd[ (dfd['image']==img_u) & (dfd['Texp-perc.']==95) ]['Texp'])/(dfd[ (dfd['image']==img_u) & (dfd['Texp-perc.'].isin(brackets))]['Texp'].sum())
    dfd.loc[(dfd['image']==img_u) & (dfd['Texp-perc.'].isin(brackets)), 'frames'] = frames_each.iloc[0]

# the "bracket" exposure time is so far just a copy of the Texp=1. Now calculate numbers for the brackets  
for img_u in img_uq:
    idx = (dfd['image']==img_u) & (dfd['Texp-perc.'].isin(brackets))
    total_dets = (dfd[ idx ]['total_dets'] * dfd[idx]['frames']).sum()
    idxb = (dfd['image']==img_u) & (dfd['Texp-perc.']=='bracket')
    dfd.loc[idxb, 'total_dets'] = total_dets 
    low_snrd = (dfd[ idx ]['dets_lowSNRD'] * dfd[idx]['frames']).sum() 
    dfd.loc[idxb, 'dets_lowSNRD'] = low_snrd
    dfd.loc[idxb, 'lowSNRD_perc'] = low_snrd/total_dets*100
    total_exp = (dfd[ idx ]['Texp'] * dfd[idx]['frames']).sum() 
    dfd.loc[idxb, 'Texp'] = total_exp 

print(dfd[dfd['Texp-perc.']=='bracket'])

for img_u in img_uq:
    print(dfd[((dfd['image']==img_u) & ((dfd['Texp-perc.']=='bracket') | dfd['Texp-perc.'].isin(brackets) ))])
    print('-'*40)

print('Mean of low SNRD detections')
print(dfd[(dfd['Texp-perc.']=='bracket')]['lowSNRD_perc'].mean())


# display images 
if 0:
    for s in s_idx[0:2]:
        f = os.path.split(res['names'][s])[1]
        img = open_img([f, ''], Thumbnail=False, hdr_dir_override=hdr_dir)
        plt.figure()
        plt.imshow(img**0.05)
        print(res['names'][s])
        print(f'DR = {res["dr"][s]}')


    # fraction of detections vs. percentile 
# set T based on 50th% pixel 


