"""
Lucas J. Koerner, koerner.lucas@stthomas.edu
Sept 2022

When using the MSI supercomputer many CSVs are created (1 for each job)
This script combines all these files into one 

"""
import os
import sys
import copy
import pickle as pkl
import pandas as pd

# location for input and output data 
#data_dir = '/Users/koer2434/My Drive/UST/research/bernoulli_imaging/bernoulli_inhibit/data/fscore/multiple_job_runs/20221014/'
data_dir = '/Users/koer2434/My Drive/UST/research/bernoulli_imaging/bernoulli_inhibit/data/fscore/multiple_job_runs/20221015_num_frames/'
data_dir = '/Users/koer2434/My Drive/UST/research/bernoulli_imaging/bernoulli_inhibit/data/fscore/multiple_job_runs/20221029_num_frames/'
data_dir = '/Users/koer2434/My Drive/UST/research/bernoulli_imaging/bernoulli_inhibit/data/fscore/multiple_job_runs/20221030/'

output_fname = 'total_out.csv'

csv_fnames = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and f.endswith('.csv')]

for f in csv_fnames:
    dft = pd.read_csv(os.path.join(data_dir, f))
    if 'mse_mask_2022-10-15' in f:
        dft['kernel_name'] = 'vs_frames'
    try: # first concat will fail
        df = pd.concat([df, dft], ignore_index=True)
    except:
        df = dft

df.to_csv(os.path.join(data_dir, output_fname), mode='w')
