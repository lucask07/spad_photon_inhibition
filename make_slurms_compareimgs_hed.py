"""
generate and submit to the scheduler slurm scripts for Bernoulli processing

uses a slurm template 
the slurm template runs a python file run_script_compareimgs.py 

2023/2/16

In this file you might change:
    1) the number of images run 
    2) the kernels run

To change whether the study is versus the number of frames or not edit the run_script_compareimgs.py

"""
import subprocess
import numpy as np
import glob 
import os
from load_config import config 

def get_bsds(search_dir, exp): 
    # create a list of the available images that have numpy binary files 
        
    fname = f'*_bernoulli_thresh(None,_{exp[0]})_maxframes1000.npy'
    search_for = os.path.join(search_dir, fname)
    fs = glob.glob(search_for)

    img_id_list = [os.path.split(fss)[1].split('_')[0] for fss in fs]

    return img_id_list 

slurm_template='run_compareimgs_hed.slurm'

# default settings 
inhibit_length = 30
inhibit_thresh = 6

img_ids = get_bsds(search_dir=config['bsds_dir'], exp=[1.0])
print(img_ids)

for inhibit_thresh in [2,6,12,24]:
# for inhibit_thresh in [8,10,12,24]:
# for inhibit_thresh in [16,24,32]:
# for inhibit_thresh in ['"[[-12,12],4,16]"', '"[[-12,12],6,24]"', '"[[-16,16],4,16]"', '"[[-8,8],4,24]"', '"[[-8,8],8,24]"', '"[[-8,8],8,36]"']:
# for inhibit_length in [2,4,8,16,32]:
    #for inhibit_length in [2]:
    for inhibit_length in [4,8,16,32]:
        for img_name in img_ids[0:20]:
            # for kernel in ['laplacian_and_avg_or_avg']:
            for kernel in ['flip_laplacian', 'laplacian']: 
                if kernel=='skip_kernel':
                    pass
                else:
                    # Read in the file
                    with open(slurm_template, 'r') as file:
                        filedata = file.read()

                    # Replace the target string
                    filedata = filedata.replace('inhibit_thresh', str(inhibit_thresh))
                    filedata = filedata.replace('img_name', str(img_name))
                    filedata = filedata.replace('inhibit_length', str(inhibit_length))
                    filedata = filedata.replace('kernel_name', str(kernel))

                    # Write the file out again
                    slurm_out = f'run_{img_name}_{inhibit_thresh}_{inhibit_length}_' + kernel + '.slurm'
                    with open(slurm_out, 'w') as file:
                        file.write(filedata)

                    p = subprocess.run(['sbatch', f'{slurm_out}'])
                    print(p)

