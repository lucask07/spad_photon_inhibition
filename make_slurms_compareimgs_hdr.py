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

# slurm template that is edited by this script 
slurm_template='run_compareimgs_hdr.slurm'

inhibit_length = 30
inhibit_thresh = 6

# These inhibit thresh values allow for the 2 policy apporach for edge detection
# for inhibit_thresh in ['"[[-2,2],2,6]"', '"[[-4,4],2,6]"', '"[[-8,8],2,6]"', '"[[-12,12],2,6]"']:
# for inhibit_thresh in ['"[[-16,16],2,6]"', '"[[-20,20],2,6]"', '"[[-24,24],2,6]"', '"[[-28,28],2,6]"', '"[[-32,32],2,6]"', '"[[-36,36],2,6]"']:


#for inhibit_thresh in [1,2,4,6,10,12,24]:
for inhibit_thresh in [12,6]:
#    for inhibit_length in [2,4,8,10,20,32,64,128]:
    for inhibit_length in [32,12]:
        for img_name in ['vatican_road_8k_smallcrop','workshop_4k_crop', 'vulture_hide_8k_crop3']: 
#        for img_name in ['vulture_hide_8k_crop2', 'workshop_4k_crop', 'vulture_hide_8k_crop3']: 
#           for kernel in ['flip_laplacian', 'laplacian', 'neighbor8', 'single_pix_bright', 'single_pix_dark', 'large13x13', 'large7x7']: 
            for kernel in ['flip_laplacian']: 
#            for kernel in ['laplacian_and_avg_or_avg']: 
#                if kernel=='neighbor8' and inhibit_thresh>8:
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

