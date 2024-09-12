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
import os

# templates without the compatibility layer
slurm_template='run_compareimgs_hdr.slurm'

# templates with the compatibility layer (no longer supported since Rocky8 works now without compatibility layer)
wrapper_template_compat = 'run_compareimgs_hdr_compat.sh' # this will become the wrapper .sh script 
slurm_template_compat = 'run_compareimgs_hdr_compat.slurm' # this is the slurm script will only change the name of the wrapper script that it runs 

inhibit_length = 30
inhibit_thresh = 6

COMPAT_OS = False # are we using the centos7 compatibility layer after the MSI transition to Rocky8? 

# skip since these ran the first time 
# '9C4A0599-3e8dd0df28_resize',  'AG8A7597-7857c02217_resize',  'AG8A3343-c4a982236d_resize',

for inhibit_thresh in [2,6,12,24]:
    for inhibit_length in [4,8,16,32,80]:
        for img_name in ['9C4A1696-a361781a04_resize', '9C4A3335-edf32a8ffe_resize', '9C4A3821-5e85168ed1_resize', '9C4A6135-c776b00d90_resize', 'AG8A2979-92ff107fdd_resize',  'AG8A5920-73af00b822_resize', 'AG8A6813-d486e1621a_resize']:
            for kernel in ['neighbor8', 'single_pix_bright']:
                if (kernel=='neighbor8' or kernel=='single_pix_bright') and inhibit_thresh>8:
                    pass # there will be no/limited inhibition in this scenario so skip. More generally can compare sum(Ks) to inhibit_thresh
                else:
                    if not COMPAT_OS:
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
                    else:
                        # for the compatibility OS need to create two files. One is a script wrapper and one is a SLURM script. 
                        # these both need to be modified for each configuration since
                        #   1) the script has varying parameters and 
                        #   2) the SLURM needs the unique name of the script 

                        # Read in the file
                        with open(wrapper_template_compat, 'r') as file:
                            filedata = file.read()

                        # Replace the target string
                        filedata = filedata.replace('inhibit_thresh', str(inhibit_thresh))
                        filedata = filedata.replace('img_name', str(img_name))
                        filedata = filedata.replace('inhibit_length', str(inhibit_length))
                        filedata = filedata.replace('kernel_name', str(kernel))

                        # Write the file out again
                        wrapper_out = f'run_{img_name}_{inhibit_thresh}_{inhibit_length}_' + kernel + '.sh'
                        with open(wrapper_out, 'w') as file:
                            file.write(filedata)

                        # ensure that the wrapper script is executable 
                        os.chmod(wrapper_out, 0o777) # leading zero has Python treat as octal 

                        # read and modify the SLURM file to run the correct wrapper script 
                        with open(slurm_template_compat, 'r') as file:
                            filedata = file.read()
                        # Replace the target string
                        filedata = filedata.replace('wrapper_script', str(wrapper_out))

                        slurm_out = f'run_{img_name}_{inhibit_thresh}_{inhibit_length}_' + kernel + '.slurm'
                        with open(slurm_out, 'w') as file:
                            file.write(filedata)
                        
                        p = subprocess.run(['sbatch', f'{slurm_out}'])
                        print(p)

