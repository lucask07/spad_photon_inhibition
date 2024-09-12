#!/bin/bash -l
#SBATCH --time=2:00:00 
#SBATCH --ntasks=4
#SBATCH --mem=128g
#SBATCH --mail-type=NONE
#SBATCH --mail-user=lkoerner@umn.edu 
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1

HED_DIR=/home/lkoerner/lkoerner/hed

module load matlab/R2021a
module load python
# python packages are already installed
# pip install -r requirements.txt
# Template file, edit img_idx and kernel_name

python ${HED_DIR}/hed_sandbox.py --caffe_model ${HED_DIR}/data/hed_pretrained_bsds.py36pickle --output ${HED_DIR}/output-inhibit-3 --test --dataset ${HED_DIR}/data/HED-Inhibit2
