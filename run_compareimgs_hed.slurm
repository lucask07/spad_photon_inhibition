#!/bin/bash -l
#SBATCH --time=2:00:00 
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --mail-type=NONE
#SBATCH --mail-user=lkoerner@umn.edu 
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1

HED_DIR=/home/lkoerner/lkoerner/hed
INHIBIT_OUT_DIR=/home/lkoerner/lkoerner/bernoulli_inhibit/tests_probability_images/tests_output_hed
export QT_QPA_PLATFORM=offscreen

module load python3
# python packages are already installed
# pip install -r requirements.txt
# Template file, edit img_idx and kernel_name
python3.8 run_script_compareimgs.py inhibit_length img_name inhibit_thresh kernel_name

# template file edit: hed_out (and the names above) 
python ${HED_DIR}/hed_sandbox.py --caffe_model ${HED_DIR}/data/hed_pretrained_bsds.py36pickle --output ${INHIBIT_OUT_DIR}/img_name/kernel_name/threshinhibit_thresh/lengthinhibit_length/hed --test --dataset ${INHIBIT_OUT_DIR}/img_name/kernel_name/threshinhibit_thresh/lengthinhibit_length/
