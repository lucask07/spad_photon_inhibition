#!/bin/bash -l
#SBATCH --time=24:00:00 
#SBATCH --ntasks=1
#SBATCH --mem=128g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lkoerner@umn.edu 
#SBATCH -p msismall

module load python3
# pip install -r requirements.txt


# python create_binary_images.py '/home/lkoerner/lkoerner/datasets/100samplesDataset/wideDR10_x4resize/' 0.01
# python create_binary_images.py '/home/lkoerner/lkoerner/datasets/100samplesDataset/wideDR10_x4resize/' 0.1
# python create_binary_images.py '/home/lkoerner/lkoerner/datasets/100samplesDataset/wideDR10_x4resize/' 1.0
# python create_binary_images.py '/home/lkoerner/lkoerner/datasets/100samplesDataset/wideDR10_x4resize/' 10.0
# python create_binary_images.py '/home/lkoerner/lkoerner/datasets/100samplesDataset/wideDR10_x4resize/' 100.0
# python create_binary_images.py '/home/lkoerner/lkoerner/bernoulli_images/hdr/' 10.0
# python create_binary_images.py '/home/lkoerner/lkoerner/bernoulli_images/bsds500/data/images/test/' 0.1 skip 100
# python create_binary_images.py '/home/lkoerner/lkoerner/bernoulli_inhibit/images/qrcode/QR/train/' 10


