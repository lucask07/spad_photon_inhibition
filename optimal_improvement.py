import numpy as np
import os
from analytical.measurement_distribution_opt import optimal_msre, img_mse
from inhibition_captures import InhibitResult
from load_config import config

img_dir = config['bsds_dir'] 
img_name = '148026'
extension = '.jpg'
home = os.path.expanduser('~')
img_dir = os.path.join(home, img_dir)

target_detections_pix = 10

for ppp in [0.05, 0.1, 0.2, 0.5, 1, 5, 10]:
    print('-'*40)
    ir = InhibitResult(img_path=os.path.join(img_dir, img_name + extension), ppp=ppp)
    intensity = ir.load_img()
    p = 1 - np.exp(-intensity)
    p[p==1] = 0.9999
    p[p<1e-6]=1e-6
    msre, Nt_opt, improve_factor = optimal_msre(p, target_detections_pix)
    print(f'ppp = {ppp}; MSRE = {msre}; Improve factor {improve_factor}')
    print(f'conventional MSRE = {msre*improve_factor}')
    print(f'ratio of measurements {np.max(Nt_opt)/np.min(Nt_opt)}')
    print(f'Sum of measurements {np.sum(Nt_opt)}. Average p = {np.mean(p)}')
