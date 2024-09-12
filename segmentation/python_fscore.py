'''
Python framework to run the Berkeley segmentation database fscore analysis 
which is written in MATLAB 

The MATLAB functions fscore_one_img and fscore_one_hdr_img need to be in the directory 
        /home/lkoerner/lkoerner/contour/

'''

import matlab.engine
import numpy as np
import matlab
import os

eng = matlab.engine.start_matlab()
eng.addpath('/home/lkoerner/lkoerner/contour')
eng.configure_paths(nargout=0)

imgdir = 'home/lkoerner/lkoerner/contour/BSDS300/images/test/'
hdr_dir = '/home/lkoerner/lkoerner/bernoulli_inhibit/images/'


def eval_fscore_fname(num_frames, iid='101085'):
        fname = f'{iid}_bernoulli_thresh(None,_1.0)_maxframes1000'
        im = np.load(os.path.join(imgdir, fname + '.npy'))
        t = eng.fscore_one_img(matlab.double(np.mean(im[:,:,0:num_frames], axis=2).tolist()), iid, True, nargout=3)
        fscore = t[0]
        segs = t[1]
        edge_map = t[2]
        return fscore, segs, edge_map


def eval_fscore(im, iid='101085', convert_to_intensity=True):
        t = eng.fscore_one_img(matlab.double(im.tolist()), iid, convert_to_intensity, nargout=3)
        fscore = t[0]
        segs = t[1]
        edge_map = t[2]
        return fscore, segs, edge_map


def eval_fscore_hdr(im, ref_name, roi, convert_to_intensity=True, nthresh=100):
        full_ref_name = os.path.join(hdr_dir, ref_name)
        t = eng.fscore_one_hdr_img(matlab.double(im.tolist()), full_ref_name, matlab.double(roi.tolist()), convert_to_intensity, nthresh, nargout=2)
        fscore = t[0]
        # segs = t[1] # segs is not available and so is not returned
        edge_map = t[1]
        return fscore, edge_map
