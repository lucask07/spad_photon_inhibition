"""
Lucas J. Koerner, koerner.lucas@stthomas.edu
Jan 2023 

objects to organize inhibition results 

"""
import os
import sys
import copy
import pickle as pkl
import numpy as np
from bernoulli_inhibit import to_intensity, mse, msre
from utils import GMSD, GradientMag
from archived.qr_decode import decode_img
from create_binary_images import open_img 
from skimage.metrics import structural_similarity as ssim

class InhibitResult():

    def __init__(self, img_path, ppp, spatial_kernel=None, time_kernel=None, thresh=None, length=None,
            captures=None, captures_nomask=None, mask=None, roi_x=None, roi_y=None, thumbnail=False):

        # metadata-like
        self.img_path = img_path # full path and filename 
        self.ppp = ppp
        self.spatial_kernel = spatial_kernel 
        self.time_kernel = time_kernel
        self.inhibit_thresh = thresh
        self.inhibit_length = length
       
        self.thumbnail=thumbnail
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.num_pixels = np.product(np.shape(self.load_img()))

        # results
        self.captures = captures  # historically this was called task. 
        self.captures_nomask = captures_nomask
        self.mask = mask

        # processed results
        self.metrics ={}
        self.metrics['mask'] = {}
        self.metrics['no_mask'] = {}

    def load_img(self):

        img_d, img_p = os.path.split(self.img_path)
        img = open_img(img_p, img_d, roi_x=self.roi_x, roi_y=self.roi_y, Thumbnail=self.thumbnail)[0]
        return img/np.mean(img)*self.ppp

    def get_p(self, frame_num, with_mask=True):
        """
        get probability image
        
        """       
        if with_mask:
            masked='captures'
        else:
            masked='nomask_captures'

        return np.divide(getattr(ir, masked)[frame_num]['counts'], getattr(ir, masked)[frame_num]['unmasked'])

    def calc_metrics(self, full_prec_frame=None, to_intensity_skip=True, QR_CODE=False, SAVE_GMSD_MAP=False, max_frames=None):

        # --------- create containers for metrics 
        metrics = {}
        metrics['mask'] = {}
        metrics['no_mask'] = {}

        metric_names = ['mse', 'msre', 'mse_grad', 'ssim', 'photons', 'measurements', 'photons_pp', 'measurements_pp', 'gmsd','gms_avg', 'qr_found']
        for m in ['mask', 'no_mask']:
            for n in metric_names:
                metrics[m][n]=np.array([])

        if full_prec_frame is None:
            full_prec_frame = self.load_img() # this is an intensity image scaled to ppp 
            if to_intensity_skip:
                full_prec_frame = 1-np.exp(-full_prec_frame) # convert to probability 

        task = self.captures
        task_nomask = self.captures_nomask

        if max_frames == None:
            max_frames = np.max(list(task.keys())) # task is a dictionary with keys of the frame numbers
        full_prec_edge = GradientMag(full_prec_frame)

        metrics['no_mask']['gmsd_map'] = {} # this is an image so cannot be a simple array of quantities as above
        metrics['mask']['gmsd_map'] = {}
        for i in range(max_frames):
            p = np.divide(task_nomask[i]['counts'], task_nomask[i]['unmasked'])
            int_bern = to_intensity(p, skip=to_intensity_skip)
            int_bern_edge = GradientMag(int_bern)
            
            metrics['no_mask']['mse'] = np.append(metrics['no_mask']['mse'], mse(int_bern, full_prec_frame))
            metrics['no_mask']['msre'] = np.append(metrics['no_mask']['msre'], msre(full_prec_frame, int_bern))
            metrics['no_mask']['ssim'] = np.append(metrics['no_mask']['ssim'], ssim(int_bern, full_prec_frame, data_range=1.0))
            metrics['no_mask']['mse_grad'] = np.append(metrics['no_mask']['mse_grad'], mse(full_prec_edge, int_bern_edge))
            
            gmsd_val, gms_avg, gmsd_map = GMSD(full_prec_frame, int_bern, ret_map=True)
            if SAVE_GMSD_MAP: 
                metrics['no_mask']['gmsd_map'][i] = gmsd_map
            metrics['no_mask']['gmsd'] = np.append(metrics['no_mask']['gmsd'], gmsd_val)
            metrics['no_mask']['gms_avg'] = np.append(metrics['no_mask']['gms_avg'], gms_avg)

            metrics['no_mask']['photons'] = np.append(metrics['no_mask']['photons'], np.sum(task_nomask[i]['counts']))
            metrics['no_mask']['measurements'] = np.append(metrics['no_mask']['measurements'], np.sum(task_nomask[i]['unmasked']))
            metrics['no_mask']['photons_pp'] = np.append(metrics['no_mask']['photons_pp'], np.sum(task_nomask[i]['counts'])/self.num_pixels)
            metrics['no_mask']['measurements_pp'] = np.append(metrics['no_mask']['measurements_pp'], np.sum(task_nomask[i]['unmasked'])/self.num_pixels)
           
            if QR_CODE:
                retval, points, straight_qrcode = decode_img(int_bern)
                if len(retval)>0:
                    qr_found = 1
                else:
                    qr_found = 0
            else:
                qr_found = 0
            metrics['no_mask']['qr_found'] = np.append(metrics['no_mask']['qr_found'], qr_found)
            
            # repeat for the masked image

            p = np.divide(task[i]['counts'], task[i]['unmasked'])
            int_bern = to_intensity(p, skip=to_intensity_skip)
            int_bern_edge = GradientMag(int_bern)
            
            metrics['mask']['mse'] = np.append(metrics['mask']['mse'], mse(int_bern, full_prec_frame))
            metrics['mask']['msre'] = np.append(metrics['mask']['msre'], msre(full_prec_frame, int_bern))
            metrics['mask']['ssim'] = np.append(metrics['mask']['ssim'], ssim(int_bern, full_prec_frame, data_range=1.0))
            metrics['mask']['mse_grad'] = np.append(metrics['mask']['mse_grad'], mse(full_prec_edge, int_bern_edge))
            gmsd_val, gms_avg, gmsd_map = GMSD(full_prec_frame, int_bern, ret_map=True)
            if SAVE_GMSD_MAP: 
                metrics['mask']['gmsd_map'][i] = gmsd_map # this is an image 
            metrics['mask']['gms_avg'] = np.append(metrics['mask']['gms_avg'], gms_avg)
            metrics['mask']['gmsd'] = np.append(metrics['mask']['gmsd'], gmsd_val)
            metrics['mask']['photons'] = np.append(metrics['mask']['photons'], np.sum(task[i]['counts']))
            metrics['mask']['measurements'] = np.append(metrics['mask']['measurements'], np.sum(task[i]['unmasked']))
            metrics['mask']['photons_pp'] = np.append(metrics['mask']['photons_pp'], np.sum(task[i]['counts'])/self.num_pixels)
            metrics['mask']['measurements_pp'] = np.append(metrics['mask']['measurements_pp'], np.sum(task[i]['unmasked'])/self.num_pixels)

            if QR_CODE:
                retval, points, straight_qrcode = decode_img(int_bern)
                if len(retval)>0:
                    qr_found = 1
                else:
                    qr_found = 0
            else:
                qr_found = 0
            metrics['mask']['qr_found'] = np.append(metrics['mask']['qr_found'], qr_found)

            self.metrics['mask'] = metrics['mask']
            self.metrics['no_mask'] = metrics['no_mask']


class InhibitResults(list):
    """
    a list of multiple InhibitResult 
    with additional properties 
    """
    def __init__(self, img_path):
        self.img_path = img_path
        self.metrics = {} # with exposure bracketing will have metrics for a list of inhibitionresults
        self.fig_dir = None # location of saved figures 

    def find_ppp(self, ppp):
        return [x for x in self if x.ppp==ppp]

    def pickle(self, filename, del_images=True):
        if del_images:
            tmp = copy.deepcopy(self)
            for i,k in enumerate(tmp):
                tmp[i].captures = None
                tmp[i].captures_nomask = None
                tmp[i].mask = None
                tmp[i].metrics['mask']['gmsd_map'] = None
                tmp[i].metrics['no_mask']['gmsd_map'] = None
        else:
            tmp = self

        with open(os.path.join(tmp.fig_dir, filename), 'wb') as f:
            pkl.dump(tmp, f, protocol=pkl.HIGHEST_PROTOCOL)

    def sum_metric(self, metric, frame_num):
        
        res_mask = np.sum([np.sum(ir.captures[frame_num][metric]) for ir in self])
        res_nomask = np.sum([np.sum(ir.captures_nomask[frame_num][metric]) for ir in self])

        return res_mask, res_nomask


    def save_nomask_mask(self, filename, metric='photons_pp', metric_val=5, bracket=True, to_intensity=False):

        ''' save an image for both mask and no mask at an equal metric value 
        '''

        mask_num = np.argmin(np.abs(self.metrics['mask'][metric] - metric_val))  
        m_mask = self.metrics['mask'][metric][mask_num]

        no_mask_num = np.argmin(np.abs(self.metrics['no_mask'][metric] - metric_val))  
        m_no_mask = self.metrics['no_mask'][metric][mask_num]

        if bracket:
            p_mask, flux_wtd_mask, wts = self.combine_exposures(mask_num, masked='captures')
            p_no_mask, flux_wtd_mask, wts = self.combine_exposures(no_mask_num, masked='captures_nomask')
            num_brackets = len(self)

        f1n = os.path.join(self.fig_dir, filename + '_mask')
        with open(f1n, 'wb') as f:
            pkl.dump(p_mask, f, protocol=pkl.HIGHEST_PROTOCOL)

        f2n = os.path.join(self.fig_dir, filename + '_nomask')
        with open(f2n, 'wb') as f:
            pkl.dump(p_no_mask, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        return f1n, f2n, m_mask, m_no_mask

    def calc_metrics_bracket(self, 
                full_prec_frame=None, 
                to_intensity_skip=True, 
                QR_CODE=False, 
                SAVE_GMSD_MAP=False,
                bracket_wts='snr',
                max_frames=None):

        # --------- create containers for metrics 
        metrics = {}
        metrics['mask'] = {}
        metrics['no_mask'] = {}

        num_pixels = self[0].num_pixels

        metric_names = ['mse', 'msre', 'mse_grad', 'ssim', 'photons', 'measurements', 'photons_pp', 'measurements_pp','gmsd', 'gms_avg', 'qr_found']
        for m in ['mask', 'no_mask']:
            for n in metric_names:
                metrics[m][n]=np.array([])

        if full_prec_frame is None:
            full_prec_frame = self.find_ppp(1)[0].load_img() # this is an intensity image with ppp=1, bracketing is referenced to ppp=1 so this exposure time is required 
            if to_intensity_skip:
                full_prec_frame = 1-np.exp(-full_prec_frame) # convert to probability 
        if max_frames==None:
            max_frames = np.max(list(self[0].captures.keys()))
        full_prec_edge = GradientMag(full_prec_frame)

        metrics['no_mask']['gmsd_map'] = {} # this is an image so cannot be a simple array of quantities as above
        metrics['mask']['gmsd_map'] = {}
        for i in range(max_frames):
            p,flux,wts = self.combine_exposures(frame_num=i, masked='captures_nomask', weighting=bracket_wts)
            int_bern = to_intensity(p, skip=to_intensity_skip)
            int_bern_edge = GradientMag(int_bern)
            
            metrics['no_mask']['mse'] = np.append(metrics['no_mask']['mse'], mse(int_bern, full_prec_frame))
            metrics['no_mask']['msre'] = np.append(metrics['no_mask']['msre'], msre(full_prec_frame, int_bern))
            metrics['no_mask']['ssim'] = np.append(metrics['no_mask']['ssim'], ssim(int_bern, full_prec_frame, data_range=1.0))
            metrics['no_mask']['mse_grad'] = np.append(metrics['no_mask']['mse_grad'], mse(full_prec_edge, int_bern_edge))
            
            gmsd_val, gms_avg, gmsd_map = GMSD(full_prec_frame, int_bern, ret_map=True)
            if SAVE_GMSD_MAP: 
                metrics['no_mask']['gmsd_map'][i] = gmsd_map
            metrics['no_mask']['gmsd'] = np.append(metrics['no_mask']['gmsd'], gmsd_val)
            metrics['no_mask']['gms_avg'] = np.append(metrics['no_mask']['gms_avg'], gms_avg)
            metrics['no_mask']['photons'] = np.append(metrics['no_mask']['photons'], 
                self.sum_metric(metric='counts', frame_num=i)[1])
            metrics['no_mask']['measurements'] = np.append(metrics['no_mask']['measurements'], 
                self.sum_metric(metric='unmasked', frame_num=i)[1])
            metrics['no_mask']['photons_pp'] = np.append(metrics['no_mask']['photons_pp'], 
                self.sum_metric(metric='counts', frame_num=i)[1]/num_pixels)
            metrics['no_mask']['measurements_pp'] = np.append(metrics['no_mask']['measurements_pp'], 
                self.sum_metric(metric='unmasked', frame_num=i)[1]/num_pixels)
            
            if QR_CODE:
                retval, points, straight_qrcode = decode_img(int_bern)
                if len(retval)>0:
                    qr_found = 1
                else:
                    qr_found = 0
            else:
                qr_found = 0
            metrics['no_mask']['qr_found'] = np.append(metrics['no_mask']['qr_found'], qr_found)
            
            # repeat for the masked image
            p,flux,wts = self.combine_exposures(frame_num=i, masked='captures', weighting=bracket_wts)
            int_bern = to_intensity(p, skip=to_intensity_skip)
            int_bern_edge = GradientMag(int_bern)
            
            metrics['mask']['mse'] = np.append(metrics['mask']['mse'], mse(int_bern, full_prec_frame))
            metrics['mask']['msre'] = np.append(metrics['mask']['msre'], msre(full_prec_frame, int_bern))
            metrics['mask']['ssim'] = np.append(metrics['mask']['ssim'], ssim(int_bern, full_prec_frame, data_range=1.0))
            metrics['mask']['mse_grad'] = np.append(metrics['mask']['mse_grad'], mse(full_prec_edge, int_bern_edge))
            gmsd_val, gms_avg, gmsd_map = GMSD(full_prec_frame, int_bern, ret_map=True)
            if SAVE_GMSD_MAP: 
                metrics['mask']['gmsd_map'][i] = gmsd_map # this is an image 
            metrics['mask']['gmsd'] = np.append(metrics['mask']['gmsd'], gmsd_val)
            metrics['mask']['gms_avg'] = np.append(metrics['mask']['gms_avg'], gms_avg)
            metrics['mask']['photons'] = np.append(metrics['mask']['photons'],
                self.sum_metric(metric='counts', frame_num=i)[0])
            metrics['mask']['measurements'] = np.append(metrics['mask']['measurements'], 
                self.sum_metric(metric='unmasked', frame_num=i)[0])
            metrics['mask']['photons_pp'] = np.append(metrics['mask']['photons_pp'],
                self.sum_metric(metric='counts', frame_num=i)[0]/num_pixels)
            metrics['mask']['measurements_pp'] = np.append(metrics['mask']['measurements_pp'], 
                self.sum_metric(metric='unmasked', frame_num=i)[0]/num_pixels)

            if QR_CODE:
                retval, points, straight_qrcode = decode_img(int_bern)
                if len(retval)>0:
                    qr_found = 1
                else:
                    qr_found = 0
            else:
                qr_found = 0
            metrics['mask']['qr_found'] = np.append(metrics['mask']['qr_found'], qr_found)

            self.metrics['mask'] = metrics['mask']
            self.metrics['no_mask'] = metrics['no_mask']

    def combine_exposures(self,
                            frame_num=999,
                            masked='captures',
                            weighting='snr'):
        """
        parameters:
            frame_num: int 
            masked: str: 'captures' (masked data) or 'captures_nomask' (for unmasked data)
            weighting: str: 'intensity' or array of same length as the InhibitionResults list 

        """
        snr_h_mat = np.zeros( np.shape(getattr(self[0], masked)[frame_num]['counts']) + tuple([len(self)]))
        for idx,ir in enumerate(self):
            p = np.divide(getattr(ir, masked)[frame_num]['counts'], getattr(ir, masked)[frame_num]['unmasked'])
            
            snr_h = np.multiply(-np.log(1-p), np.multiply( np.sqrt(np.divide(1-p, p)), np.sqrt(getattr(ir, masked)[frame_num]['unmasked'])))
            snr_h_mat[:,:,idx] = snr_h 
            
            # if p is 0 or 1 set SNR to 0
            snr_h_mat[p==0,idx]=0
            snr_h_mat[p==1,idx]=0  # goal is to have p=1 if all exposures have p=1 (otherwise p=1 should not be weighted) 
            
            # print(f'{idx}: Inv var {inv_var[i,j,idx]}, probability {p[i,j]}')

        if isinstance(weighting, (list, np.ndarray)): # assume weighting is a vector 
            wts = weighting
        elif weighting == 'snr':
            wts = np.divide(snr_h_mat**2, np.nansum(snr_h_mat**2, axis=2)[:,:,None]) # normalize the weights 
            wts[np.isnan(wts)]=0 
        else:
            print('error')

        flux_wtd = np.zeros(np.shape(p))
        
        all_ones = np.ones(np.shape(p), dtype=bool) # determine if every exposure time had a probability of 1
        p_ateach = []

        ppp_arr = np.array([ir.ppp for ir in self])
        ppp_min = np.min(ppp_arr)

        for idx,ir in enumerate(self):
            p = np.divide(getattr(ir, masked)[frame_num]['counts'], getattr(ir, masked)[frame_num]['unmasked'])
            p_ateach.append(p)
            all_ones = all_ones & (p==1)
            h_flux = to_intensity(p)/ir.ppp # calculate h and convert to flux 
            #print(f'H flux at {ir.ppp} = {h_flux}')
            
            np.nan_to_num(h_flux, copy=False, posinf=0) # replace nan and inf with 0. These should be from a p==0 or p==1 
            # print(f'{idx}: wts {wts[i,j,idx]}, probability {p[i,j]}')
            
            if isinstance(weighting, (list, np.ndarray)): # assume weighting is a vector 
                flux_wtd+= np.multiply(h_flux, wts[idx])
            elif weighting == 'snr': 
                flux_wtd += np.multiply(h_flux, wts[:,:,idx])
            else:
                print('error')
        p = 1-np.exp(-flux_wtd)
        # replace p with 1 for any that are all 1s at all the exposure times 
        p[all_ones]=1
        # find and correct other instances where all weights are 0 
        # TODO: generalize to any exposure time -- MLE solution specific to 0.1,1,10  
        p[(p_ateach[0]==0) & (p_ateach[1]==1) & (p_ateach[2]==1)] = 0.909 
        p[(p_ateach[0]==0) & (p_ateach[1]==0) & (p_ateach[2]==1)] = 0.2063 
    
        return p, flux_wtd, wts 


def load_pickle(filename):

    with open(filename, 'rb') as f:
        irs = pkl.load(f)
    return irs
