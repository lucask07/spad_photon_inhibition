"""
Lucas J. Koerner, koerner.lucas@stthomas.edu
Creat a GIF of images and mask 

"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import kernels_from_x, my_savefig
from bernoulli_inhibit import mask_gif, img_gif, determine_masks, to_intensity, mse, msre
from create_binary_images import open_img 
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
from kernels import spatial_kernels
#from neutompy.metrics.metrics import GMSD # gradient magnitude similarity deviation
from utils import GMSD, GradientMag
from archived.qr_decode import decode_img
from inhibition_captures import InhibitResult, InhibitResults
np.seterr(divide='ignore') # prevent warnings when calculating divide by zero and log(0)
np.seterr(invalid='ignore')

plt.ion()


def summary_gif(figure_dir, figure_prefix, img, task, task_nomask, 
        metrics, mask_final, metric_name='msre', num_imgs=40, fps=3):
    """
        Create a GIF (movie) of the image reconstruction with the 3rd dimension (i.e. time)
        a new frame.

        create another function that is always used to calculated intensity from image and mask
        and that summarizes the total number of photon dections

        Args:
            figure_dir (string): directory to save GIF
            figure_prefix (string): Name for GIF omit extension
            img (np.ndarray): should be 3D
    """
    from matplotlib.animation import FuncAnimation

    max_frames = 1000
    plt_photons = False # otherwise plot measurements

    # num_imgs can be an array of img indices or an int that describes how many steps in log space
    if type(num_imgs) == int:
        fig_range = np.floor(np.logspace(0, np.log10(img.shape[2]-3), num_imgs)).astype(int)
        fig_range = np.unique(fig_range) # remove duplicates 
    fig_range[fig_range == max_frames] = max_frames - 1

    fig,ax=plt.subplots(nrows=3, ncols=3, figsize=(12.8,9.6))

    
    im_out = []

    im_tmp = ax[0,0].imshow(img[:,:,i], cmap='gray')
    im_out.append(im_tmp)

    im_tmp = ax[0,1].imshow(img[:,:,i], cmap='gray')
    im_out.append(im_tmp)
    
    im_tmp = ax[0,2].imshow(metrics['mask']['gmsd_map'][i], cmap='gray')
    im_tmp.set_clim([0,1])
    im_out.append(im_tmp)

    im_tmp = ax[1,0].imshow(mask_final[:,:,i], cmap='gray')
    im_out.append(im_tmp)
    
    im_tmp = ax[1,0].imshow(mask_final[:,:,i], cmap='gray')
    im_out.append(im_tmp)
    
    ax[1,2].remove()

    # two lines
    merge_metric = np.concatenate((metrics['no_mask'][metric_name], metrics['mask'][metric_name]))
    max_metric = np.nanmax(merge_metric)
    min_metric = np.nanmin(merge_metric)
    if metric_name == 'qr_found':
        line1a, = ax[2,0].semilogx([],[],label='inhibit', marker='*', linestyle='None')
        line1b, = ax[2,0].semilogx([],[],label='conv.', marker='o', linestyle='None')
        ax[2,0].set_ylim([-0.1, 1.1]) # scaling is from 0 to 1
    else:
        line1a, = ax[2,0].loglog([],[],label='inhibit', marker='*', linestyle='None')
        line1b, = ax[2,0].loglog([],[],label='conv.', marker='o', linestyle='None')
        ax[2,0].set_ylim([min_metric, max_metric])
    ax[2,0].set_ylabel(metric_name)
    ax[2,0].set_xlim([1,np.max(fig_range)])
    ax[2,0].set_xlabel('Frames')
    ax[2,0].legend()
 
    if metric_name == 'qr_found':
        line2a, = ax[2,1].semilogx([],[],label='inhibit', marker='*', linestyle='None')
        line2b, = ax[2,1].semilogx([],[],label='conv.', marker='o', linestyle='None')
        ax[2,1].set_ylim([-0.1, 1.1]) # scaling is from 0 to 1
        qr_code_offset = 0.05
    else: 
        line2a, = ax[2,1].loglog([],[], label='inhibit', marker='*')
        line2b, = ax[2,1].loglog([],[], label='conv.', marker='o')
        ax[2,1].set_ylim([min_metric, max_metric])
        qr_code_offset = 0.00

    one_frame_ph = metrics['no_mask']['photons'][0]
    ax[2,1].set_xlim([one_frame_ph, np.max(metrics['no_mask']['photons'])])
    ax[2,1].set_xlabel('photons')
    ax[2,1].set_ylabel(metric_name)
    ax[2,1].legend()
    
    one_frame_ph = metrics['no_mask']['measurements'][0]

    if metric_name == 'qr_found':
        line3a, = ax[2,2].semilogx([],[],label='inhibit', marker='*', linestyle='None')
        line3b, = ax[2,2].semilogx([],[],label='conv.', marker='o', linestyle='None')
        ax[2,2].set_ylim([-0.1, 1.1]) # scaling is from 0 to 1
    else: 
        line3a, = ax[2,2].loglog([],[], label='inhibit', marker='*')
        line3b, = ax[2,2].loglog([],[], label='conv.', marker='o')
        ax[2,2].set_ylim([min_metric, max_metric])

    ax[2,2].set_xlim([one_frame_ph, np.max(metrics['no_mask']['measurements'])])
    ax[2,2].set_xlabel('measurements')
    ax[2,2].set_ylabel(metric_name)
    ax[2,2].legend()

    im_out.append(line1a)
    im_out.append(line1b)
    im_out.append(line2a)
    im_out.append(line2b)
    im_out.append(line3a)
    im_out.append(line3b)
    print(f'Length of artists {len(im_out)}')

    def animate(i):
        print(i)
        im_out[0] = ax[0,0].imshow(img[:,:,i], cmap='gray')
        ax[0,0].set_title(f'Image: {i}')

        p = np.divide(task[i]['counts'], task[i]['unmasked'])
        int_bern = to_intensity(p, skip=to_intensity_skip)

        perc_det_ph = np.sum(metrics['mask']['photons'][0:i])/np.sum(metrics['no_mask']['photons'][0:i])*100

        im_out[1] = ax[0,1].imshow(int_bern, cmap='gray')
        ax[0,1].set_title(f'Composite Frame: 0-{i}. %ph={perc_det_ph:0.3g}')


        im_out[2] = ax[0,2].imshow(metrics['mask']['gmsd_map'][i], cmap='gray')
        ax[0,2].set_title(f'GMS')

        im_out[3] = ax[1,0].imshow(mask_final[:,:,i], cmap='gray')
        ax[1,0].set_title(f'Mask Frame: {i}')
        
        try:
            avg_mask = np.mean(mask_final[:,:,0:i], axis=2) 
        except:
            avg_mask = mask_final[:,:,0:i]
        im_out[4] = ax[1,1].imshow(avg_mask, cmap='gray')
        ax[1,1].set_title(f'Composite Mask: 0-{i}')
        
        # must have x and y data for set_data
        im_out[5].set_data(np.arange(i)+1, metrics['mask'][metric_name][0:i]+qr_code_offset)
        im_out[6].set_data(np.arange(i)+1, metrics['no_mask'][metric_name][0:i]-qr_code_offset)
        im_out[7].set_data(metrics['mask']['photons'][0:i], metrics['mask'][metric_name][0:i]+qr_code_offset)
        im_out[8].set_data(metrics['no_mask']['photons'][0:i], metrics['no_mask'][metric_name][0:i]-qr_code_offset)
        im_out[9].set_data(metrics['mask']['measurements'][0:i], metrics['mask'][metric_name][0:i]+qr_code_offset)
        im_out[10].set_data(metrics['no_mask']['measurements'][0:i], metrics['no_mask'][metric_name][0:i]-qr_code_offset)
        
        return im_out

    # interval is in ms (fps seems more impactful)
    #  func: callable run every frame
    #  frames: iterable 
    anim = FuncAnimation(fig, func=animate, frames=fig_range, interval=200, blit=True)

    #anim.save(os.path.join(figure_dir, f'{figure_prefix}_masks.mp4'), writer='ffmpeg', fps=fps)
    anim.save(os.path.join(figure_dir, f'{figure_prefix}_{metric_name}_composite.gif'), fps=fps)


def metric_plots(metrics, figure_dir, prob_img, unmasked):
    for vs in ['photons', 'measurements']:
        
        xlabel = {'photons':'detections', 'measurements': 'measurements'}

        fig,ax=plt.subplots(2,2)
        ax[0,0].loglog(metrics['no_mask']['mse'], marker='o', label='conv.')
        ax[0,0].loglog(metrics['mask']['mse'], marker='*', label='inhibit')
        ax[0,0].legend()

        ax[0,1].loglog(metrics['no_mask']['msre'], marker='o', label='conv.')
        ax[0,1].loglog(metrics['mask']['msre'], marker='*', label='inhibit')

        ax[1,0].loglog(metrics['no_mask']['ssim'], marker='o', label='conv.')
        ax[1,0].loglog(metrics['mask']['ssim'], marker='*', label='inhibit')

        fig,ax=plt.subplots(2,2)
        ax[0,0].loglog(metrics['no_mask'][vs],metrics['no_mask']['mse'], marker='o', label='conv.')
        ax[0,0].loglog(metrics['mask'][vs],metrics['mask']['mse'], marker='*', label='inhibit')
        ax[0,0].set_ylabel('MSE')
        ax[0,0].set_xlabel(xlabel[vs])
        ax[0,0].legend()

        ax[0,1].loglog(metrics['no_mask'][vs],metrics['no_mask']['msre'], marker='o', label='conv.')
        ax[0,1].loglog(metrics['mask'][vs],metrics['mask']['msre'], marker='*', label='inhibit')
        ax[0,1].set_ylabel('MSRE')
        ax[0,1].set_xlabel(xlabel[vs])
        ax[0,1].legend()

        ax[1,0].loglog(metrics['no_mask'][vs], metrics['no_mask']['ssim'], marker='o', label='conv.')
        ax[1,0].loglog(metrics['mask'][vs], metrics['mask']['ssim'], marker='*', label='inhibit')
        ax[1,0].set_ylabel('SSIM')
        ax[1,0].set_xlabel(xlabel[vs])
        ax[1,0].legend()
        
        ax[1,1].plot(prob_img.ravel(), unmasked.ravel(), linestyle = 'None', marker='*')
        ax[1,1].set_ylabel('Meas.')
        ax[1,1].set_xlabel('p')

        my_savefig(fig, figure_dir, f'summary_tests_vs{vs}')

        return fig, ax

def main(img_dir='bernoulli_images/BSDS500/data/images/test', img_name='140088', extension='.jpg', kernel_name='neighbor8', thresholds=[0.1,1.0,10.0], 
    inhibit_thresh=6, inhibit_length=8, thumbnail=False, roi_x=(16,100), roi_y=(16,100), max_frames=None):
    
    '''
    evaluate inhibtion over multiple exposure times (thresholds) and create an InhibitResults object 
    once masks are determined uses calc_metrics_bracket to create plots 

    '''
    
    # directory for source images
    home = os.path.expanduser('~')
    figure_output_dir = os.path.join(home, 'bernoulli_inhibit/tests_probability_images/')
    print(f'Figure output directory: {figure_output_dir}')
    
    img_dir = os.path.join(home, img_dir)

    to_intensity_skip = True
    QR_CODE = False
        
    irs = InhibitResults(img_path=os.path.join(img_dir, img_name + extension))
    irs.fig_dir = figure_output_dir

    for threshold in [f'(None,_{th})' for th in thresholds]:
        print(f'Threshold {threshold}')
        binary_img_name = f'{img_name}_bernoulli_thresh{threshold}_maxframes1000.npy'

        folder_name = f'kernel_{kernel_name}'
        spatial_kernel = spatial_kernels[kernel_name][0]
        try:
            inhibit_combine_operators = spatial_kernels[kernel_name][2]
        except:
            inhibit_combine_operators = None
        figure_dir = os.path.join(figure_output_dir, 'tests_output', img_name, threshold, folder_name)
        print(figure_dir)

        # make directories if needed 
        for d in [figure_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        thresh_str = threshold.replace('(None,_','').replace(')','')
        ppp = float(thresh_str) # photons per pixel
        full_prec_frame = open_img(img_name + extension, img_dir, roi_x=roi_x, roi_y=roi_y, Thumbnail=thumbnail)[0]
        
        # exposure image of full precision
        full_prec_frame = full_prec_frame/np.mean(full_prec_frame)*ppp
        # if all statistics and tasks should be completed on the probability iamges 
        full_prec_frame = 1 - np.exp(-full_prec_frame)

        # the mean of this is a stochastic probability image 
        frame = np.load(os.path.join(img_dir, binary_img_name))
        if thumbnail:
            frame = frame[ roi_x[1]:(roi_x[0]+roi_x[1]), roi_y[1]:(roi_y[0]+roi_y[1])]

        # Make this routine a function -------
        fig,ax=plt.subplots(2,2)
        int_bern = to_intensity(np.mean(frame,axis=2), skip=to_intensity_skip) 
        im = ax[0,0].imshow(full_prec_frame)
        vmin,vmax = im.get_clim()
        ax[0,1].imshow(int_bern, vmin=vmin,vmax=vmax)

        # intensity histograms
        (n,bins,patches)=ax[1,0].hist(int_bern.ravel(), bins=30, label='bern.')
        ax[1,0].hist(full_prec_frame.ravel(), bins=bins, label='full')
        ax[1,0].legend()

        # probabiliy histogram 
        (n,bins,patches)=ax[1,1].hist(1-np.exp(-int_bern.ravel()), bins=30)
        ax[1,1].hist(1-np.exp(-full_prec_frame.ravel()), bins=bins)
        # ------------------------------------
        inhibit_threshold = inhibit_thresh
        # ---------- apply masks 
        ir = InhibitResult(img_path=os.path.join(img_dir, img_name + extension), ppp=ppp, roi_x=roi_x, roi_y=roi_y, thumbnail=thumbnail)
        # time_filter defaults to [1,1,1,1] -- length of 4
        task, task_nomask, mask_final, summary_stats = determine_masks(frame, 
                spatial_kernel=spatial_kernel,
                threshold=inhibit_threshold, 
                inhibition_length=inhibit_length, 
                random_mask=False, 
                inhibit_result=ir, 
                combine_operators=inhibit_combine_operators)
       
        if max_frames is None:
            max_frames = np.max(list(task.keys())) # task is a dictionary with keys of the frame numbers
        
        ir.calc_metrics(full_prec_frame=full_prec_frame, 
            to_intensity_skip=to_intensity_skip, 
            QR_CODE=QR_CODE, SAVE_GMSD_MAP=True, max_frames=max_frames)

        irs.append(ir) # append the inhibit result 
        plt.close('all')
        i=0
        if 0:
            summary_gif(figure_dir, img_name, frame, ir.captures, ir.captures_nomask, ir.metrics, mask_final, 'msre', num_imgs=100, fps=3)
            summary_gif(figure_dir, img_name, frame, ir.captures, ir.captures_nomask, ir.metrics,mask_final, 'mse', num_imgs=100, fps=3)
            summary_gif(figure_dir, img_name, frame, ir.captures, ir.captures_nomask, ir.metrics, mask_final, 'ssim', num_imgs=100, fps=3)
            summary_gif(figure_dir, img_name, frame, ir.captures, ir.captures_nomask, ir.metrics,mask_final, 'gmsd', num_imgs=100, fps=3)
        if QR_CODE:
            summary_gif(figure_dir, img_name, frame, ir.captures, ir.captures_nomask, ir.metrics, mask_final, 'qr_found', num_imgs=100, fps=3)

        max_frame = 999
        #fig, ax = metric_plots(metrics, figure_dir, prob_img=int_bern, 
        #    unmasked=task[max_frame]['unmasked'])

        total_photons = np.sum(task_nomask[max_frame]['counts'])
        detected_photons = np.sum(task[max_frame]['counts'])

        print(f'Percent detected: {detected_photons/total_photons}')

    irs.calc_metrics_bracket(full_prec_frame=None,
                            to_intensity_skip=True, 
                            QR_CODE=QR_CODE,
                            SAVE_GMSD_MAP=False,
                            bracket_wts='snr',
                            max_frames=max_frames) # was 'snr'

    # check if the metrics are the same -- only works if bracketing is effectively disabled by setting wts = 1 just for the ppp=1 capture 
    # print(f'Checking that metrics of ir and irs: {np.allclose(irs.metrics["mask"]["mse"], ir.metrics["mask"]["mse"])}')

    # save the list of inhibition results 
    # irs.pickle(f'inhibition_results_bracket.pkl')

    return irs 
