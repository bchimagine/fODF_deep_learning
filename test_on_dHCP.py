#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

to run on command line:  python dHCP_one_subject.py gpu_ind subj_ind
gpu_ind:   gpu number to use: 1-3
subj_ind:  subject index: 0-504.  Not all subjects have diffusion data;
           you will get a message and the code wil exit in a few seconds
           if there is no diffusion data.

"""






from __future__ import division

import numpy as np
import os
import sys
#from numpy import dot
#from dipy.core.geometry import sphere2cart
#from dipy.core.geometry import vec2vec_rotmat
#from dipy.reconst.utils import dki_design_matrix
#from scipy.special import jn
#from dipy.data import get_fnames
from dipy.core.gradients import gradient_table
#import scipy.optimize as opt
#import pybobyqa
from dipy.data import get_sphere
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import SimpleITK as sitk
#from sklearn import linear_model
#from sklearn.linear_model import OrthogonalMatchingPursuit
#import spams
#import dipy.core.sphere as dipysphere
from tqdm import tqdm
import crl_aux
#import crl_dti
#import crl_dci
#from scipy.stats import f
#from importlib import reload
import h5py
#import dipy.reconst.sfm as sfm
#import dipy.data as dpd
#from dipy.viz import window, actor
#import dipy.direction.peaks as dpp
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
#from dipy.data import default_sphere
#from dipy.direction import peaks_from_model
from dipy.reconst.csdeconv import auto_response
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, color_fa
#from dipy.denoise.nlmeans import nlmeans
#from dipy.denoise.noise_estimate import estimate_sigma
#from dipy.denoise.pca_noise_estimate import pca_noise_estimate
#from dipy.denoise.localpca import localpca
#from dipy.denoise.localpca import mppca
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
#from dipy.viz import has_fury
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
#from dipy.viz import colormap
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
import nibabel as nib
from dipy.direction import DeterministicMaximumDirectionGetter
#from dipy.data import small_sphere
from dipy.direction import ProbabilisticDirectionGetter
#from dipy.direction import ClosestPeakDirectionGetter
from dipy.io.streamline import load_trk
#import dipy.tracking.life as life
import dk_aux
import pandas as pd
from os import listdir
from os.path import isdir, join
#from dipy.data.fetcher import get_two_hcp842_bundles
#from dipy.data.fetcher import (fetch_target_tractogram_hcp,
#                               fetch_bundle_atlas_hcp842,
#                               get_bundle_atlas_hcp842,
#                               get_target_tractogram_hcp)
#import numpy as np
#from dipy.segment.bundles import RecoBundles
#from dipy.align.streamlinear import whole_brain_slr
#from fury import actor, window
#from dipy.io.stateful_tractogram import Space, StatefulTractogram
#from dipy.io.streamline import load_trk, save_trk
#from dipy.io.utils import create_tractogram_header
#from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
#from scipy.ndimage.morphology import binary_dilation
from dipy.tracking.utils import length
#from dipy.tracking.metrics import downsample
#from dipy.tracking.distances import approx_polygon_track
#import dipy.core.optimize as dipy_opt
from dipy.tracking.streamline import cluster_confidence
#from dipy.direction.peaks import peak_directions
#import matplotlib.patches as patches
import tensorflow as tf
import dk_model
import time
from dipy.direction import ClosestPeakDirectionGetter
from dipy.reconst.forecast import ForecastModel
from dipy.reconst.shore import ShoreModel
import dipy.reconst.sfm as sfm
from dipy.reconst.dsi import DiffusionSpectrumModel
from dipy.reconst.gqi import GeneralizedQSamplingModel



run_DL_tractography= True
run_dipy_tractography= True
run_dipy_additional= False
run_dipy_tractography_full_gtab= False
save_fodf= False
save_data_thumbs= False
remove_short_streams= True
min_stream_length= 20
keep_confident_streams= False
confidence_percentile= 20


#   only test if the selected subject's age is in this range
min_age= 38
max_age= 44



assert( len(sys.argv)==3)

gpu_ind  = int(sys.argv[1])
subj_ind = int(sys.argv[2])

assert( gpu_ind>0 and gpu_ind<4)

print('Running on subject ' + str(subj_ind) + ' on GPU ' + str(gpu_ind) + '.')



# set the directories for dHCP data and results
dhcp_dir= ' .... /dHCP/data/'
res_dir=  '...  /dHCP/results/'




# read subject info

subj_info= pd.read_csv( dhcp_dir + 'participants.tsv', delimiter= '\t')

subj = subj_info['participant_id'][subj_ind]

anat_dir= dhcp_dir + 'dhcp_anat_pipeline/sub-' + subj
dmri_dir= dhcp_dir + 'dhcp_dmri_pipeline/sub-' + subj

dmri_sess = [d for d in listdir(dmri_dir) if isdir(join(dmri_dir, d))]

sess_tsv=  anat_dir + '/sub-' + subj + '_sessions.tsv'
sess_info= pd.read_csv( sess_tsv , delimiter= '\t')

j=k=0

if len(dmri_sess)==0 or not sess_info.loc[j, 'session_id']== int(dmri_sess[k].split('ses-')[1]):
    print('Subject ' + str(subj_ind) + ' does not have diffusion data. Exiting.')
    exit()

age_c= sess_info.loc[j, 'scan_age']

if age_c<min_age or age_c>max_age:
    print('Subject ' + str(subj_ind) + ' age, ' +  str(age_c) +  ', is outside the range.')
    exit()

print('Reading data for Subject ' + str(subj_ind) + '.')




# reads the data

subject= 'sub-' + subj
session= 'ses-'+ str(sess_info.loc[j, 'session_id'])

dwi_dir = dmri_dir + '/' + session + '/' + 'dwi/'
ant_dir = anat_dir + '/' + session + '/' + 'anat/'
dav_dir = res_dir  + 'Subject_' + str(subj_ind) + '_' + subject + '/'  + session + '/' # '_baseline/'
os.makedirs(dav_dir, exist_ok=True)

file_name= subject + '_' + session + '_desc-preproc_dwi.nii.gz'
d_img= sitk.ReadImage( dwi_dir + file_name )
d_img= sitk.GetArrayFromImage(d_img)
d_img= np.transpose( d_img, [3,2,1,0] )

file_name= subject + '_' + session + '_desc-preproc_dwi'
b_vals= np.loadtxt( dwi_dir + file_name + '.bval' )
b_vecs= np.loadtxt( dwi_dir + file_name + '.bvec' )

file_name= subject + '_' + session + '_desc-preproc_space-dwi_brainmask.nii.gz'
brain_mask_img= sitk.ReadImage( dwi_dir + file_name )
sitk.WriteImage(brain_mask_img, dav_dir + 'brain_mask_img.mhd')
brain_mask= sitk.GetArrayFromImage(brain_mask_img)
brain_mask= np.transpose( brain_mask, [2,1,0] )

ref_dir= brain_mask_img.GetDirection()
ref_org= brain_mask_img.GetOrigin()
ref_spc= brain_mask_img.GetSpacing()

file_name= subject + '_' + session + '_desc-restore_T2w.nii.gz'
t2_img= sitk.ReadImage( ant_dir + file_name )
t2_img= dk_aux.resample_imtar_to_imref(t2_img, brain_mask_img, sitk.sitkBSpline, False)
sitk.WriteImage(t2_img, dav_dir + 't2_img.mhd')
t2_img= sitk.GetArrayFromImage(t2_img)
t2_img= np.transpose( t2_img, [2,1,0] )

file_name= subject + '_' + session + '_desc-drawem87_space-T2w_dseg.nii.gz'
pr_img= sitk.ReadImage( ant_dir + file_name )
pr_img= dk_aux.resample_imtar_to_imref(pr_img, brain_mask_img, sitk.sitkNearestNeighbor, False)
sitk.WriteImage(pr_img, dav_dir + 'pr_img.mhd')
pr_img= sitk.GetArrayFromImage(pr_img)
pr_img= np.transpose( pr_img, [2,1,0] )

file_name= subject + '_' + session + '_desc-drawem9_space-T2w_dseg.nii.gz'
ts_img= sitk.ReadImage( ant_dir + file_name )
ts_img= dk_aux.resample_imtar_to_imref(ts_img, brain_mask_img, sitk.sitkNearestNeighbor, False)
sitk.WriteImage(ts_img, dav_dir + 'ts_img.mhd')
ts_img= sitk.GetArrayFromImage(ts_img)
ts_img= np.transpose( ts_img, [2,1,0] )

gtab = gradient_table(b_vals, b_vecs)
sx, sy, sz, _= d_img.shape



#  create a skull mask for stripping

skull= crl_aux.skull_from_brain_mask(brain_mask, radius= 2.0)
skull_img= np.transpose(skull, [2,1,0])
skull_img= sitk.GetImageFromArray(skull_img)
skull_img.SetDirection(ref_dir)
skull_img.SetOrigin(ref_org)
skull_img.SetSpacing(ref_spc)
sitk.WriteImage(skull_img, dav_dir + 'skull.mhd')




# basic DTI model (used later as stopping criteria for tractography)

tenmodel = dti.TensorModel(gtab)

tenfit = tenmodel.fit(d_img, brain_mask)

FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0
FA[skull==1]= 0

FA_img= np.transpose(FA, [2,1,0])
FA_img= sitk.GetImageFromArray(FA_img)
FA_img.SetDirection(ref_dir)
FA_img.SetOrigin(ref_org)
FA_img.SetSpacing(ref_spc)
sitk.WriteImage(FA_img, dav_dir + 'FA.mhd')
sitk.WriteImage(FA_img, dav_dir + 'FA.nii.gz' )

#RGB = color_fa(FA, tenfit.evecs)
#
#GFA_img= np.transpose(RGB, [2,1,0,3])
#GFA_img= sitk.GetImageFromArray(GFA_img)
#GFA_img.SetDirection(ref_dir)
#GFA_img.SetOrigin(ref_org)
#GFA_img.SetSpacing(ref_spc)
#sitk.WriteImage(GFA_img, dav_dir + 'ColorFA.mhd')

FA_img_nii= nib.load( dav_dir + 'FA.nii.gz' )
FA_img_nii.header['srow_x']= FA_img_nii.affine[0,:]
FA_img_nii.header['srow_y']= FA_img_nii.affine[1,:]
FA_img_nii.header['srow_z']= FA_img_nii.affine[2,:]

affine= FA_img_nii.affine

ten_img= tenfit.lower_triangular()
ten_img= ten_img[:,:,:,(0,1,3,2,4,5)]
array_img = nib.Nifti1Image(ten_img, affine)
nib.save(array_img, dav_dir + 'TN_img.nii.gz' )





# estimating fiber response from corpus callosum

cc_label= 48
cc_pixels= np.where(pr_img==cc_label)
cc_x_min, cc_x_max, cc_y_min, cc_y_max, cc_z_min, cc_z_max= \
        cc_pixels[0].min(), cc_pixels[0].max(), cc_pixels[1].min(), cc_pixels[1].max(), cc_pixels[2].min(), cc_pixels[2].max()

slc= (cc_z_min+cc_z_max)//2

response, ratio = auto_response(gtab, d_img[cc_x_min:cc_x_max, cc_y_min:cc_y_max, cc_z_min:cc_z_max,:], roi_radius=300, fa_thr=0.7)





#  choose the b=1000 shell

ind_0_1000= np.logical_or(b_vals==0, b_vals==1000)
d_img_0_1000=  d_img[:,:,:,ind_0_1000].copy()
b_vecs_0_1000= b_vecs[:,ind_0_1000]
b_vals_0_1000= b_vals[ind_0_1000]
gtab_0_1000 = gradient_table(b_vals_0_1000, b_vecs_0_1000)

response_0_1000, ratio_0_1000 = auto_response(gtab_0_1000, d_img_0_1000[cc_x_min:cc_x_max, cc_y_min:cc_y_max, cc_z_min:cc_z_max,:], roi_radius=300, fa_thr=0.7)





# the following is for measurement sub-sampling used in some experiments
''''
np.random.seed(0)

temp= np.where(b_vals==1000)[0]
np.random.shuffle(temp)
ind_60_1000= temp[:60]

temp= np.where(b_vals==0)[0]

ind_60= np.concatenate((temp,ind_60_1000))

d_img_60=  d_img[:,:,:,ind_60].copy()
b_vecs_60= b_vecs[:,ind_60]
b_vals_60= b_vals[ind_60]
gtab_60 = gradient_table(b_vals_60, b_vecs_60)

response_60, ratio_60 = auto_response(gtab_60, d_img_60[cc_x_min:cc_x_max, cc_y_min:cc_y_max, cc_z_min:cc_z_max,:], roi_radius=300, fa_thr=0.7)


np.random.seed(0)

temp= np.where(b_vals==1000)[0]
np.random.shuffle(temp)
ind_40_1000= temp[:40]

temp= np.where(b_vals==0)[0]

ind_40= np.concatenate((temp,ind_40_1000))

d_img_40=  d_img[:,:,:,ind_40].copy()
b_vecs_40= b_vecs[:,ind_40]
b_vals_40= b_vals[ind_40]
gtab_40 = gradient_table(b_vals_40, b_vecs_40)

response_40, ratio_40 = auto_response(gtab_40, d_img_40[cc_x_min:cc_x_max, cc_y_min:cc_y_max, cc_z_min:cc_z_max,:], roi_radius=300, fa_thr=0.7)



b_vecs_test_60= b_vecs[:,ind_60_1000]
b1_img_60= d_img[:,:,:,ind_60_1000]
b0_img_60= d_img[:,:,:,b_vals==0]
b0_img_60= np.mean(b0_img_60, axis=-1)

mask_60= b0_img_60>0



b_vecs_test_40= b_vecs[:,ind_40_1000]
b1_img_40= d_img[:,:,:,ind_40_1000]
b0_img_40= d_img[:,:,:,b_vals==0]
b0_img_40= np.mean(b0_img_40, axis=-1)

mask_40= b0_img_40>0

'''









##########   Load and apply our trained model

print('Initializng and loading MLP')

M= 3

sphere_sig = get_sphere('repulsion200')
v_sig = sphere_sig.vertices
n_sig = v_sig.shape[0]
sphere_fod = get_sphere('repulsion724')
v_fod = sphere_fod.vertices
n_fod = v_fod.shape[0]

#n_feat_vec= np.array([n_sig, 300, 200, 300, 200, 500, n_fod])
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_hardi_v2_3.ckpt'
n_feat_vec= np.array([n_sig, 300, 300, 300, 400, 500, 600, n_fod])
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_v2_M3_power35_normal.ckpt'
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_v2_M3_power25_corr_lrate_1n2.ckpt'
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_v2_M3_power25_corr_lrate_1n2_3fibers.ckpt'
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_v2_M3_power25_normal_less3.ckpt'


# baseline
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_snr20_M3_power25_corr_lrate_1n2_3fibers.ckpt'
## fodf_50
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_snr20_M3_power50_corr_lrate_1n2_3fibers.ckpt'
## snr 30
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_snr30_M3_power25_corr_lrate_1n2_3fibers.ckpt'
## M 7
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_snr20_M7_power25_corr_lrate_1n2_3fibers.ckpt'
## fodf 51
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_snr20_M3_power51_corr_lrate_1n2_3fibers.ckpt'
## power 8
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_snr20_M3_power8_corr_lrate_1n2_3fibers.ckpt'
## snr 15
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_snr15_M3_power25_corr_lrate_1n2_3fibers_Dpar1923_ang45.ckpt'
## diff diff
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_snr20_M3_power25_corr_lrate_1n2_3fibers_Dpar1923_ang45.ckpt'
## M 1
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_snr20_M1_power25_corr_lrate_1n2_3fibers.ckpt'
## M 10
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_snr20_M10_power25_corr_lrate_1n2_3fibers.ckpt'
## M 15
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_snr20_M15_power25_corr_lrate_1n2_3fibers_Dpar1923_ang45.ckpt'
## dipy_fa_M_3
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_csd_mask_wm_M_3.ckpt'
## dipy_fa_M_3
#temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_csd_mask_fa_M_3_power_2.ckpt'
# dipy_fa_M_3_subject_0_136
temp_path = '/media/nerossd2/ML-DCI/MLP-model/model_saved_dhcp_csd_mask_fa_M_3_subject_0_136.ckpt'





b_vecs_test= b_vecs[:,b_vals==1000]
b1_img= d_img[:,:,:,b_vals==1000]
b0_img= d_img[:,:,:,b_vals==0]
b0_img= np.mean(b0_img, axis=-1)

mask= b0_img>0









###########################################################

# seeds

#seed_mask= FA>0.15
seed_mask= ts_img==3
#seed_mask= seed_mask *  (1-skull)
seeds = utils.seeds_from_mask(seed_mask, affine, density=1)

# Generalized FA to be used as stopping criterion
'''
csa_model = CsaOdfModel(gtab, sh_order=6)
gfa = csa_model.fit(d_img, mask=mask).gfa

gfa_img= np.transpose(gfa, [2,1,0])
gfa_img= sitk.GetImageFromArray(gfa_img)
gfa_img.SetDirection(ref_dir)
gfa_img.SetOrigin(ref_org)
gfa_img.SetSpacing(ref_spc)
sitk.WriteImage(gfa_img, dav_dir + 'gfa.mhd')'''

#stopping_criterion = ThresholdStoppingCriterion(gfa, .20)
stopping_criterion = ThresholdStoppingCriterion(FA, .15)

###########################################################




if run_DL_tractography:
    
    # Run the proposed method
    
    X = tf.placeholder("float32", [None, n_sig ])
    Y = tf.placeholder("float32", [None, n_fod ])
    
    p_keep_hidden = tf.placeholder("float")
    
    Y_p_un = dk_model.davood_reg_net(X, n_feat_vec, p_keep_hidden, bias_init=0.001)
    Y_s= tf.reduce_sum(Y_p_un, axis=1)
    Y_p= Y_p_un/ tf.reshape(Y_s, [-1,1])
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)
    saver = tf.train.Saver(max_to_keep=50)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, temp_path)
    
    ######
    
    print('Estimating FODF with DL')
    t1= time.time()
    
    fodf_ml_v2= np.zeros( (sx, sy, sz, n_fod) )
    
    for ix in tqdm(range(sx), ascii=True):
        for iy in range(sy):
            for iz in range(sz):
                
                if mask[ix, iy, iz]:
                    
                    s= b1_img[ix, iy, iz,:]/ b0_img[ix, iy, iz]
                    
                    sig_interp= crl_aux.interpolate_s_2_sphere(s, b_vecs_test, v_sig, M= M, full_circ= False)
                    fodf_pred= sess.run(Y_p, feed_dict={X: sig_interp[np.newaxis,:], p_keep_hidden: 1.0}).astype(np.float64)
                    
                    fodf_ml_v2[ix,iy,iz,:]= fodf_pred
    
    
    '''
    fodf_ml_v2_60= np.zeros( (sx, sy, sz, n_fod) )
    
    for ix in tqdm(range(sx), ascii=True):
        for iy in range(sy):
            for iz in range(sz):
                
                if mask_60[ix, iy, iz]:
                    
                    s= b1_img_60[ix, iy, iz,:]/ b0_img_60[ix, iy, iz]
                    
                    sig_interp= crl_aux.interpolate_s_2_sphere(s, b_vecs_test_60, v_sig, M= M, full_circ= False)
                    fodf_pred= sess.run(Y_p, feed_dict={X: sig_interp[np.newaxis,:], p_keep_hidden: 1.0}).astype(np.float64)
                    
                    fodf_ml_v2_60[ix,iy,iz,:]= fodf_pred
    
    fodf_ml_v2_40= np.zeros( (sx, sy, sz, n_fod) )
    
    for ix in tqdm(range(sx), ascii=True):
        for iy in range(sy):
            for iz in range(sz):
                
                if mask_40[ix, iy, iz]:
                    
                    s= b1_img_40[ix, iy, iz,:]/ b0_img_40[ix, iy, iz]
                    
                    sig_interp= crl_aux.interpolate_s_2_sphere(s, b_vecs_test_40, v_sig, M= M, full_circ= False)
                    fodf_pred= sess.run(Y_p, feed_dict={X: sig_interp[np.newaxis,:], p_keep_hidden: 1.0}).astype(np.float64)
                    
                    fodf_ml_v2_40[ix,iy,iz,:]= fodf_pred             
    '''
    
    #sess.close()
    #tf.compat.v1.reset_default_graph()
    
    if save_fodf:
        h5f = h5py.File(dav_dir + 'fodf_ml_v2.h5','w')
        h5f['fodf_ml_v2']= fodf_ml_v2
        h5f.close()
       
    
    '''
    h5f = h5py.File( dav_dir + 'fodf_ml_v2.h5', 'r')
    fodf_ml_v2 = h5f['fodf_ml_v2'][:]
    h5f.close()
    '''
    
    
    pmf = fodf_ml_v2.copy()
    pmf[np.isnan(pmf)]= 1/n_fod
    #pmf= pmf**2
    
    fodf_pred_sum= np.sum(pmf, axis= -1)
    for i in range(pmf.shape[0]):
        for j in range(pmf.shape[1]):
            for k in range(pmf.shape[2]):
                if fodf_pred_sum[i,j,k]>0:
                    pmf[i,j,k,:]/= fodf_pred_sum[i,j,k]
    
    pmf_dl= pmf.copy()
    print('Finished estimating FODF with DL, time elapsed (minutes): ' + str(int( (time.time()-t1)/60 ) ) )
    



if run_dipy_tractography:
    
    print('Estimating FODF with CSD')
    t1= time.time()
    
    csd_model = ConstrainedSphericalDeconvModel(gtab_0_1000, response_0_1000, sh_order=6)
    csd_fit = csd_model.fit(d_img_0_1000, mask=mask)
    
    fodf_dipy = csd_fit.odf(sphere_fod)
    
    if save_fodf:
        h5f = h5py.File(dav_dir + 'fodf_dipy.h5','w')
        h5f['fodf_dipy']= fodf_dipy
        h5f.close()
    '''
    h5f = h5py.File( dav_dir + 'fodf_dipy.h5', 'r')
    fodf_dipy = h5f['fodf_dipy'][:]
    h5f.close()
    '''
    
    pmf = fodf_dipy.clip(min=0)
    #pmf= pmf**2
    
    fodf_pred_sum= np.sum(pmf, axis= -1)
    for i in range(pmf.shape[0]):
        for j in range(pmf.shape[1]):
            for k in range(pmf.shape[2]):
                if fodf_pred_sum[i,j,k]>0:
                    pmf[i,j,k,:]/= fodf_pred_sum[i,j,k]
    
    pmf_dipy= pmf.copy()
    print('Finished estimating FODF with DIPY, time elapsed (minutes): ' + str(int( (time.time()-t1)/60 ) ) )
    
    del fodf_dipy, pmf
    
    
    
    
    
    
    
    print('Estimating FODF with sfm')
    t1= time.time()
    
    sf_model = sfm.SparseFascicleModel(gtab_0_1000, sphere=sphere_fod,
                                       l1_ratio=0.5, alpha=0.001,
                                       response=response_0_1000[0])
    
    sf_fit = sf_model.fit(d_img_0_1000, mask=mask)  # mask
    
    odf_sfm = sf_fit.odf(sphere_fod)
        
    if save_fodf:
        h5f = h5py.File(dav_dir + 'odf_sfm.h5','w')
        h5f['odf_sfm']= odf_sfm
        h5f.close()
    '''
    h5f = h5py.File( dav_dir + 'odf_sfm.h5', 'r')
    odf_sfm = h5f['odf_sfm'][:]
    h5f.close()
    '''
    
    pmf = odf_sfm.clip(min=0)
    #pmf= pmf**2
    
    fodf_pred_sum= np.sum(pmf, axis= -1)
    for i in range(pmf.shape[0]):
        for j in range(pmf.shape[1]):
            for k in range(pmf.shape[2]):
                if fodf_pred_sum[i,j,k]>0:
                    pmf[i,j,k,:]/= fodf_pred_sum[i,j,k]
    
    pmf_sfm= pmf.copy()
    print('Finished estimating FODF with sfm, time elapsed (minutes): ' + str(int( (time.time()-t1)/60 ) ) )
    
    del odf_sfm, pmf
    
    
    
    
    '''
    print('Estimating FODF with sfm')
    t1= time.time()
    
    sf_model = sfm.SparseFascicleModel(gtab_40, sphere=sphere_fod,
                                       l1_ratio=0.5, alpha=0.001,
                                       response=response_40[0])
    
    sf_fit = sf_model.fit(d_img_40, mask=mask_40)  # mask
    
    odf_sfm = sf_fit.odf(sphere_fod)
        
    if save_fodf:
        h5f = h5py.File(dav_dir + 'odf_sfm.h5','w')
        h5f['odf_sfm']= odf_sfm
        h5f.close()
   
    
    pmf = odf_sfm.clip(min=0)
    #pmf= pmf**2
    
    fodf_pred_sum= np.sum(pmf, axis= -1)
    for i in range(pmf.shape[0]):
        for j in range(pmf.shape[1]):
            for k in range(pmf.shape[2]):
                if fodf_pred_sum[i,j,k]>0:
                    pmf[i,j,k,:]/= fodf_pred_sum[i,j,k]
    
    pmf_sfm_40= pmf.copy()
    print('Finished estimating FODF with sfm, time elapsed (minutes): ' + str(int( (time.time()-t1)/60 ) ) )
    
    del odf_sfm, pmf
    '''
    
    
    
    




























if run_dipy_tractography:
    
    print('Running tractography; CSD')
    t1= time.time()
    
    det_dg = DeterministicMaximumDirectionGetter.from_pmf(pmf_dipy, max_angle=30.,
                                                          sphere=sphere_fod)
    
    streamline_generator = LocalTracking(det_dg, stopping_criterion, seeds,
                                         affine=affine, step_size=.5)
    
    streamlines = Streamlines(streamline_generator)
    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
    save_trk(sft, dav_dir+"tractogram_dipy_deterministic.trk",bbox_valid_check=False)
    
    print('Finished tractography with DIPY, time elapsed (minutes): ' + str(int( (time.time()-t1)/60 ) ) )
    
    '''print('Running tractography; CSD 60')
    t1= time.time()
    
    det_dg = DeterministicMaximumDirectionGetter.from_pmf(pmf_dipy_60, max_angle=30.,
                                                          sphere=sphere_fod)
    
    streamline_generator = LocalTracking(det_dg, stopping_criterion, seeds,
                                         affine=affine, step_size=.5)
    
    streamlines = Streamlines(streamline_generator)
    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
    save_trk(sft, dav_dir+"tractogram_dipy_60_deterministic.trk",bbox_valid_check=False)
    
    print('Finished tractography with DIPY, time elapsed (minutes): ' + str(int( (time.time()-t1)/60 ) ) )
    
    print('Running tractography; CSD 40')
    t1= time.time()
    
    det_dg = DeterministicMaximumDirectionGetter.from_pmf(pmf_dipy_40, max_angle=30.,
                                                          sphere=sphere_fod)
    
    streamline_generator = LocalTracking(det_dg, stopping_criterion, seeds,
                                         affine=affine, step_size=.5)
    
    streamlines = Streamlines(streamline_generator)
    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
    save_trk(sft, dav_dir+"tractogram_dipy_40_deterministic.trk",bbox_valid_check=False)
    
    print('Finished tractography with DIPY, time elapsed (minutes): ' + str(int( (time.time()-t1)/60 ) ) )'''
    

    
    '''
    print('Running tractography; DIPY')
    t1= time.time()
    
    pmf = fodf_dipy.clip(min=0)
    #pmf= pmf**2
    
    fodf_pred_sum= np.sum(pmf, axis= -1)
    for i in range(pmf.shape[0]):
        for j in range(pmf.shape[1]):
            for k in range(pmf.shape[2]):
                if fodf_pred_sum[i,j,k]>0:
                    pmf[i,j,k,:]/= fodf_pred_sum[i,j,k]
    
    pmf_dipy= pmf.copy()
    
    det_dg = DeterministicMaximumDirectionGetter.from_pmf(pmf_dipy, max_angle=30.,
                                                          sphere=sphere_fod)
    
    streamline_generator = LocalTracking(det_dg, stopping_criterion, seeds,
                                         affine=affine, step_size=.5)
    
    streamlines = Streamlines(streamline_generator)
    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
    save_trk(sft, dav_dir+"tractogram_dipy_deterministic_unpowered.trk",bbox_valid_check=False)
    
    print('Finished tractography with DIPY, time elapsed (minutes): ' + str(int( (time.time()-t1)/60 ) ) )
    
    
    
    print('Running tractography; DIPY')
    t1= time.time()
    
    pmf = fodf_dipy.clip(min=0)
    
    pmf_dipy= pmf.copy()
    
    det_dg = DeterministicMaximumDirectionGetter.from_pmf(pmf_dipy, max_angle=30.,
                                                          sphere=sphere_fod)
    
    streamline_generator = LocalTracking(det_dg, stopping_criterion, seeds,
                                         affine=affine, step_size=.5)
    
    streamlines = Streamlines(streamline_generator)
    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
    save_trk(sft, dav_dir+"tractogram_dipy_deterministic_unnormalized.trk",bbox_valid_check=False)
    
    print('Finished tractography with DIPY, time elapsed (minutes): ' + str(int( (time.time()-t1)/60 ) ) )
    '''



    
    

    
    
    print('Running tractography; DIPY')
    t1= time.time()
    
    det_dg = DeterministicMaximumDirectionGetter.from_pmf(pmf_sfm, max_angle=30.,
                                                          sphere=sphere_fod)
    
    streamline_generator = LocalTracking(det_dg, stopping_criterion, seeds,
                                         affine=affine, step_size=.5)
    
    streamlines = Streamlines(streamline_generator)
    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
    save_trk(sft, dav_dir+"tractogram_dipy_deterministic_sfm.trk",bbox_valid_check=False)
    
    print('Finished tractography with DIPY, time elapsed (minutes): ' + str(int( (time.time()-t1)/60 ) ) )
    
    
    '''print('Running tractography; SFM')
    t1= time.time()
    
    det_dg = DeterministicMaximumDirectionGetter.from_pmf(pmf_sfm_40, max_angle=30.,
                                                          sphere=sphere_fod)
    
    streamline_generator = LocalTracking(det_dg, stopping_criterion, seeds,
                                         affine=affine, step_size=.5)
    
    streamlines = Streamlines(streamline_generator)
    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
    save_trk(sft, dav_dir+"tractogram_dipy_deterministic_sfm_40.trk",bbox_valid_check=False)
    
    print('Finished tractography with SFM, time elapsed (minutes): ' + str(int( (time.time()-t1)/60 ) ) )'''
    
    






if remove_short_streams:
    
    if run_dipy_tractography:
        
        tract_address= dav_dir+"tractogram_dipy_deterministic.trk"
        
        all_sl = load_trk( tract_address , FA_img_nii)
        all_sl = all_sl.streamlines
        all_sl= Streamlines(all_sl)
        
        lengths = list(length(all_sl))
        
        print('Removing small streamlines, DIPY. Number of streamlines: ' + str(len(lengths)) )
        t1= time.time()
        
        long_sl = Streamlines()
        #n_long= 0
        c_long= 0
        for i, sl in enumerate(all_sl):
            if i % 10000==0:
                print(i)
            if lengths[i] > min_stream_length:
                long_sl.append(sl)
                #n_long+= sl.shape[0]
                c_long+= 1
        
        print('Finished removing small streamlines, DIPY, time elapsed (minutes): ' 
              + str(int( (time.time()-t1)/60 ) ) 
              + '. Percentage of streamlines kept: ' 
              +  str(int( 100*c_long/len(lengths) ) ) )
        
        sft = StatefulTractogram(long_sl, FA_img_nii, Space.RASMM)
        save_trk(sft, dav_dir+"tractogram_dipy_deterministic_long_"+ str(min_stream_length) +".trk")
        
        
        ####################################################
        
        tract_address= dav_dir+"tractogram_dipy_deterministic.trk"
        all_sl = load_trk( tract_address , FA_img_nii)
        all_sl = all_sl.streamlines
        all_sl= Streamlines(all_sl)
        lengths = np.array(list(length(all_sl)))
        L_dp= np.sort(lengths)[::-1]
        L_dp_cumsum= np.cumsum(L_dp)
        temp=np.where(L_dp<min_stream_length)[0][0]
        L_connectome_dp= L_dp_cumsum[temp]
        
        
        '''tract_address= dav_dir+"tractogram_dipy_60_deterministic.trk"
        all_sl = load_trk( tract_address , FA_img_nii)
        all_sl = all_sl.streamlines
        all_sl= Streamlines(all_sl)
        lengths = np.array(list(length(all_sl)))
        L_dl= np.sort(lengths)[::-1]
        L_dl_cumsum= np.cumsum(L_dl)
        
        if L_dl_cumsum[-1]>L_connectome_dp:
            temp= np.where(L_dl_cumsum>L_connectome_dp)[0][0]
            min_stream_length_dl= L_dl[temp]
        else:
            min_stream_length_dl= 0
        
        if min_stream_length_dl>0:
            
            print('Removing small streamlines, CSD. Number of streamlines: ' + str(len(lengths)) )
            t1= time.time()
            
            long_sl = Streamlines()
            #n_long= 0
            c_long= 0
            for i, sl in enumerate(all_sl):
                if i % 10000==0:
                    print(i)
                if lengths[i] >= min_stream_length_dl:
                    long_sl.append(sl)
                    #n_long+= sl.shape[0]
                    c_long+= 1
            
            print('Finished removing small streamlines, DL, time elapsed (minutes): ' 
                  + str(int( (time.time()-t1)/60 ) ) 
                  + '. Percentage of streamlines kept: ' 
                  +  str(int( 100.0*c_long/len(lengths) ) ) )
            
            sft = StatefulTractogram(long_sl, FA_img_nii, Space.RASMM)
            save_trk(sft, dav_dir+"tractogram_dipy_60_deterministic_long_"+ 
                     str(int(round(min_stream_length_dl))) +
                     '_' + str(int( 100.0*c_long/len(lengths) ) ) + ".trk")
        
        fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
        plt.subplot(1, 1, 1)
        plt.plot(L_dp, 'b')
        plt.plot(L_dl, 'r')
        fig.savefig(dav_dir + 'lengths_csd_60.png')
        plt.close(fig)'''
        
       
        
        
        
        
        
        
        
         
        tract_address= dav_dir+"tractogram_dipy_deterministic_sfm.trk"
        
        all_sl = load_trk( tract_address , FA_img_nii)
        all_sl = all_sl.streamlines
        all_sl= Streamlines(all_sl)
        
        lengths = list(length(all_sl))
        
        print('Removing small streamlines, DIPY. Number of streamlines: ' + str(len(lengths)) )
        t1= time.time()
        
        long_sl = Streamlines()
        #n_long= 0
        c_long= 0
        for i, sl in enumerate(all_sl):
            if i % 10000==0:
                print(i)
            if lengths[i] > min_stream_length:
                long_sl.append(sl)
                #n_long+= sl.shape[0]
                c_long+= 1
        
        print('Finished removing small streamlines, DIPY, time elapsed (minutes): ' 
              + str(int( (time.time()-t1)/60 ) ) 
              + '. Percentage of streamlines kept: ' 
              +  str(int( 100*c_long/len(lengths) ) ) )
        
        sft = StatefulTractogram(long_sl, FA_img_nii, Space.RASMM)
        save_trk(sft, dav_dir+"tractogram_dipy_deterministic_sfm_long_"+ str(min_stream_length) +".trk")
        
        
        tract_address= dav_dir+"tractogram_dipy_deterministic.trk"
        all_sl = load_trk( tract_address , FA_img_nii)
        all_sl = all_sl.streamlines
        all_sl= Streamlines(all_sl)
        lengths = np.array(list(length(all_sl)))
        L_dp= np.sort(lengths)[::-1]
        L_dp_cumsum= np.cumsum(L_dp)
        temp=np.where(L_dp<min_stream_length)[0][0]
        L_connectome_dp= L_dp_cumsum[temp]
            
            


            
        '''tract_address= dav_dir+"tractogram_dipy_deterministic_sfm_40.trk"
        all_sl = load_trk( tract_address , FA_img_nii)
        all_sl = all_sl.streamlines
        all_sl= Streamlines(all_sl)
        lengths = np.array(list(length(all_sl)))
        L_dl= np.sort(lengths)[::-1]
        L_dl_cumsum= np.cumsum(L_dl)
        
        if L_dl_cumsum[-1]>L_connectome_dp:
            temp= np.where(L_dl_cumsum>L_connectome_dp)[0][0]
            min_stream_length_dl= L_dl[temp]
        else:
            min_stream_length_dl= 0
        
        if min_stream_length_dl>0:
            
            print('Removing small streamlines, DL. Number of streamlines: ' + str(len(lengths)) )
            t1= time.time()
            
            long_sl = Streamlines()
            #n_long= 0
            c_long= 0
            for i, sl in enumerate(all_sl):
                if i % 10000==0:
                    print(i)
                if lengths[i] >= min_stream_length_dl:
                    long_sl.append(sl)
                    #n_long+= sl.shape[0]
                    c_long+= 1
            
            print('Finished removing small streamlines, DL, time elapsed (minutes): ' 
                  + str(int( (time.time()-t1)/60 ) ) 
                  + '. Percentage of streamlines kept: ' 
                  +  str(int( 100.0*c_long/len(lengths) ) ) )
            
            sft = StatefulTractogram(long_sl, FA_img_nii, Space.RASMM)
            save_trk(sft, dav_dir+"tractogram_dipy_deterministic_sfm_40_long_"+ 
                     str(int(round(min_stream_length_dl))) +
                     '_' + str(int( 100.0*c_long/len(lengths) ) ) + ".trk")
        
        fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
        plt.subplot(1, 1, 1)
        plt.plot(L_dp, 'b')
        plt.plot(L_dl, 'r')
        fig.savefig(dav_dir + 'lengths_sfm.png')
        plt.close(fig)'''
        
        
        
        



if run_DL_tractography:
    
    
    tract_address= dav_dir+"tractogram_dipy_deterministic.trk"
    all_sl = load_trk( tract_address , FA_img_nii)
    all_sl = all_sl.streamlines
    all_sl= Streamlines(all_sl)
    lengths = np.array(list(length(all_sl)))
    L_dp= np.sort(lengths)[::-1]
    L_dp_cumsum= np.cumsum(L_dp)
    temp=np.where(L_dp<min_stream_length)[0][0]
    L_connectome_dp= L_dp_cumsum[temp]
    
    
    
    pmf = fodf_ml_v2.copy()
    pmf[np.isnan(pmf)]= 1/n_fod
    #pmf= pmf**8
    
    fodf_pred_sum= np.sum(pmf, axis= -1)
    for i in range(pmf.shape[0]):
        for j in range(pmf.shape[1]):
            for k in range(pmf.shape[2]):
                if fodf_pred_sum[i,j,k]>0:
                    pmf[i,j,k,:]/= fodf_pred_sum[i,j,k]
    
    pmf_dl= pmf.copy()
    
    print('\n Running tractography; DL')
    t1= time.time()
    
    det_dg = DeterministicMaximumDirectionGetter.from_pmf(pmf_dl, max_angle=30.,
                                                          sphere=sphere_fod)
    
    streamline_generator = LocalTracking(det_dg, stopping_criterion, seeds,
                                         affine=affine, step_size=.5)
    
    streamlines = Streamlines(streamline_generator)
    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
    save_trk(sft, dav_dir+"det.trk",bbox_valid_check=False)
    
    print('Finished tractography with DL, time elapsed (minutes): ' + str(int( (time.time()-t1)/60 ) ) )
    
    tract_address= dav_dir+"det.trk"
    all_sl = load_trk( tract_address , FA_img_nii)
    all_sl = all_sl.streamlines
    all_sl= Streamlines(all_sl)
    lengths = np.array(list(length(all_sl)))
    L_dl= np.sort(lengths)[::-1]
    L_dl_cumsum= np.cumsum(L_dl)
    
    if L_dl_cumsum[-1]>L_connectome_dp:
        temp= np.where(L_dl_cumsum>L_connectome_dp)[0][0]
        min_stream_length_dl= L_dl[temp]
    else:
        min_stream_length_dl= 0
    
    if min_stream_length_dl>0:
        
        print('Removing small streamlines, DL. Number of streamlines: ' + str(len(lengths)) )
        t1= time.time()
        
        long_sl = Streamlines()
        #n_long= 0
        c_long= 0
        for i, sl in enumerate(all_sl):
            if i % 10000==0:
                print(i)
            if lengths[i] >= min_stream_length_dl:
                long_sl.append(sl)
                #n_long+= sl.shape[0]
                c_long+= 1
        
        print('Finished removing small streamlines, DL, time elapsed (minutes): ' 
              + str(int( (time.time()-t1)/60 ) ) 
              + '. Percentage of streamlines kept: ' 
              +  str(int( 100.0*c_long/len(lengths) ) ) )
        
        sft = StatefulTractogram(long_sl, FA_img_nii, Space.RASMM)
        save_trk(sft, dav_dir+"det_long_"+ str(int(round(min_stream_length_dl))) +
                 '_' + str(int( 100.0*c_long/len(lengths) ) ) + ".trk")
    
    fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
    plt.subplot(1, 1, 1)
    plt.plot(L_dp, 'b')
    plt.plot(L_dl, 'r')
    fig.savefig(dav_dir + 'lengths.png')
    plt.close(fig)
    
    
    
    '''
    pmf = fodf_ml_v2_60.copy()
    pmf[np.isnan(pmf)]= 1/n_fod
    #pmf= pmf**8
    
    fodf_pred_sum= np.sum(pmf, axis= -1)
    for i in range(pmf.shape[0]):
        for j in range(pmf.shape[1]):
            for k in range(pmf.shape[2]):
                if fodf_pred_sum[i,j,k]>0:
                    pmf[i,j,k,:]/= fodf_pred_sum[i,j,k]
    
    pmf_dl= pmf.copy()
    
    print('\n Running tractography; DL')
    t1= time.time()
    
    det_dg = DeterministicMaximumDirectionGetter.from_pmf(pmf_dl, max_angle=30.,
                                                          sphere=sphere_fod)
    
    streamline_generator = LocalTracking(det_dg, stopping_criterion, seeds,
                                         affine=affine, step_size=.5)
    
    streamlines = Streamlines(streamline_generator)
    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
    save_trk(sft, dav_dir+"det.trk",bbox_valid_check=False)
    
    print('Finished tractography with DL, time elapsed (minutes): ' + str(int( (time.time()-t1)/60 ) ) )
    
    tract_address= dav_dir+"det.trk"
    all_sl = load_trk( tract_address , FA_img_nii)
    all_sl = all_sl.streamlines
    all_sl= Streamlines(all_sl)
    lengths = np.array(list(length(all_sl)))
    L_dl= np.sort(lengths)[::-1]
    L_dl_cumsum= np.cumsum(L_dl)
    
    if L_dl_cumsum[-1]>L_connectome_dp:
        temp= np.where(L_dl_cumsum>L_connectome_dp)[0][0]
        min_stream_length_dl= L_dl[temp]
    else:
        min_stream_length_dl= 0
    
    if min_stream_length_dl>0:
        
        print('Removing small streamlines, DL. Number of streamlines: ' + str(len(lengths)) )
        t1= time.time()
        
        long_sl = Streamlines()
        #n_long= 0
        c_long= 0
        for i, sl in enumerate(all_sl):
            if i % 10000==0:
                print(i)
            if lengths[i] >= min_stream_length_dl:
                long_sl.append(sl)
                #n_long+= sl.shape[0]
                c_long+= 1
        
        print('Finished removing small streamlines, DL, time elapsed (minutes): ' 
              + str(int( (time.time()-t1)/60 ) ) 
              + '. Percentage of streamlines kept: ' 
              +  str(int( 100.0*c_long/len(lengths) ) ) )
        
        sft = StatefulTractogram(long_sl, FA_img_nii, Space.RASMM)
        save_trk(sft, dav_dir+"det_60_long_"+ str(int(round(min_stream_length_dl))) +
                 '_' + str(int( 100.0*c_long/len(lengths) ) ) + ".trk")
    
    fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
    plt.subplot(1, 1, 1)
    plt.plot(L_dp, 'b')
    plt.plot(L_dl, 'r')
    fig.savefig(dav_dir + 'lengths_60.png')
    plt.close(fig)
    '''
    
    
    
    '''
    pmf = fodf_ml_v2_40.copy()
    pmf[np.isnan(pmf)]= 1/n_fod
    #pmf= pmf**8
    
    fodf_pred_sum= np.sum(pmf, axis= -1)
    for i in range(pmf.shape[0]):
        for j in range(pmf.shape[1]):
            for k in range(pmf.shape[2]):
                if fodf_pred_sum[i,j,k]>0:
                    pmf[i,j,k,:]/= fodf_pred_sum[i,j,k]
    
    pmf_dl= pmf.copy()
    
    print('\n Running tractography; DL')
    t1= time.time()
    
    det_dg = DeterministicMaximumDirectionGetter.from_pmf(pmf_dl, max_angle=30.,
                                                          sphere=sphere_fod)
    
    streamline_generator = LocalTracking(det_dg, stopping_criterion, seeds,
                                         affine=affine, step_size=.5)
    
    streamlines = Streamlines(streamline_generator)
    sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
    save_trk(sft, dav_dir+"det.trk",bbox_valid_check=False)
    
    print('Finished tractography with DL, time elapsed (minutes): ' + str(int( (time.time()-t1)/60 ) ) )
    
    tract_address= dav_dir+"det.trk"
    all_sl = load_trk( tract_address , FA_img_nii)
    all_sl = all_sl.streamlines
    all_sl= Streamlines(all_sl)
    lengths = np.array(list(length(all_sl)))
    L_dl= np.sort(lengths)[::-1]
    L_dl_cumsum= np.cumsum(L_dl)
    
    if L_dl_cumsum[-1]>L_connectome_dp:
        temp= np.where(L_dl_cumsum>L_connectome_dp)[0][0]
        min_stream_length_dl= L_dl[temp]
    else:
        min_stream_length_dl= 0
    
    if min_stream_length_dl>0:
        
        print('Removing small streamlines, DL. Number of streamlines: ' + str(len(lengths)) )
        t1= time.time()
        
        long_sl = Streamlines()
        #n_long= 0
        c_long= 0
        for i, sl in enumerate(all_sl):
            if i % 10000==0:
                print(i)
            if lengths[i] >= min_stream_length_dl:
                long_sl.append(sl)
                #n_long+= sl.shape[0]
                c_long+= 1
        
        print('Finished removing small streamlines, DL, time elapsed (minutes): ' 
              + str(int( (time.time()-t1)/60 ) ) 
              + '. Percentage of streamlines kept: ' 
              +  str(int( 100.0*c_long/len(lengths) ) ) )
        
        sft = StatefulTractogram(long_sl, FA_img_nii, Space.RASMM)
        save_trk(sft, dav_dir+"det_40_long_"+ str(int(round(min_stream_length_dl))) +
                 '_' + str(int( 100.0*c_long/len(lengths) ) ) + ".trk")
    
    fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
    plt.subplot(1, 1, 1)
    plt.plot(L_dp, 'b')
    plt.plot(L_dl, 'r')
    fig.savefig(dav_dir + 'lengths_40.png')
    plt.close(fig)
    '''



















