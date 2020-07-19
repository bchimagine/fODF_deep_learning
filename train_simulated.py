#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:24:47 2020

@author: davood
"""



import numpy as np
import crl_dci
import crl_aux
#import matplotlib.pyplot as plt
from dipy.data import get_sphere
#from sklearn import svm
import tensorflow as tf
import dk_model
import os
from tqdm import tqdm
import h5py
#from numpy import matlib as mb
from os import listdir
from os.path import isfile, join
import dipy.core.sphere as dipysphere


base_dir= '...'
hardi_dir= '...'

gtab_scheme=  'dHCP'

if gtab_scheme=='dHCP':
    
    b_vals= np.loadtxt( base_dir + 'sub-CC00060XX03_ses-12501_desc-preproc_dwi.bval' )
    b_vecs= np.loadtxt( base_dir + 'sub-CC00060XX03_ses-12501_desc-preproc_dwi.bvec' ).T
    b_vecs= b_vecs[b_vals==1000]
    b_vals= b_vals[b_vals==1000]
    
    b_vals_test= np.loadtxt( base_dir + 'sub-CC00124XX09_ses-42302_desc-preproc_dwi.bval' )
    b_vecs_test= np.loadtxt( base_dir + 'sub-CC00124XX09_ses-42302_desc-preproc_dwi.bvec' ).T
    b_vecs_test= b_vecs_test[b_vals_test==1000]
    b_vals_test= b_vals_test[b_vals_test==1000]
    
    train_bvecs = [f for f in listdir(base_dir+'train_gtabs/') if isfile(join(base_dir+'train_gtabs/', f)) and 'vec' in f]
    train_bvals = [f for f in listdir(base_dir+'train_gtabs/') if isfile(join(base_dir+'train_gtabs/', f)) and 'val' in f]
    train_bvecs.sort()
    train_bvals.sort()
    n_train_gtab= len(train_bvals)
    
elif gtab_scheme=='HARDI2013':
    
    b_vals= np.loadtxt( hardi_dir+'hardi-scheme.bval' )
    b_vecs= np.loadtxt( hardi_dir+'hardi-scheme.bvec' ).T
    b_vecs= b_vecs[b_vals>10,:]
    b_vals= b_vals[b_vals>10]
    
    b_vals_test= b_vals
    b_vecs_test= b_vecs
    
    train_bvecs= [b_vecs]
    train_bvals= [b_vals]
    n_train_gtab= len(train_bvals)
    
elif gtab_scheme=='irontract':
    
    b_vals= np.loadtxt( '/media/davood/bd794557-342d-4d50-8b7f-f6c209248f89/davood/DWI_data/irontract/bvalues.hcpl.txt' )
    b_vecs= np.loadtxt( '/media/davood/bd794557-342d-4d50-8b7f-f6c209248f89/davood/DWI_data/irontract/gradients.hcpl.txt' )
    b_norms= np.zeros(b_vecs.shape[0])
    
    b_val_ind= np.where( b_vals <10000 )[0][1:]
    
    for i in range(b_vecs.shape[0]):
        temp= np.linalg.norm(b_vecs[i,:])
        b_norms[i]= temp
        if temp>0:
            b_vecs[i,:]/= temp
    
    b_vals= b_vals[b_val_ind]
    b_vecs= b_vecs[b_val_ind,:]
    
    b_vecs= b_vecs[b_vals>10,:]
    b_vals= b_vals[b_vals>10]
    
    b_vals_test= b_vals
    b_vecs_test= b_vecs
    
    train_bvecs= [b_vecs]
    train_bvals= [b_vals]
    n_train_gtab= len(train_bvals)





####    Generate training and test data

min_fiber_separation= 35

D_par_range= [0.00185, 0.00235]
D_per_range= [0.00040, 0.00060]

if gtab_scheme=='irontract':
    D_par_range= [0.0015, 0.0020]
    D_per_range= [0.00035, 0.00045]


###########

snr= 20
fodf_power= 25
M= 5


'''sphere_sig = get_sphere('repulsion200')
v_sig = sphere_sig.vertices
n_sig = v_sig.shape[0]

#sphere_fod = get_sphere('symmetric362')
sphere_fod = get_sphere('repulsion724')
v_fod = sphere_fod.vertices
n_fod = v_fod.shape[0]'''

n_sig= 200
Xp, Yp, Zp= crl_aux.distribute_on_sphere_spiral(n_sig)
sphere_sig = dipysphere.Sphere(Xp, Yp, Zp)
v_sig, _ = sphere_sig.vertices, sphere_sig.faces

n_fod= 724
Xp, Yp, Zp= crl_aux.distribute_on_sphere_spiral(n_fod)
sphere_fod = dipysphere.Sphere(Xp, Yp, Zp)
v_fod, _ = sphere_fod.vertices, sphere_fod.faces

###########


n_fib= 1
CSF_range= [0.0, 0.40]

n_fib_sim= 3000000

X_train_1= np.zeros( (n_fib_sim,n_sig)  , np.float32 )
Y_train_1= np.zeros( (n_fib_sim,n_fod)  , np.float32 )
i_train= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    if gtab_scheme=='dHCP' and i_fib_sim%1000==0:
        i_gtab= np.random.randint(0,n_train_gtab)
        b_vals= np.loadtxt( base_dir+'train_gtabs/' + train_bvals[i_gtab] )
        b_vecs= np.loadtxt( base_dir+'train_gtabs/' + train_bvecs[i_gtab] ).T
        b_vecs= b_vecs[b_vals==1000]
        b_vals= b_vals[b_vals==1000]
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers= crl_dci.random_points_on_sphere(n_fib, min_separation=min_fiber_separation, n_try_angle= 100)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals, b_vecs, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    sig_interp= crl_aux.interpolate_s_2_sphere(s, b_vecs.T, v_sig, M= M, full_circ= False)
    fodf= crl_aux.fibers_2_fodf_weighted(true_fibers, v_fod.T, fodf_power)
    
    i_train+= 1
    Y_train_1[i_train,:]= fodf
    X_train_1[i_train,:]= sig_interp
    
n_fib_sim= 10000

X_test_1= np.zeros( (n_fib_sim,n_sig)  , np.float32)
Y_test_1= np.zeros( (n_fib_sim,n_fod)  , np.float32)
i_test= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers= crl_dci.random_points_on_sphere(n_fib, min_separation=min_fiber_separation, n_try_angle= 100)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals_test, b_vecs_test, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    sig_interp= crl_aux.interpolate_s_2_sphere(s, b_vecs_test.T, v_sig, M= M, full_circ= False)
    fodf= crl_aux.fibers_2_fodf_weighted(true_fibers, v_fod.T, fodf_power)
    
    i_test+= 1
    X_test_1[i_test,:]= sig_interp
    Y_test_1[i_test,:]= fodf
            
############

n_fib= 2
CSF_range= [0.0, 0.20]
Fr1_range= [0.30, 0.50]

n_fib_sim= 3000000

X_train_2= np.zeros( (n_fib_sim,n_sig)  , np.float32 )
Y_train_2= np.zeros( (n_fib_sim,n_fod)  , np.float32 )
i_train= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    if gtab_scheme=='dHCP' and i_fib_sim%1000==0:
        i_gtab= np.random.randint(0,n_train_gtab)
        b_vals= np.loadtxt( base_dir+'train_gtabs/' + train_bvals[i_gtab] )
        b_vecs= np.loadtxt( base_dir+'train_gtabs/' + train_bvecs[i_gtab] ).T
        b_vecs= b_vecs[b_vals==1000]
        b_vals= b_vals[b_vals==1000]
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers= crl_dci.random_points_on_sphere(n_fib, min_separation=min_fiber_separation, n_try_angle= 100)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= (1-Fr1-CSF)/np.linalg.norm(true_fibers[:,1])
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals, b_vecs, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    sig_interp= crl_aux.interpolate_s_2_sphere(s, b_vecs.T, v_sig, M= M, full_circ= False)
    fodf= crl_aux.fibers_2_fodf_weighted(true_fibers, v_fod.T, fodf_power)
    
    i_train+= 1
    Y_train_2[i_train,:]= fodf
    X_train_2[i_train,:]= sig_interp

n_fib_sim= 10000

X_test_2= np.zeros( (n_fib_sim,n_sig) , np.float32)
Y_test_2= np.zeros( (n_fib_sim,n_fod) , np.float32 )
i_test= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers= crl_dci.random_points_on_sphere(n_fib, min_separation=min_fiber_separation, n_try_angle= 100)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= (1-Fr1-CSF)/np.linalg.norm(true_fibers[:,1])
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals_test, b_vecs_test, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    sig_interp= crl_aux.interpolate_s_2_sphere(s, b_vecs_test.T, v_sig, M= M, full_circ= False)
    fodf= crl_aux.fibers_2_fodf_weighted(true_fibers, v_fod.T, fodf_power)
    
    i_test+= 1
    X_test_2[i_test,:]= sig_interp
    Y_test_2[i_test,:]= fodf

############

n_fib= 3
CSF_range= [0.0, 0.05]
Fr1_range= [0.2, 0.4]
Fr2_range= [0.2, 0.35]

n_fib_sim= 3000000

X_train_3= np.zeros( (n_fib_sim,n_sig)  , np.float32 )
Y_train_3= np.zeros( (n_fib_sim,n_fod)  , np.float32 )
i_train= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    if gtab_scheme=='dHCP' and i_fib_sim%1000==0:
        i_gtab= np.random.randint(0,n_train_gtab)
        b_vals= np.loadtxt( base_dir+'train_gtabs/' + train_bvals[i_gtab] )
        b_vecs= np.loadtxt( base_dir+'train_gtabs/' + train_bvecs[i_gtab] ).T
        b_vecs= b_vecs[b_vals==1000]
        b_vals= b_vals[b_vals==1000]
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers= crl_dci.random_points_on_sphere(n_fib, min_separation=min_fiber_separation, n_try_angle= 100)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    Fr2= np.random.rand() * (Fr2_range[1]-Fr2_range[0]) + Fr2_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= Fr2/np.linalg.norm(true_fibers[:,1])
    true_fibers[:,2]*= (1-Fr1-Fr2-CSF)/np.linalg.norm(true_fibers[:,2])
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals, b_vecs, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    sig_interp= crl_aux.interpolate_s_2_sphere(s, b_vecs.T, v_sig, M= M, full_circ= False)
    fodf= crl_aux.fibers_2_fodf_weighted(true_fibers, v_fod.T, fodf_power)
    
    i_train+= 1
    X_train_3[i_train,:]= sig_interp
    Y_train_3[i_train,:]= fodf

n_fib_sim= 10000

X_test_3= np.zeros( (n_fib_sim,n_sig)  , np.float32 )
Y_test_3= np.zeros( (n_fib_sim,n_fod)  , np.float32 )
i_test= -1

for i_fib_sim in tqdm(range(n_fib_sim), ascii=True):
    
    d_par= np.random.rand() * (D_par_range[1]-D_par_range[0]) + D_par_range[0]
    d_per= np.random.rand() * (D_per_range[1]-D_per_range[0]) + D_per_range[0]
    
    true_fibers= crl_dci.random_points_on_sphere(n_fib, min_separation=min_fiber_separation, n_try_angle= 100)
    
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    Fr2= np.random.rand() * (Fr2_range[1]-Fr2_range[0]) + Fr2_range[0]
    
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= Fr2/np.linalg.norm(true_fibers[:,1])
    true_fibers[:,2]*= (1-Fr1-Fr2-CSF)/np.linalg.norm(true_fibers[:,2])
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals_test, b_vecs_test, d_par, d_per, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    sig_interp= crl_aux.interpolate_s_2_sphere(s, b_vecs_test.T, v_sig, M= M, full_circ= False)
    fodf= crl_aux.fibers_2_fodf_weighted(true_fibers, v_fod.T, fodf_power)
    
    i_test+= 1
    X_test_3[i_test,:]= sig_interp
    Y_test_3[i_test,:]= fodf












#h5f = h5py.File(base_dir + 'data_dhcp_snr_20_v2_200_724_power25_M15_correct_1.h5','w')
#h5f['X_train_1']= X_train_1
#h5f['Y_train_1']= Y_train_1
#h5f['X_test_1']= X_test_1
#h5f['Y_test_1']= Y_test_1
#h5f['X_train_2']= X_train_2
#h5f['Y_train_2']= Y_train_2
#h5f['X_test_2']= X_test_2
#h5f['Y_test_2']= Y_test_2
#h5f['X_train_3']= X_train_3
#h5f['Y_train_3']= Y_train_3
#h5f['X_test_3']= X_test_3
#h5f['Y_test_3']= Y_test_3
#h5f.close()



#h5f = h5py.File(base_dir + 'data_dhcp_snr_15_v2_200_724_power25_M3_correct_1.h5', 'r')
#X_train_1 = h5f['X_train_1'][:]
#Y_train_1 = h5f['Y_train_1'][:]
#X_test_1 = h5f['X_test_1'][:]
#Y_test_1 = h5f['Y_test_1'][:]
#X_train_2 = h5f['X_train_2'][:]
#Y_train_2 = h5f['Y_train_2'][:]
#X_test_2 = h5f['X_test_2'][:]
#Y_test_2 = h5f['Y_test_2'][:]
#X_train_3 = h5f['X_train_3'][:]
#Y_train_3 = h5f['Y_train_3'][:]
#X_test_3 = h5f['X_test_3'][:]
#Y_test_3 = h5f['Y_test_3'][:]
#h5f.close()




#######################################################################

X_train= np.concatenate( (X_train_1, X_train_2, X_train_3), axis=0 )
#X_train= np.concatenate( (X_train_1, X_train_2), axis=0 )
del X_train_1, X_train_2, X_train_3
Y_train= np.concatenate( (Y_train_1, Y_train_2, Y_train_3), axis=0 )
#Y_train= np.concatenate( (Y_train_1, Y_train_2), axis=0 )
del Y_train_1, Y_train_2, Y_train_3

X_test= np.concatenate( (X_test_1, X_test_2, X_test_3), axis=0 )
#X_test= np.concatenate( (X_test_1, X_test_2), axis=0 )
del X_test_1, X_test_2, X_test_3
Y_test= np.concatenate( (Y_test_1, Y_test_2, Y_test_3), axis=0 )
#Y_test= np.concatenate( (Y_test_1, Y_test_2), axis=0 )
del Y_test_1, Y_test_2, Y_test_3











###############################################################################

###   Build and train MLP

gpu_ind= 2

L_Rate = 1.0e-2


#n_feat_vec= np.array([n_sig, 300, 200, 300, 200, 500, n_fod])
n_feat_vec= np.array([n_sig, 300, 300, 300, 400, 500, 600, n_fod])


X = tf.placeholder("float32", [None, n_feat_vec[0]])
Y = tf.placeholder("float32", [None, n_feat_vec[-1]])

learning_rate = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")


Y_p_un = dk_model.davood_reg_net(X, n_feat_vec, p_keep_hidden, bias_init=0.001)

Y_s= tf.reduce_sum(Y_p_un, axis=1)

Y_p= Y_p_un/ tf.reshape(Y_s, [-1,1])

cost= tf.reduce_mean( tf.pow(( Y - Y_p ), 2) )
#cost= tf.reduce_mean( tf.abs( Y - Y_p ) )

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)


saver = tf.train.Saver(max_to_keep=50)

sess = tf.Session()
sess.run(tf.global_variables_initializer())



i_global = 0
best_test = 0

i_eval = -1



batch_size = 1000
n_epochs = 10000


n_train= X_train.shape[0]
n_test = X_test.shape[0]

test_interval = n_train//batch_size * 5


for epoch_i in range(n_epochs):
    
    for i_train in range(n_train//batch_size):
        
        q= np.random.randint(0,n_train,(batch_size))
        batch_x = X_train[q, :].copy()
        batch_y = Y_train[q, :].copy()
        
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, learning_rate: L_Rate, p_keep_hidden: 1.0})
        
        i_global += 1
        
        if i_global % test_interval == 0:
            
            i_eval += 1
            
            print('\n', epoch_i, i_train, i_global)
            
            cost_v = np.zeros(n_train//batch_size)
            
            for i_v in tqdm(range(n_train//batch_size), ascii=True):
                
                batch_x = X_train[i_v * batch_size:(i_v + 1) * batch_size, :].copy()
                batch_y = Y_train[i_v * batch_size:(i_v + 1) * batch_size, :].copy()
                
                cost_v[i_v]= sess.run(cost, feed_dict={X: batch_x, Y: batch_y, p_keep_hidden: 1.0})
            
            print('Train cost  %.6f' % (cost_v.mean()*1000))
            
            cost_v = np.zeros(n_test//batch_size)
            
            for i_v in tqdm(range(n_test//batch_size), ascii=True):
                
                batch_x = X_test[i_v * batch_size:(i_v + 1) * batch_size, :].copy()
                batch_y = Y_test[i_v * batch_size:(i_v + 1) * batch_size, :].copy()
                
                cost_v[i_v]= sess.run(cost, feed_dict={X: batch_x, Y: batch_y, p_keep_hidden: 1.0})
            
            print('Test cost   %.6f' % (cost_v.mean()*1000))



temp_path = base_dir + 'MLP-model/model_saved_dhcp_snr20_M10_power25_corr_lrate_1n2_3fibers.ckpt'

#saver.save(sess, temp_path)

saver.restore(sess, temp_path)



##############################################################################











#   Preliminary test

n_fib= 1



if n_fib==1:
    CSF_range= [0.0, 0.50]
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    fodf= crl_aux.fibers_2_fodf_weighted(true_fibers, v_fod.T, fodf_power)
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    true_fibers= (1-CSF)/n_fib* true_fibers
elif n_fib==2:
    CSF_range= [0.0, 0.30]
    Fr1_range= [0.3, 0.5]
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    fodf= crl_aux.fibers_2_fodf_weighted(true_fibers, v_fod.T, fodf_power)
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= (1-Fr1-CSF)/np.linalg.norm(true_fibers[:,1])
elif n_fib==3:
    CSF_range= [0.0, 0.20]
    Fr1_range= [0.3, 0.4]
    Fr2_range= [0.2, 0.3]
    true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
    true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
    fodf= crl_aux.fibers_2_fodf_weighted(true_fibers, v_fod.T, fodf_power)
    CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
    Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
    Fr2= np.random.rand() * (Fr2_range[1]-Fr2_range[0]) + Fr2_range[0]
    true_fibers= (1-CSF)/n_fib* true_fibers
    true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
    true_fibers[:,1]*= Fr2/np.linalg.norm(true_fibers[:,1])
    true_fibers[:,2]*= (1-Fr1-Fr2-CSF)/np.linalg.norm(true_fibers[:,2])




s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals_test, b_vecs_test, 0.0021, 0.00050, 0.003)
s= crl_aux.add_rician_noise(s, snr=snr)

sig_interp= crl_aux.interpolate_s_2_sphere(s, b_vecs_test.T, v_sig, M= 1, full_circ= False)


fodf_pred= sess.run(Y_p, feed_dict={X: sig_interp[np.newaxis,:], p_keep_hidden: 1.0})

crl_aux.plot_odf_and_fibers(v_fod.T, fodf_pred, true_fibers)










##############################################################################


n_fib_max= 3

N_test= 1000*3
Pred_N= np.zeros(N_test, np.int8)
Pred_E= np.zeros((N_test,n_fib_max))

for i_test in range(N_test):
    
    n_fib= (i_test*3)//N_test + 1
    
    if n_fib==1:
        CSF_range= [0.0, 0.50]
        true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
        true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
        fodf= crl_aux.fibers_2_fodf_weighted(true_fibers, v_fod.T, fodf_power)
        CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
        true_fibers= (1-CSF)/n_fib* true_fibers
    elif n_fib==2:
        CSF_range= [0.0, 0.30]
        Fr1_range= [0.3, 0.5]
        true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
        true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
        fodf= crl_aux.fibers_2_fodf_weighted(true_fibers, v_fod.T, fodf_power)
        CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
        Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
        true_fibers= (1-CSF)/n_fib* true_fibers
        true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
        true_fibers[:,1]*= (1-Fr1-CSF)/np.linalg.norm(true_fibers[:,1])
    elif n_fib==3:
        CSF_range= [0.0, 0.20]
        Fr1_range= [0.3, 0.4]
        Fr2_range= [0.2, 0.3]
        true_fibers_polar, _, _= crl_dci.polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003, min_separation= min_fiber_separation, n_try_angle= 100)
        true_fibers= crl_dci.polar_fibers_from_solution(true_fibers_polar, n_fib)
        fodf= crl_aux.fibers_2_fodf_weighted(true_fibers, v_fod.T, fodf_power)
        CSF= np.random.rand() * (CSF_range[1]-CSF_range[0]) + CSF_range[0]
        Fr1= np.random.rand() * (Fr1_range[1]-Fr1_range[0]) + Fr1_range[0]
        Fr2= np.random.rand() * (Fr2_range[1]-Fr2_range[0]) + Fr2_range[0]
        true_fibers= (1-CSF)/n_fib* true_fibers
        true_fibers[:,0]*= Fr1/np.linalg.norm(true_fibers[:,0])
        true_fibers[:,1]*= Fr2/np.linalg.norm(true_fibers[:,1])
        true_fibers[:,2]*= (1-Fr1-Fr2-CSF)/np.linalg.norm(true_fibers[:,2])
    
    s= crl_dci.hardi2013_cylinders_and_iso_simulate(true_fibers, b_vals_test, b_vecs_test, 0.0021, 0.00050, 0.003)
    s= crl_aux.add_rician_noise(s, snr=snr)
    
    sig_interp= crl_aux.interpolate_s_2_sphere(s, b_vecs_test.T, v_sig, M= 1, full_circ= False)
    fodf_pred= sess.run(Y_p, feed_dict={X: sig_interp[np.newaxis,:], p_keep_hidden: 1.0}).astype(np.float64)
    
    pred_fibers, pred_resps= crl_dci.find_dominant_fibers_dipy_way(sphere_fod, np.squeeze(fodf_pred), 40, n_fib, peak_thr=.01, optimize=False, Psi= None, opt_tol=1e-7)
    
    
    n_pred= pred_fibers.shape[1]
    if n_pred>n_fib_max:
        argsort= np.argsort(pred_resps)[::-1][:n_fib_max]
        pred_fibers= pred_fibers[:,argsort]
        pred_resps=  pred_resps[argsort]
        n_pred= n_fib_max
    
    ang_diff= crl_dci.compute_min_angle_between_true_pred( true_fibers, pred_fibers )
    
    Pred_N[i_test]= n_pred
    Pred_E[i_test,:n_pred]= ang_diff
    

CM= np.zeros((3,4))
for i_test in range(N_test):
    CM[ (i_test*3)//N_test, Pred_N[i_test]]+= 1


AE= np.array([ Pred_E[:N_test//3,0].mean()   ])


























































