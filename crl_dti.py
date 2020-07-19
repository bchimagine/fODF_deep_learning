#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:09:14 2019

@author: ch209389
"""



#from __future__ import division

import numpy as np
#from numpy import dot
#from dipy.core.geometry import sphere2cart
#from dipy.core.geometry import vec2vec_rotmat
#from dipy.reconst.utils import dki_design_matrix
#from scipy.special import jn
#from dipy.data import get_fnames
#from dipy.core.gradients import gradient_table
#import scipy.optimize as opt
#import pybobyqa
#from dipy.data import get_sphere
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#import SimpleITK as sitk
#from sklearn import linear_model
#from sklearn.linear_model import OrthogonalMatchingPursuit
#from dipy.direction.peaks import peak_directions
#import spams
#import dipy.core.sphere as dipysphere
#from tqdm import tqdm




def from_lower_triangular(D):
    
    tensor_indices = np.array([[0, 3, 4],
                           [3, 1, 5],
                           [4, 5, 2]])
    
    return D[..., tensor_indices]

def design_matrix(bvals, bvecs):
    
    '''Design matrix for DTI computation'''
    
    B = np.zeros(( len(bvals), 6))
    
    B[:, 0] = -     bvecs[:, 0] * bvecs[:, 0] * bvals
    B[:, 1] = -     bvecs[:, 1] * bvecs[:, 1] * bvals
    B[:, 2] = -     bvecs[:, 2] * bvecs[:, 2] * bvals
    B[:, 3] = - 2 * bvecs[:, 0] * bvecs[:, 1] * bvals
    B[:, 4] = - 2 * bvecs[:, 0] * bvecs[:, 2] * bvals
    B[:, 5] = - 2 * bvecs[:, 1] * bvecs[:, 2] * bvals
    
    return B


def rho_2_d(Rho):
    
    '''Convert the factorization of a diffusion tensor to the tensor itself.'''
    
    U= np.array( [ [ Rho[0] , Rho[3] , Rho[5] ] , 
                   [   0    , Rho[1] , Rho[4] ] ,
                   [   0    ,    0   , Rho[2] ] ] )
    
    return np.matmul(U.T, U)


def angle_between_vectors(v1, v2):
    
    '''Compute enalge between two vectors in 3D'''
    
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def angle_between_vectors_0_90(v1, v2):
    
    '''Compute enalge between two vectors in 3D'''
    
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    ang= np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    if ang>np.pi/2:
        ang= np.pi - ang
    
    return ang


def spherical_2_unit_vector(theta, phi):
    
    v= np.array( [ np.sin(theta)*np.cos(phi),  np.sin(theta)*np.sin(phi), np.cos(theta) ] )
    
    return v
    

def angle_between_vectors_sph(r1, r2):
    
    '''Compute enalge between two vectors in spherical 3D'''
    
    v1= spherical_2_unit_vector(r1[0], r1[1])
    v2= spherical_2_unit_vector(r2[0], r2[1])
    
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def angle_between_vectors_cart_sph(r1, r2):
    
    '''Compute enalge between two vectors in spherical 3D'''
    
    v1= r1
    v2= spherical_2_unit_vector(r2[0], r2[1])
    
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def angle_between_tensors(D1, D2):
    
    '''Compute enalge between the major eigen-vectors of two tensors'''
    
    try:
        
        eigenvals, eigenvecs = np.linalg.eigh(D1)
        
        order = eigenvals.argsort()[::-1]
        eigenvecs = eigenvecs[:, order]
        eigenvals = eigenvals[order]
        
        ev1_1 = eigenvecs[:,0]
        
        eigenvals, eigenvecs = np.linalg.eigh(D2)
        
        order = eigenvals.argsort()[::-1]
        eigenvecs = eigenvecs[:, order]
        eigenvals = eigenvals[order]
        
        ev1_2 = eigenvecs[:,0]
        
        return angle_between_vectors(ev1_1, ev1_2)
        
    except:
        
        return None




def fa_from_eigv(lam1, lam2, lam3):
    
    lam_m= (lam1+lam2+lam3)/3
    
    num= np.sqrt( (lam1- lam_m)**2 + (lam2- lam_m)**2 + (lam3- lam_m)**2 )
    den= np.sqrt( lam1**2 + lam2**2 + lam3**2 )
    
    fa= np.sqrt(3/2) * num/den
    
    return fa





def cnls_resid(Rho, design_matrix, data, weight):
    
    tensor= np.array([ Rho[0]**2 , 
                       Rho[1]**2 + Rho[3]**2 , 
                       Rho[2]**2 + Rho[4]**2 + Rho[5]**2,
                       Rho[0]*Rho[3],
                       Rho[0]*Rho[5],
                       Rho[1]*Rho[4] + Rho[3]*Rho[5] ])
    
    y = np.exp(np.matmul(design_matrix, tensor))
    
    residuals = data - y
    
    residuals= weight * residuals
    
    return residuals




def cwlls_resid(Rho, design_matrix, data, weight):
    
    tensor= np.array([ Rho[0]**2 , 
                       Rho[1]**2 + Rho[3]**2 , 
                       Rho[2]**2 + Rho[4]**2 + Rho[5]**2,
                       Rho[0]*Rho[3],
                       Rho[0]*Rho[5],
                       Rho[1]*Rho[4] + Rho[3]*Rho[5] ])
    
    y = np.matmul(design_matrix, tensor)
    
    residuals = data - y
    
    residuals= weight * residuals
    
    return residuals









def nls_resid(tensor, design_matrix, data, weight):
    
    y = np.exp(np.matmul(design_matrix, tensor))
    
    residuals = data - y
    
    residuals= weight * residuals
    
    return residuals





def fa_from_tensor(my_tensor):
    
    try:
                        
        eigenvals, eigenvecs = np.linalg.eigh(my_tensor)
        
        order = eigenvals.argsort()[::-1]
        eigenvecs = eigenvecs[:, order]
        eigenvals = eigenvals[order]
        
        ev1, ev2, ev3 = eigenvals
        all_zero = (eigenvals == 0).all(axis=0)
        fa = np.sqrt(0.5 * ((ev1 - ev2) ** 2 +
                            (ev2 - ev3) ** 2 +
                            (ev3 - ev1) ** 2) /
                     ((eigenvals * eigenvals).sum(0) + all_zero))
                
    except:
        
        fa= 0
        
    return fa






def cfa_from_tensor(my_tensor):
    
    try:
                        
        eigenvals, eigenvecs = np.linalg.eigh(my_tensor)
        
        order = eigenvals.argsort()[::-1]
        eigenvecs = eigenvecs[:, order]
        eigenvals = eigenvals[order]
        
        ev1, ev2, ev3 = eigenvals
        all_zero = (eigenvals == 0).all(axis=0)
        fa = np.sqrt(0.5 * ((ev1 - ev2) ** 2 +
                            (ev2 - ev3) ** 2 +
                            (ev3 - ev1) ** 2) /
                     ((eigenvals * eigenvals).sum(0) + all_zero))
        
        cfa= np.abs( eigenvecs[:,0]*fa )
                
    except:
        
        fa= 0
        
    return fa, cfa








def evals_and_evecs_from_tensor(my_tensor):
    
    try:
                        
        eigenvals, eigenvecs = np.linalg.eigh(my_tensor)
        
        order = eigenvals.argsort()[::-1]
        eigenvecs = eigenvecs[:, order]
        eigenvals = eigenvals[order]
        
    except:
        
        eigenvals= np.zeros(3)
        eigenvecs= np.zeros( (3,3) )
        
    return eigenvals, eigenvecs

















