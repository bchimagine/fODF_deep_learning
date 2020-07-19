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
import scipy.optimize as opt
#import pybobyqa
#from dipy.data import get_sphere
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import SimpleITK as sitk
from sklearn import linear_model
#from sklearn.linear_model import OrthogonalMatchingPursuit
#from dipy.direction.peaks import peak_directions
#import spams
#import dipy.core.sphere as dipysphere
#from tqdm import tqdm
from scipy import linalg
import numpy.polynomial.polynomial as poly
from scipy.stats import f
from scipy.special import lpmv, gammaln
from scipy.special import genlaguerre, gamma, hyp2f1
from math import factorial
from dipy.reconst.recspeed import local_maxima
from dipy.reconst.recspeed import remove_similar_vertices
from numpy import matlib as mb


def two_stick_and_iso_simulate(Rho, bvals_vector, bvecs_vector):
    
    tensor1_0= np.diag([ np.exp(Rho[0]) , 0 , 0 ])
    
    tensor2_0= np.diag([ np.exp(Rho[1]) , 0 , 0 ])
    
    R1= np.array( [ [     1      ,           0        ,           0           ],
                    [     0      ,   np.cos(Rho[2])   ,    -np.sin(Rho[2])    ],
                    [     0      ,   np.sin(Rho[2])   ,     np.cos(Rho[2])    ]] )
    
    R2= np.array( [ [  np.cos(Rho[3])    ,     0    ,   np.sin(Rho[3])    ],
                    [         0          ,     1    ,       0             ],
                    [ -np.sin(Rho[3])    ,     0    ,   np.cos(Rho[3])    ]] )
    
    R31= np.array( [ [ np.cos(Rho[4]-Rho[5])   ,   -np.sin(Rho[4]-Rho[5])   ,     0    ],
                    [  np.sin(Rho[4]-Rho[5])   ,    np.cos(Rho[4]-Rho[5])    ,     0    ],
                    [          0               ,            0               ,     1    ]] )
    
    R32= np.array( [ [ np.cos(Rho[4]+Rho[5])   ,   -np.sin(Rho[4]+Rho[5])   ,     0    ],
                    [  np.sin(Rho[4]+Rho[5])   ,    np.cos(Rho[4]+Rho[5])   ,     0    ],
                    [          0               ,            0               ,     1    ]] )
    
    R_1= np.matmul( R1 , np.matmul( R2, R31 )  )
    R_2= np.matmul( R1 , np.matmul( R2, R32 )  )
    
    tensor1= np.matmul( R_1.T , np.matmul( tensor1_0 , R_1 ) )
    tensor2= np.matmul( R_2.T , np.matmul( tensor2_0 , R_2 ) )
    
    y1 = np.exp( - bvals_vector * np.sum( np.matmul(bvecs_vector.T,tensor1)*bvecs_vector.T, axis=1) )
    y2 = np.exp( - bvals_vector * np.sum( np.matmul(bvecs_vector.T,tensor2)*bvecs_vector.T, axis=1) )
    
    y3 = np.exp( - Rho[6]*bvals_vector )
    
    data_sim = Rho[7]*y1 + Rho[8]*y2 + (1-Rho[7]-Rho[8])*y3
    
    return data_sim


def two_stick_and_iso_resid(Rho, bvals_vector, bvecs_vector, data, weight):
    
    tensor1_0= np.diag([ np.exp(Rho[0]) , 0 , 0 ])
    
    tensor2_0= np.diag([ np.exp(Rho[1]) , 0 , 0 ])
    
    R1= np.array( [ [     1      ,           0        ,           0           ],
                    [     0      ,   np.cos(Rho[2])   ,    -np.sin(Rho[2])    ],
                    [     0      ,   np.sin(Rho[2])   ,     np.cos(Rho[2])    ]] )
    
    R2= np.array( [ [  np.cos(Rho[3])    ,     0    ,   np.sin(Rho[3])    ],
                    [         0          ,     1    ,       0             ],
                    [ -np.sin(Rho[3])    ,     0    ,   np.cos(Rho[3])    ]] )
    
    R31= np.array( [ [ np.cos(Rho[4]-Rho[5])   ,   -np.sin(Rho[4]-Rho[5])   ,     0    ],
                    [  np.sin(Rho[4]-Rho[5])   ,    np.cos(Rho[4]-Rho[5])    ,     0    ],
                    [          0               ,            0               ,     1    ]] )
    
    R32= np.array( [ [ np.cos(Rho[4]+Rho[5])   ,   -np.sin(Rho[4]+Rho[5])   ,     0    ],
                    [  np.sin(Rho[4]+Rho[5])   ,    np.cos(Rho[4]+Rho[5])   ,     0    ],
                    [          0               ,            0               ,     1    ]] )
    
    R_1= np.matmul( R1 , np.matmul( R2, R31 )  )
    R_2= np.matmul( R1 , np.matmul( R2, R32 )  )
    
    tensor1= np.matmul( R_1.T , np.matmul( tensor1_0 , R_1 ) )
    tensor2= np.matmul( R_2.T , np.matmul( tensor2_0 , R_2 ) )
    
    y1 = np.exp( - bvals_vector * np.sum( np.matmul(bvecs_vector.T,tensor1)*bvecs_vector.T, axis=1) )
    y2 = np.exp( - bvals_vector * np.sum( np.matmul(bvecs_vector.T,tensor2)*bvecs_vector.T, axis=1) )
    
    y3 = np.exp( - Rho[6]*bvals_vector )
    
    residuals = data - Rho[7]*y1 - Rho[8]*y2 - (1-Rho[7]-Rho[8])*y3
    
    residuals= weight * residuals
    
    return residuals

def two_stick_and_iso_params(Rho, bvals_vector, bvecs_vector, data):
    
    tensor1_0= np.diag([ np.exp(Rho[0]) , 0 , 0 ])
    
    tensor2_0= np.diag([ np.exp(Rho[1]) , 0 , 0 ])
    
    R1= np.array( [ [     1      ,           0        ,           0           ],
                    [     0      ,   np.cos(Rho[2])   ,    -np.sin(Rho[2])    ],
                    [     0      ,   np.sin(Rho[2])   ,     np.cos(Rho[2])    ]] )
    
    R2= np.array( [ [  np.cos(Rho[3])    ,     0    ,   np.sin(Rho[3])    ],
                    [         0          ,     1    ,       0             ],
                    [ -np.sin(Rho[3])    ,     0    ,   np.cos(Rho[3])    ]] )
    
    R31= np.array( [ [ np.cos(Rho[4]-Rho[5])   ,   -np.sin(Rho[4]-Rho[5])   ,     0    ],
                    [  np.sin(Rho[4]-Rho[5])   ,    np.cos(Rho[4]-Rho[5])    ,     0    ],
                    [          0               ,            0               ,     1    ]] )
    
    R32= np.array( [ [ np.cos(Rho[4]+Rho[5])   ,   -np.sin(Rho[4]+Rho[5])   ,     0    ],
                    [  np.sin(Rho[4]+Rho[5])   ,    np.cos(Rho[4]+Rho[5])   ,     0    ],
                    [          0               ,            0               ,     1    ]] )
    
    R_1= np.matmul( R1 , np.matmul( R2, R31 )  )
    R_2= np.matmul( R1 , np.matmul( R2, R32 )  )
    
    tensor1= np.matmul( R_1.T , np.matmul( tensor1_0 , R_1 ) )
    tensor2= np.matmul( R_2.T , np.matmul( tensor2_0 , R_2 ) )
    
    y1 = np.exp( - bvals_vector * np.sum( np.matmul(bvecs_vector.T,tensor1)*bvecs_vector.T, axis=1) )
    y2 = np.exp( - bvals_vector * np.sum( np.matmul(bvecs_vector.T,tensor2)*bvecs_vector.T, axis=1) )
    
    y3 = np.exp( - Rho[6]*bvals_vector )
    
    y_pred= Rho[7]*y1 + Rho[8]*y2 + (1-Rho[7]-Rho[8])*y3
    
    residuals = data - y_pred
    
    fr= np.array( [Rho[7], Rho[8], 1- Rho[7]- Rho[8]] )
        
    return tensor1, tensor2, fr, Rho[6], y_pred, np.linalg.norm(residuals)




















def two_simplified_tensor_and_iso_simulate(Rho, bvals_vector, bvecs_vector):
    
    tensor1_0= np.diag([ np.exp(Rho[0]) + 0.5* np.exp(Rho[1]) + 0.5*np.exp(Rho[3]),  
                       np.exp(Rho[1]) , 
                       np.exp(Rho[1]) ])
    
    tensor2_0= np.diag([ np.exp(Rho[2]) + 0.5* np.exp(Rho[1]) + 0.5*np.exp(Rho[3]),  
                       np.exp(Rho[3]) , 
                       np.exp(Rho[3]) ])
    
    R1= np.array( [ [     1      ,           0        ,           0           ],
                    [     0      ,   np.cos(Rho[4])   ,    -np.sin(Rho[4])    ],
                    [     0      ,   np.sin(Rho[4])   ,     np.cos(Rho[4])    ]] )
    
    R2= np.array( [ [  np.cos(Rho[5])    ,     0    ,   np.sin(Rho[5])    ],
                    [         0          ,     1    ,       0             ],
                    [ -np.sin(Rho[5])    ,     0    ,   np.cos(Rho[5])    ]] )
    
    R31= np.array( [ [ np.cos(Rho[6]-Rho[7])   ,   -np.sin(Rho[6]-Rho[7])   ,     0    ],
                    [   np.sin(Rho[6]-Rho[7])   ,   np.cos(Rho[6]-Rho[7])    ,     0    ],
                    [           0               ,            0               ,     1    ]] )
    
    R32= np.array( [ [ np.cos(Rho[6]+Rho[7])   ,   -np.sin(Rho[6]+Rho[7])   ,     0    ],
                    [   np.sin(Rho[6]+Rho[7])   ,    np.cos(Rho[6]+Rho[7])   ,     0    ],
                    [           0               ,            0               ,     1    ]] )
    
    R_1= np.matmul( R1 , np.matmul( R2, R31 )  )
    R_2= np.matmul( R1 , np.matmul( R2, R32 )  )
    
    tensor1= np.matmul( R_1.T , np.matmul( tensor1_0 , R_1 ) )
    tensor2= np.matmul( R_2.T , np.matmul( tensor2_0 , R_2 ) )
    
    y1 = np.exp( - bvals_vector * np.sum( np.matmul(bvecs_vector.T,tensor1)*bvecs_vector.T, axis=1) )
    y2 = np.exp( - bvals_vector * np.sum( np.matmul(bvecs_vector.T,tensor2)*bvecs_vector.T, axis=1) )
    
    y3 = np.exp( - Rho[8]*bvals_vector )
    
    data_sim = Rho[9]*y1 + Rho[10]*y2 + (1-Rho[9]-Rho[10])*y3
    
    return data_sim

def two_simplified_tensor_and_iso_resid(Rho, bvals_vector, bvecs_vector, data, weight):
    
    tensor1_0= np.diag([ np.exp(Rho[0]) + 0.5* np.exp(Rho[1]) + 0.5*np.exp(Rho[3]),  
                       np.exp(Rho[1]) , 
                       np.exp(Rho[1]) ])
    
    tensor2_0= np.diag([ np.exp(Rho[2]) + 0.5* np.exp(Rho[1]) + 0.5*np.exp(Rho[3]),  
                       np.exp(Rho[3]) , 
                       np.exp(Rho[3]) ])
    
    R1= np.array( [ [     1      ,           0        ,           0           ],
                    [     0      ,   np.cos(Rho[4])   ,    -np.sin(Rho[4])    ],
                    [     0      ,   np.sin(Rho[4])   ,     np.cos(Rho[4])    ]] )
    
    R2= np.array( [ [  np.cos(Rho[5])    ,     0    ,   np.sin(Rho[5])    ],
                    [         0          ,     1    ,       0             ],
                    [ -np.sin(Rho[5])    ,     0    ,   np.cos(Rho[5])    ]] )
    
    R31= np.array( [ [ np.cos(Rho[6]-Rho[7])   ,   -np.sin(Rho[6]-Rho[7])   ,     0    ],
                    [   np.sin(Rho[6]-Rho[7])   ,   np.cos(Rho[6]-Rho[7])    ,     0    ],
                    [           0               ,            0               ,     1    ]] )
    
    R32= np.array( [ [ np.cos(Rho[6]+Rho[7])   ,   -np.sin(Rho[6]+Rho[7])   ,     0    ],
                    [   np.sin(Rho[6]+Rho[7])   ,    np.cos(Rho[6]+Rho[7])   ,     0    ],
                    [           0               ,            0               ,     1    ]] )
    
    R_1= np.matmul( R1 , np.matmul( R2, R31 )  )
    R_2= np.matmul( R1 , np.matmul( R2, R32 )  )
    
    tensor1= np.matmul( R_1.T , np.matmul( tensor1_0 , R_1 ) )
    tensor2= np.matmul( R_2.T , np.matmul( tensor2_0 , R_2 ) )
    
    y1 = np.exp( - bvals_vector * np.sum( np.matmul(bvecs_vector.T,tensor1)*bvecs_vector.T, axis=1) )
    y2 = np.exp( - bvals_vector * np.sum( np.matmul(bvecs_vector.T,tensor2)*bvecs_vector.T, axis=1) )
    
    y3 = np.exp( - Rho[8]*bvals_vector )
    
    residuals = data - Rho[9]*y1 - Rho[10]*y2 - (1-Rho[9]-Rho[10])*y3
    
    residuals= weight * residuals
    
    return residuals

def two_simplified_tensor_and_iso_resid_bbq(Rho, bvals_vector, bvecs_vector, data, weight):

    tensor1_0= np.diag([ np.exp(Rho[0]) + 0.5* np.exp(Rho[1]) + 0.5*np.exp(Rho[3]),
                       np.exp(Rho[1]) ,
                       np.exp(Rho[1]) ])

    tensor2_0= np.diag([ np.exp(Rho[2]) + 0.5* np.exp(Rho[1]) + 0.5*np.exp(Rho[3]),
                       np.exp(Rho[3]) ,
                       np.exp(Rho[3]) ])

    R1= np.array( [ [     1      ,           0        ,           0           ],
                    [     0      ,   np.cos(Rho[4])   ,    -np.sin(Rho[4])    ],
                    [     0      ,   np.sin(Rho[4])   ,     np.cos(Rho[4])    ]] )

    R2= np.array( [ [  np.cos(Rho[5])    ,     0    ,   np.sin(Rho[5])    ],
                    [         0          ,     1    ,       0             ],
                    [ -np.sin(Rho[5])    ,     0    ,   np.cos(Rho[5])    ]] )

    R31= np.array( [ [ np.cos(Rho[6]-Rho[7])   ,   -np.sin(Rho[6]-Rho[7])   ,     0    ],
                    [   np.sin(Rho[6]-Rho[7])   ,   np.cos(Rho[6]-Rho[7])    ,     0    ],
                    [           0               ,            0               ,     1    ]] )

    R32= np.array( [ [ np.cos(Rho[6]+Rho[7])   ,   -np.sin(Rho[6]+Rho[7])   ,     0    ],
                    [   np.sin(Rho[6]+Rho[7])   ,    np.cos(Rho[6]+Rho[7])   ,     0    ],
                    [           0               ,            0               ,     1    ]] )

    R_1= np.matmul( R1 , np.matmul( R2, R31 )  )
    R_2= np.matmul( R1 , np.matmul( R2, R32 )  )

    tensor1= np.matmul( R_1.T , np.matmul( tensor1_0 , R_1 ) )
    tensor2= np.matmul( R_2.T , np.matmul( tensor2_0 , R_2 ) )

    y1 = np.exp( - bvals_vector * np.sum( np.matmul(bvecs_vector.T,tensor1)*bvecs_vector.T, axis=1) )
    y2 = np.exp( - bvals_vector * np.sum( np.matmul(bvecs_vector.T,tensor2)*bvecs_vector.T, axis=1) )

    y3 = np.exp( - Rho[8]*bvals_vector )

    residuals = data - Rho[9]*y1 - Rho[10]*y2 - (1-Rho[9]-Rho[10])*y3

    residuals= np.sum( (weight**2) * (residuals**2) )

    return residuals

def two_simplified_tensor_and_iso_params(Rho, bvals_vector, bvecs_vector, data):
    
    tensor1_0= np.diag([ np.exp(Rho[0]) + 0.5* np.exp(Rho[1]) + 0.5*np.exp(Rho[3]),  
                       np.exp(Rho[1]) , 
                       np.exp(Rho[1]) ])
    
    tensor2_0= np.diag([ np.exp(Rho[2]) + 0.5* np.exp(Rho[1]) + 0.5*np.exp(Rho[3]),  
                       np.exp(Rho[3]) , 
                       np.exp(Rho[3]) ])
    
    R1= np.array( [ [     1      ,           0        ,           0           ],
                    [     0      ,   np.cos(Rho[4])   ,    -np.sin(Rho[4])    ],
                    [     0      ,   np.sin(Rho[4])   ,     np.cos(Rho[4])    ]] )
    
    R2= np.array( [ [  np.cos(Rho[5])    ,     0    ,   np.sin(Rho[5])    ],
                    [         0          ,     1    ,       0             ],
                    [ -np.sin(Rho[5])    ,     0    ,   np.cos(Rho[5])    ]] )
    
    R31= np.array( [ [ np.cos(Rho[6]-Rho[7])   ,   -np.sin(Rho[6]-Rho[7])   ,     0    ],
                    [   np.sin(Rho[6]-Rho[7])   ,   np.cos(Rho[6]-Rho[7])    ,     0    ],
                    [           0               ,            0               ,     1    ]] )
    
    R32= np.array( [ [ np.cos(Rho[6]+Rho[7])   ,   -np.sin(Rho[6]+Rho[7])   ,     0    ],
                    [   np.sin(Rho[6]+Rho[7])   ,    np.cos(Rho[6]+Rho[7])   ,     0    ],
                    [           0               ,            0               ,     1    ]] )
    
    R_1= np.matmul( R1 , np.matmul( R2, R31 )  )
    R_2= np.matmul( R1 , np.matmul( R2, R32 )  )
    
    tensor1= np.matmul( R_1.T , np.matmul( tensor1_0 , R_1 ) )
    tensor2= np.matmul( R_2.T , np.matmul( tensor2_0 , R_2 ) )
    
    y1 = np.exp( - bvals_vector * np.sum( np.matmul(bvecs_vector.T,tensor1)*bvecs_vector.T, axis=1) )
    y2 = np.exp( - bvals_vector * np.sum( np.matmul(bvecs_vector.T,tensor2)*bvecs_vector.T, axis=1) )
    
    y3 = np.exp( - Rho[8]*bvals_vector )
    
    y_pred= Rho[9]*y1 + Rho[10]*y2 + (1-Rho[9]-Rho[10])*y3
    
    residuals = data - y_pred
    
    fr= np.array( [Rho[9], Rho[10], 1- Rho[9]- Rho[10]] )
    
    return tensor1, tensor2, fr, Rho[8], y_pred, np.linalg.norm(residuals)














def two_tensor_and_iso_simulate(Rho, design_matrix, bvals_vector):
    
    tensor1= np.array([ Rho[0]**2 , 
                       Rho[1]**2 + Rho[3]**2 , 
                       Rho[2]**2 + Rho[4]**2 + Rho[5]**2,
                       Rho[0]*Rho[3],
                       Rho[0]*Rho[5],
                       Rho[1]*Rho[4] + Rho[3]*Rho[5] ])
    
    tensor2= np.array([ Rho[6]**2 , 
                       Rho[7]**2 + Rho[9]**2 , 
                       Rho[8]**2 + Rho[10]**2 + Rho[11]**2,
                       Rho[6]*Rho[9],
                       Rho[6]*Rho[11],
                       Rho[7]*Rho[10] + Rho[9]*Rho[11] ])
    
    y1 = np.exp(np.matmul(design_matrix, tensor1))
    
    y2 = np.exp(np.matmul(design_matrix, tensor2))
    
    y3 = np.exp( - Rho[12]*bvals_vector )
    
    data_sim = Rho[13]*y1 + Rho[14]*y2 + (1-Rho[13]-Rho[14])*y3
    
    return data_sim

def two_tensor_and_iso_resid(Rho, design_matrix, bvals_vector, data, weight):
    
    tensor1= np.array([ Rho[0]**2 , 
                       Rho[1]**2 + Rho[3]**2 , 
                       Rho[2]**2 + Rho[4]**2 + Rho[5]**2,
                       Rho[0]*Rho[3],
                       Rho[0]*Rho[5],
                       Rho[1]*Rho[4] + Rho[3]*Rho[5] ])
    
    tensor2= np.array([ Rho[6]**2 , 
                       Rho[7]**2 + Rho[9]**2 , 
                       Rho[8]**2 + Rho[10]**2 + Rho[11]**2,
                       Rho[6]*Rho[9],
                       Rho[6]*Rho[11],
                       Rho[7]*Rho[10] + Rho[9]*Rho[11] ])
    
    y1 = np.exp(np.matmul(design_matrix, tensor1))
    
    y2 = np.exp(np.matmul(design_matrix, tensor2))
    
    y3 = np.exp( - Rho[12]*bvals_vector )
    
    residuals = data - Rho[13]*y1 - Rho[14]*y2 - (1-Rho[13]-Rho[14])*y3
    
    residuals= weight * residuals
    
    return residuals

def two_tensor_and_iso_resid_bbq(Rho, design_matrix, bvals_vector, data, weight):
    
    tensor1= np.array([ Rho[0]**2 , 
                       Rho[1]**2 + Rho[3]**2 , 
                       Rho[2]**2 + Rho[4]**2 + Rho[5]**2,
                       Rho[0]*Rho[3],
                       Rho[0]*Rho[5],
                       Rho[1]*Rho[4] + Rho[3]*Rho[5] ])
    
    tensor2= np.array([ Rho[6]**2 , 
                       Rho[7]**2 + Rho[9]**2 , 
                       Rho[8]**2 + Rho[10]**2 + Rho[11]**2,
                       Rho[6]*Rho[9],
                       Rho[6]*Rho[11],
                       Rho[7]*Rho[10] + Rho[9]*Rho[11] ])
    
    y1 = np.exp(np.matmul(design_matrix, tensor1))
    
    y2 = np.exp(np.matmul(design_matrix, tensor2))
    
    y3 = np.exp( - Rho[12]*bvals_vector )
    
    residuals = data - Rho[13]*y1 - Rho[14]*y2 - (1-Rho[13]-Rho[14])*y3
    
    residuals= np.sum( (weight**2) * (residuals**2) )
    
    return residuals

def two_tensor_and_iso_params(Rho, design_matrix, bvals_vector, data):
    
    tensor1= np.array([ Rho[0]**2 , 
                       Rho[1]**2 + Rho[3]**2 , 
                       Rho[2]**2 + Rho[4]**2 + Rho[5]**2,
                       Rho[0]*Rho[3],
                       Rho[0]*Rho[5],
                       Rho[1]*Rho[4] + Rho[3]*Rho[5] ])
    
    tensor2= np.array([ Rho[6]**2 , 
                       Rho[7]**2 + Rho[9]**2 , 
                       Rho[8]**2 + Rho[10]**2 + Rho[11]**2,
                       Rho[6]*Rho[9],
                       Rho[6]*Rho[11],
                       Rho[7]*Rho[10] + Rho[9]*Rho[11] ])
    
    y1 = np.exp(np.matmul(design_matrix, tensor1))
    
    y2 = np.exp(np.matmul(design_matrix, tensor2))
    
    y3 = np.exp( - Rho[12]*bvals_vector )
    
    y_pred= Rho[13]*y1 + Rho[14]*y2 + (1-Rho[13]-Rho[14])*y3
    
    residuals = data - y_pred
    
    fr= np.array( [Rho[13], Rho[14], 1- Rho[13]- Rho[14]] )
    
    return tensor1, tensor2, fr, Rho[12], y_pred, np.linalg.norm(residuals)























def two_polar_stick_and_iso_simulate(R, b, q, lam_bar= 0.0007):
    
    y1 = np.exp( -b * ( R[0] + 3*(lam_bar-R[0])* ( q[:,0]*np.sin(R[1])*np.cos(R[2]) + q[:,1]*np.sin(R[1])*np.sin(R[2]) + q[:,2]*np.cos(R[1]) )**2 ) )
    
    y2 = np.exp( -b * ( R[3] + 3*(lam_bar-R[3])* ( q[:,0]*np.sin(R[4])*np.cos(R[5]) + q[:,1]*np.sin(R[4])*np.sin(R[5]) + q[:,2]*np.cos(R[4]) )**2 ) )
    
    y3 = np.exp( - R[6]*b )
    
    data_sim = R[7]*y1 + R[8]*y2 + (1-R[7]-R[8])*y3
    
    return data_sim

def two_polar_stick_and_iso_resid(R, b, q, data, weight, lam_bar= 0.0007):
    
    y1 = np.exp( -b * ( R[0] + 3*(lam_bar-R[0])* ( q[:,0]*np.sin(R[1])*np.cos(R[2]) + q[:,1]*np.sin(R[1])*np.sin(R[2]) + q[:,2]*np.cos(R[1]) )**2 ) )
    
    y2 = np.exp( -b * ( R[3] + 3*(lam_bar-R[3])* ( q[:,0]*np.sin(R[4])*np.cos(R[5]) + q[:,1]*np.sin(R[4])*np.sin(R[5]) + q[:,2]*np.cos(R[4]) )**2 ) )
    
    y3 = np.exp( - R[6]*b )
    
    residuals = data - R[7]*y1 - R[8]*y2 - (1-R[7]-R[8])*y3
    
    residuals= weight * residuals
    
    return residuals

def two_polar_stick_and_iso_resid_bbq(R, b, q, data, weight, lam_bar= 0.0007):

    y1 = np.exp( -b * ( R[0] + 3*(lam_bar-R[0])* ( q[:,0]*np.sin(R[1])*np.cos(R[2]) + q[:,1]*np.sin(R[1])*np.sin(R[2]) + q[:,2]*np.cos(R[1]) )**2 ) )

    y2 = np.exp( -b * ( R[3] + 3*(lam_bar-R[3])* ( q[:,0]*np.sin(R[4])*np.cos(R[5]) + q[:,1]*np.sin(R[4])*np.sin(R[5]) + q[:,2]*np.cos(R[4]) )**2 ) )

    y3 = np.exp( - R[6]*b )

    residuals = data - R[7]*y1 - R[8]*y2 - (1-R[7]-R[8])*y3

    residuals= np.sum( (weight**2) * (residuals**2) )

    return residuals










def sd_matrix(lam, d_iso, bvl, bvc, sph, with_iso=False):
    
    if with_iso:
        H= np.zeros( ( len(bvl), len(sph)+1 ) )
    else:
        H= np.zeros( ( len(bvl), len(sph) ) )
    
    for i in range( len(bvl) ):
        
        for j in range( len(sph) ):
            
            cs2= np.dot( bvc[i,:], sph[j,:])**2
            
            H[i,j]= np.exp( -bvl[i]* (lam[0]*cs2 + lam[1]*(1-cs2))  )
        
        if with_iso:
            H[i,j+1]= np.exp( -bvl[i]* d_iso )
    
    return H



def sd_matrix_fast(lam, d_iso, bvl, bvc, sph, with_iso=False):
    
    if with_iso:
        H= np.zeros( ( len(bvl), len(sph)+1 ) )
    else:
        H= np.zeros( ( len(bvl), len(sph) ) )
    
    bvl_m= mb.repmat(bvl[:,np.newaxis], 1,len(sph))
    HH= np.matmul(bvc, sph.T)**2
    HHH= (lam[0]*HH + lam[1]*(1-HH)) 
    H[:,:len(sph)]= np.exp( -bvl_m* HHH  )
    
    if with_iso:
        H[:,-1]= np.exp( -bvl* d_iso )
    
    return H

def sd_matrix_half(lam, d_iso, bvl, bvc, sph, with_iso=False):
    
    sph_x, _ = sph.shape
    sph= sph[:sph_x//2,:]
    
    if with_iso:
        H= np.zeros( ( len(bvl), len(sph)+1 ) )
    else:
        H= np.zeros( ( len(bvl), len(sph) ) )
    
    for i in range( len(bvl) ):
        
        for j in range( len(sph) ):
            
            cs2= np.dot( bvc[i,:], sph[j,:])**2
            
            H[i,j]= np.exp( -bvl[i]* (lam[0]*cs2 + lam[1]*(1-cs2))  )
        
        if with_iso:
            H[i,j+1]= np.exp( -bvl[i]* d_iso )
    
    return H

def sd_two_stick_and_iso_simulate(R, b, q, lam_bar= 0.0009):
    
    y1 = np.exp( -b * ( R[0] + 3*(lam_bar-R[0])* ( q[:,0]*np.sin(R[1])*np.cos(R[2]) + q[:,1]*np.sin(R[1])*np.sin(R[2]) + q[:,2]*np.cos(R[1]) )**2 ) )
    
    y2 = np.exp( -b * ( R[3] + 3*(lam_bar-R[3])* ( q[:,0]*np.sin(R[4])*np.cos(R[5]) + q[:,1]*np.sin(R[4])*np.sin(R[5]) + q[:,2]*np.cos(R[4]) )**2 ) )
    
    y3 = np.exp( - R[6]*b )
    
    data_sim = R[7]*y1 + R[8]*y2 + (1-R[7]-R[8])*y3
    
    return data_sim

def RL_deconv(H, s, f_0, n_iter= 100):
    
    HT= H.T
    HTH= np.matmul( HT, H )
    f= f_0.copy()
    
    for i in range(n_iter):
        
        f= f * np.matmul( HT, s ) / np.matmul( HTH, f )
        
    return f

def dRL_deconv(H, s, f_0, nu=8, etha=0.06, n_iter= 100):

    mu= np.max( [0, 1-4*s.std()] )

    HT= H.T
    HTH= np.matmul( HT, H )
    f= f_0.copy()

    for i in range(n_iter):

        r= 1 - ( f**nu ) / ( f**nu + etha**nu )

        u= 1- mu*r

        HTHf= np.matmul( HTH, f )
        HTs=  np.matmul( HT, s )

        f= f * ( 1 + u *  ( ( HTs - HTHf ) / HTHf ) )

    return f

def find_dominant_fibers(v, f_n, min_angle, n_fib):
    
    min_cos= np.cos(min_angle)
    
    fibers= np.zeros( (3, n_fib) )
    response= np.zeros(n_fib)
    
    arg_s= np.argsort(-f_n)
    
    fibers[:,0]= v[ arg_s[0], :]
    response[0]= f_n[ arg_s[0] ]
    
    k= 1
    
    for i in range(1, len(f_n)):
        
        if k<n_fib:
            
            v_c= v[arg_s[i], :]
            
            too_close= False
            
            for kk in range(k):
                
                if np.abs( np.dot(v_c, fibers[:,kk]) )>min_cos:
                    
                    too_close= True
            
            if not too_close:
                
                fibers[:,k]= v[ arg_s[i], :]
                response[k]= f_n[ arg_s[i] ]
                
                k+= 1
    
    return fibers, response

def find_dominant_fibers_2(v, f_n, min_angle, n_fib):
    
    min_cos= np.cos(min_angle)
    
    fibers= np.zeros( (3, n_fib) )
    response= np.zeros(n_fib)
    
    arg_s= np.argsort(-f_n)
    
    fibers[:,0]= v[ arg_s[0], :]
    response[0]= f_n[ arg_s[0] ]
    
    k= 1
    
    for i in range(1, len(f_n)):
        
        if k<n_fib:
            
            v_c= v[arg_s[i], :]
            
            too_close= False
            
            for kk in range(i):
                
                if np.abs( np.dot(v_c, v[arg_s[kk], :]) )>min_cos:
                    
                    too_close= True
            
            if not too_close:
                
                fibers[:,k]= v[ arg_s[i], :]
                response[k]= f_n[ arg_s[i] ]
                
                k+= 1
    
    return fibers, response



def compute_min_angle_between_fibers(R, n_fib):
    
    max_cos= 0
    
    for i in range(n_fib-1):
        for j in range(i+1,n_fib):
            
            v1= np.array( [ np.sin(R[i*5+3])*np.cos(R[i*5+4]) , np.sin(R[i*5+3])*np.sin(R[i*5+4]) , np.cos(R[i*5+3]) ] )
            v2= np.array( [ np.sin(R[j*5+3])*np.cos(R[j*5+4]) , np.sin(R[j*5+3])*np.sin(R[j*5+4]) , np.cos(R[j*5+3]) ] )
            
            cur_cos= np.abs( np.dot(v1, v2) )
            
            if cur_cos>max_cos:
                
                max_cos= cur_cos
    
    min_ang= np.arccos(max_cos)*180/np.pi
    
    return min_ang







def compute_min_angle_between_fiberset(V_set_orig):
    
    V_set= V_set_orig.copy()
    
    n_fib= V_set.shape[1]
    max_cos= 0
    
    for i in range(V_set.shape[1]):
        V_set[:,i]= V_set[:,i]/ np.linalg.norm(V_set[:,i])
        
    for i in range(n_fib-1):
        for j in range(i+1,n_fib):
            
            v1= V_set[:,i]
            v2= V_set[:,j]
            
            cur_cos= np.abs( np.dot(v1, v2) )
            
            if cur_cos>max_cos:
                
                max_cos= cur_cos
    
    min_ang= np.arccos(max_cos)*180/np.pi
    
    return min_ang





def compute_min_angle_between_vector_sets(V1_orig, V2_orig):
    
    V1= V1_orig.copy()
    V2= V2_orig.copy()
    
    assert(V1.shape[0]==3 and V2.shape[0]==3)
    
    for i in range(V1.shape[1]):
        V1[:,i]= V1[:,i]/ np.linalg.norm(V1[:,i])
    
    for i in range(V2.shape[1]):
        V2[:,i]= V2[:,i]/ np.linalg.norm(V2[:,i])
    
    max_cos= 0
    
    for i in range(V1.shape[1]):
        for j in range(V2.shape[1]):
            
            cur_cos= np.clip( np.abs( np.dot(V1[:,i], V2[:,j]) ), 0, 1)
            
            if cur_cos>max_cos:
                
                max_cos= cur_cos
    
    min_ang= np.arccos(max_cos)*180/np.pi
    
    return min_ang




def compute_min_angle_between_vector_sets_full_sphere(V1_sphere, V2_fibers):
    
    V1= V1_sphere.copy()
    V2= V2_fibers.copy()
    
    if len(V1)<len(V2):
        raise ValueError('Sphere is smaller than fibers')
    
    assert(V1.shape[0]==3 and V2.shape[0]==3)
    
    for i in range(V2.shape[1]):
        V2[:,i]= V2[:,i]/ np.linalg.norm(V2[:,i])
    
    cosines= np.clip( np.abs( np.dot(V1.T, V2) ), 0, 1)
    
    cosines= np.max(cosines, axis=1)
    
    min_ang= np.arccos(cosines)*180/np.pi
    
    return min_ang






def polar_fibers_and_iso_simulate(R, n_fib, b, q):
    
    y = (1-R[0:-1:5].sum()) * np.exp( - R[-1]*b )
    
    for i in range(n_fib):
        
        y+= R[i*5]* np.exp( -b * ( R[i*5+2] + (R[i*5+1]-R[i*5+2])* 
             ( q[:,0]*np.sin(R[i*5+3])*np.cos(R[i*5+4]) + q[:,1]*np.sin(R[i*5+3])*np.sin(R[i*5+4]) + q[:,2]*np.cos(R[i*5+3]) )**2 ) )
    
    return y



def find_dominant_fibers_dipy_way(sph, f_n, min_angle, n_fib, peak_thr=.25, optimize=False, Psi= None, opt_tol=1e-7):
    
    if optimize:
        
        v = sph.vertices
        
        '''values, indices = local_maxima(f_n, sph.edges)
        directions= v[indices,:]
        
        order = values.argsort()[::-1]
        values = values[order]
        directions = directions[order]
        
        directions, idx = remove_similar_vertices(directions, 25,
                                                  return_index=True)
        values = values[idx]
        directions= directions.T
        
        seeds = directions
        
        def SHORE_ODF_f(x):
            fx= np.dot(Psi, x)
            return -fx
        
        num_seeds = seeds.shape[1]
        theta = np.empty(num_seeds)
        phi = np.empty(num_seeds)
        values = np.empty(num_seeds)
        for i in range(num_seeds):
            peak = opt.fmin(SHORE_ODF_f, seeds[:,i], xtol=opt_tol, disp=False)
            theta[i], phi[i] = peak
            
        # Evaluate on new-found peaks
        small_sphere = Sphere(theta=theta, phi=phi)
        values = sphere_eval(small_sphere)
    
        # Sort in descending order
        order = values.argsort()[::-1]
        values = values[order]
        directions = small_sphere.vertices[order]
    
        # Remove directions that are too small
        n = search_descending(values, relative_peak_threshold)
        directions = directions[:n]
    
        # Remove peaks too close to each-other
        directions, idx = remove_similar_vertices(directions, min_separation_angle,
                                                  return_index=True)
        values = values[idx]'''
        
    else:
        
        v = sph.vertices
        
        values, indices = local_maxima(f_n, sph.edges)
        directions= v[indices,:]
        
        order = values.argsort()[::-1]
        values = values[order]
        directions = directions[order]
        
        directions, idx = remove_similar_vertices(directions, min_angle,
                                                  return_index=True)
        values = values[idx]
        directions= directions.T
        
    return directions, values




def angles_2_fibers(f_true ):
    
    f_true_sph= f_true.copy()
    
    f_true= np.zeros( (3, f_true.shape[0]) )
    
    for i in range( f_true.shape[1] ):
        
        f_true[0, i]= np.sin( f_true_sph[i,0] ) * np.cos( f_true_sph[i,1] )
        f_true[1, i]= np.sin( f_true_sph[i,0] ) * np.sin( f_true_sph[i,1] )
        f_true[2, i]= np.cos( f_true_sph[i,0] )
    
    return f_true

#def find_min_angle_diff( f_pred, f_true ):
#    
#    f_true_sph= f_true.copy()
#    
#    f_true= np.zeros( f_pred.shape )
#    
#    for i in range( f_true.shape[1] ):
#        
#        f_true[0, i]= np.sin( f_true_sph[i,0] ) * np.cos( f_true_sph[i,1] )
#        f_true[1, i]= np.sin( f_true_sph[i,0] ) * np.sin( f_true_sph[i,1] )
#        f_true[2, i]= np.cos( f_true_sph[i,0] )
#    
#    A= np.zeros( ( f_true.shape[1] , f_pred.shape[1] ) )
#    
#    for i in range( f_true.shape[1] ):
#        for j in range( f_pred.shape[1] ):
#            
#            A[i,j]= np.arccos( np.abs( np.dot( f_true[:,i] , f_pred[:,j] ) ) )
#            
#    min_angles= np.zeros( f_true.shape[1] )
#    
#    for i in range( f_true.shape[1] ):
#        
#        min_ind= np.argmin(A)
#        min_i= min_ind// A.shape[1]
#        min_j= min_ind % A.shape[1]
#        
#        min_angles[i]= A[min_i, min_j]
#        
#        A[min_i,:]= np.inf
#    
#    return min_angles

def find_min_angle_diff( f_pred, f_true ):
    
    if not f_true.shape==f_pred.shape:
        print('True and predicted fibers do not have the same shape.')
        
    A= np.zeros( ( f_true.shape[1] , f_pred.shape[1] ) )
    
    for i in range(f_true.shape[1] ):
        f_true[:,i]/= np.linalg.norm(f_true[:,i])
    
    for i in range(f_pred.shape[1] ):
        f_pred[:,i]/= np.linalg.norm(f_pred[:,i])
    
    for i in range( f_true.shape[1] ):
        for j in range( f_pred.shape[1] ):
            
            A[i,j]= np.arccos( np.abs( np.dot( f_true[:,i] , f_pred[:,j] ) ) )
            
    min_angles= np.zeros( f_true.shape[1] )
    
    for i in range( f_true.shape[1] ):
        
        min_ind= np.argmin(A)
        min_i= min_ind// A.shape[1]
        min_j= min_ind % A.shape[1]
        
        min_angles[i]= A[min_i, min_j]
        
        A[min_i,:]= np.inf
        A[:,min_j]= np.inf
    
    return min_angles






def compute_min_angle_between_true_pred(V_true, V_pred, normalize=True):
    
    V1= V_true.copy()
    V2= V_pred.copy()
    
    assert(V1.shape[0]==3 and V2.shape[0]==3)
    
    if normalize:
        for i in range(V1.shape[1]):
            V1[:,i]= V1[:,i]/ np.linalg.norm(V1[:,i])
        for i in range(V2.shape[1]):
            V2[:,i]= V2[:,i]/ np.linalg.norm(V2[:,i])
    
    min_ang= np.zeros(V2.shape[1])
    
    for i in range(V2.shape[1]):
            
            cur_cos= np.clip( np.abs( np.dot(V1.T, V2[:,i:i+1]) ), 0, 1)
            
            max_cos= cur_cos.max()
            
            min_ang[i]= np.arccos(max_cos)*180/np.pi
    
    return min_ang







def iterative_l1(H, s, lam=0.001, n_iter=2):
    
    m, n = H.shape
    i0= int( m/(4*np.log(n/m)) )
    
    clf = linear_model.Lasso(alpha=lam,positive=True, max_iter=1000)
    clf.fit( H,s  )
    f_n= clf.coef_
    
    for i_iter in range(n_iter):
        
        f_n_abs= np.sort( np.abs(f_n) )[::-1]
        eps= np.max( [ f_n_abs[i0] , 0.001 ] )
        
        w= 1/ ( np.abs(f_n) + eps )
        
        temp= spams.lassoWeighted(s[:,np.newaxis], np.asfortranarray(H), w[:,np.newaxis], L= -1, lambda1= 0.01, mode= 2, 
                                  pos= True, numThreads= -1, verbose = False)
        
        f_n= np.zeros(f_n.shape)
        f_n[ temp.indices ]= temp.data
    
    return f_n

def bayesian_l1(H, s, lam=0.001, alph= 0.001, n_iter=2):
    
    m, n = H.shape
    
    clf = linear_model.Lasso(alpha=lam,positive=True, max_iter=1000)
    clf.fit( H,s  )
    f_n= clf.coef_
    
    w= np.ones(n)
    
    for i_iter in range(n_iter):
        
        w= np.diag( w * f_n )
        
        w= np.linalg.inv( alph * np.eye(m) + np.matmul ( H, np.matmul( w, H.T ) ) )
        
        w= np.sum( np.matmul(H.T,w)*H.T, axis=1)
        
        w= np.sqrt( np.clip( w , 0, 1e6) )
        
        temp= spams.lassoWeighted(s[:,np.newaxis], np.asfortranarray(H), w[:,np.newaxis], L= -1, lambda1= 0.001, mode= 2, 
                                  pos= True, numThreads= -1, verbose = False)
        
        f_n= np.zeros(f_n.shape)
        f_n[ temp.indices ]= temp.data
    
    return f_n
















def hardi2013_cylinders_and_iso_simulate(true_fibers_orig, b, q, lam_1=0.0019, lam_2=0.0004, d_iso= 0.003):
    
    true_fibers= true_fibers_orig.copy()
    
    n_f= true_fibers.shape[1]
    
    if n_f>0:
        v= true_fibers[:,0]
        f1= np.linalg.norm(v)
        v/= f1
        y1 = np.exp( -b * ( lam_2 + (lam_1-lam_2)* ( q[:,0]* v[0]+ q[:,1]*v[1] + q[:,2]*v[2] )**2 ) )
    else:
        f1= y1= 0
    
    if n_f>1:
        v= true_fibers[:,1]
        f2= np.linalg.norm(v)
        v/= f2
        y2 = np.exp( -b * ( lam_2 + (lam_1-lam_2)* ( q[:,0]* v[0]+ q[:,1]*v[1] + q[:,2]*v[2] )**2 ) )
    else:
        f2= y2= 0
    
    if n_f>2:
        v= true_fibers[:,2]
        f3= np.linalg.norm(v)
        v/= f3
        y3 = np.exp( -b * ( lam_2 + (lam_1-lam_2)* ( q[:,0]* v[0]+ q[:,1]*v[1] + q[:,2]*v[2] )**2 ) )
    else:
        f3= y3= 0
    
    if n_f>3:
        v= true_fibers[:,3]
        f4= np.linalg.norm(v)
        v/= f4
        y4 = np.exp( -b * ( lam_2 + (lam_1-lam_2)* ( q[:,0]* v[0]+ q[:,1]*v[1] + q[:,2]*v[2] )**2 ) )
    else:
        f4= y4= 0
    
    if n_f>4:
        v= true_fibers[:,4]
        f5= np.linalg.norm(v)
        v/= f5
        y5 = np.exp( -b * ( lam_2 + (lam_1-lam_2)* ( q[:,0]* v[0]+ q[:,1]*v[1] + q[:,2]*v[2] )**2 ) )
    else:
        f5= y5= 0
    
    y_iso= np.exp( - d_iso*b )
    f_iso= 1- f1- f2- f3- f4- f5
    
    y= f1*y1 + f2*y2 + f3*y3 + f4*y4 + f5*y5 + f_iso*y_iso 
    
    return y



def polar_fibers_and_iso_resid(R, n_fib, b, q, data, weight):
    
    y = (1-R[0:-1:5].sum()) * np.exp( - R[-1]*b )
    
    for i in range(n_fib):
        
        y+= R[i*5]* np.exp( -b * ( R[i*5+2] + (R[i*5+1]-R[i*5+2])* 
             ( q[:,0]*np.sin(R[i*5+3])*np.cos(R[i*5+4]) + q[:,1]*np.sin(R[i*5+3])*np.sin(R[i*5+4]) + q[:,2]*np.cos(R[i*5+3]) )**2 ) )
    
    residuals = data - y
    
    residuals= weight * residuals
    
    return residuals

#def polar_fibers_and_iso_resid_jac(R, n_fib, b, q, data, weight):
#    
#    jac= np.zeros( ( len(R), len(b) ) )
#    
##    jac[-1,:] = - b * (1-R[0:-1:5].sum()) * 
##    y = (1-R[0:-1:5].sum()) * np.exp( - R[-1]*b )
#    
#    for i in range(n_fib):
#        
#        jac[i*5,:]= np.exp( -b * ( R[i*5+2] + (R[i*5+1]-R[i*5+2])* 
#             ( q[:,0]*np.sin(R[i*5+3])*np.cos(R[i*5+4]) + q[:,1]*np.sin(R[i*5+3])*np.sin(R[i*5+4]) + q[:,2]*np.cos(R[i*5+3]) )**2 ) )
#            - np.exp( - R[-1]*b )
#    
#    residuals = data - y
#    
#    residuals= weight * residuals
#    
#    return residuals

def polar_fibers_and_iso_resid_bbq(R, n_fib, b, q, data, weight):
    
    y = (1-R[0:-1:5].sum()) * np.exp( - R[-1]*b )
    
    for i in range(n_fib):
        
        y+= R[i*5]* np.exp( -b * ( R[i*5+2] + (R[i*5+1]-R[i*5+2])* 
             ( q[:,0]*np.sin(R[i*5+3])*np.cos(R[i*5+4]) + q[:,1]*np.sin(R[i*5+3])*np.sin(R[i*5+4]) + q[:,2]*np.cos(R[i*5+3]) )**2 ) )
    
    residuals = data - y
    
    residuals= np.sum( (weight**2) * (residuals**2) )
    
    return residuals

def polar_fibers_and_iso_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003, min_separation= 45, n_try_angle= 100):
    
    R_0=        np.zeros( 5*n_fib+1 )
    bounds_lo=  np.zeros( 5*n_fib+1 )
    bounds_hi=  np.zeros( 5*n_fib+1 )
    
    good_separation= False
    
    i_try_angle= 0
    
    while not good_separation:
        
        for i in range(n_fib):
            R_0[5*i+3]= np.pi/10 + np.random.rand() * ( np.pi - np.pi/5)
            R_0[5*i+4]= np.random.rand() * 2 * np.pi
        
        i_try_angle+= 1
        separation_temp= compute_min_angle_between_fibers(R_0, n_fib)
        #print(separation_temp)
        
        if separation_temp>min_separation:
            good_separation= True
            
        if i_try_angle==n_try_angle:
            good_separation= True
            print('Fibers are not well-separated!')
        
    R_0[-1]= d_iso
    if n_fib>0:
        bounds_lo[-1]= d_iso*0.90
        bounds_hi[-1]= d_iso*1.10
    else:
        bounds_lo[-1]= d_iso*0.90
        bounds_hi[-1]= d_iso*1.10
    
    for i in range(n_fib):
        
        R_0[5*i+1]= lam_1
        bounds_lo[5*i+1]= lam_1*0.8
        bounds_hi[5*i+1]= lam_1*1.2
        
        R_0[5*i+2]= lam_2
        bounds_lo[5*i+2]= lam_2*0.8
        bounds_hi[5*i+2]= lam_2*1.2
        
        bounds_lo[5*i+3]= 0
        bounds_hi[5*i+3]= np.pi
        bounds_lo[5*i+4]= 0
        bounds_hi[5*i+4]= 2*np.pi
    
    if n_fib==1:
        R_0[0]= 0.7
        bounds_lo[0]= 0.05
        bounds_hi[0]= 1.0
    elif n_fib==2:
        R_0[0]= 0.4
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.75
        R_0[5]= 0.4
        bounds_lo[5]= 0.05
        bounds_hi[5]= 0.75
    elif n_fib==3:
        R_0[0]= 0.30
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.4
        R_0[5]= 0.30
        bounds_lo[5]= 0.05
        bounds_hi[5]= 0.3
        R_0[10]= 0.15
        bounds_lo[10]= 0.05
        bounds_hi[10]= 0.3
    elif n_fib==4:
        R_0[0]= 0.25
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.3
        R_0[5]= 0.25
        bounds_lo[5]= 0.05
        bounds_hi[5]= 0.3
        R_0[10]= 0.25
        bounds_lo[10]= 0.05
        bounds_hi[10]= 0.3
        R_0[15]= 0.25
        bounds_lo[15]= 0.05
        bounds_hi[15]= 0.3
    elif n_fib==5:
        R_0[0]= 0.20
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.3
        R_0[5]= 0.20
        bounds_lo[5]= 0.05
        bounds_hi[5]= 0.3
        R_0[10]= 0.20
        bounds_lo[10]= 0.05
        bounds_hi[10]= 0.3
        R_0[15]= 0.20
        bounds_lo[15]= 0.05
        bounds_hi[15]= 0.3
        R_0[20]= 0.20
        bounds_lo[20]= 0.05
        bounds_hi[20]= 0.3
    
    return R_0, bounds_lo, bounds_hi




def polar_fibers_from_solution(R_fin, n_fib):
    
    f_fin= np.zeros( (3, n_fib) )
    
    for i in range(n_fib):
        
        f_fin[0,i]= np.sin(R_fin[i*5+3])*np.cos(R_fin[i*5+4]) 
        f_fin[1,i]= np.sin(R_fin[i*5+3])*np.sin(R_fin[i*5+4])
        f_fin[2,i]= np.cos(R_fin[i*5+3])
        f_fin[:,i] /= np.linalg.norm( f_fin[:,i] )
    
    return f_fin



def fibers_2_rho(ft):
    
    rho_t= np.zeros(9)
    
    rho_t[0]= rho_t[3]= 0.0004
    rho_t[6]= 0.003
    
    f1= np.linalg.norm(ft[:,0])
    theta1= np.arccos(ft[2,0]/f1)
    phi1=   np.arctan2(ft[1,0],ft[0,0])
    if phi1<0:
        phi1+= 2*np.pi
    
    rho_t[7]= f1
    rho_t[1]= theta1
    rho_t[2]= phi1
    
    if ft.shape[1]>1:
        
        f2= np.linalg.norm(ft[:,1])
        theta1= np.arccos(ft[2,1]/f2)
        phi1=   np.arctan2(ft[1,1],ft[0,1])
        if phi1<0:
            phi1+= 2*np.pi
        
        rho_t[8]= f2
        rho_t[4]= theta1
        rho_t[5]= phi1
    
    return rho_t






###############################################################################
    
def diamond_simulate(R, n_fib, b, q):
    
    y = (1-R[0:-2:6].sum()) * ( 1 + (R[-2]*b) / R[-1]  )** (-R[-1])
    
    for i in range(n_fib):
        
        y+= R[i*6]* ( 1 + b * ( R[i*6+2] + (R[i*6+1]-R[i*6+2])* 
             ( q[:,0]*np.sin(R[i*6+3])*np.cos(R[i*6+4]) + q[:,1]*np.sin(R[i*6+3])*np.sin(R[i*6+4]) + q[:,2]*np.cos(R[i*6+3]) )**2 ) / R[i*6+5] ) ** (- R[i*6+5] )
    
    return y

def diamond_simulate_log(R, n_fib, b, q):
    
    y = (1-R[0:-2:6].sum()) * ( 1 + (R[-2]*b) / R[-1]  )** (-R[-1])
    
    for i in range(n_fib):
        
        y+= R[i*6]* ( 1 + b * ( np.exp(R[i*6+2]) + (np.exp(R[i*6+1])-np.exp(R[i*6+2]))* 
             ( q[:,0]*np.sin(R[i*6+3])*np.cos(R[i*6+4]) + q[:,1]*np.sin(R[i*6+3])*np.sin(R[i*6+4]) + q[:,2]*np.cos(R[i*6+3]) )**2 ) / R[i*6+5] ) ** (- R[i*6+5] )
    
    return y

def diamond_resid(R, n_fib, b, q, data, weight):
    
    y = (1-R[0:-2:6].sum()) * ( 1 + ( R[-2]*b) / R[-1]  )** (-R[-1])
    
    for i in range(n_fib):
        
        y+= R[i*6]* ( 1 + b * ( R[i*6+2] + (R[i*6+1]-R[i*6+2])* 
             ( q[:,0]*np.sin(R[i*6+3])*np.cos(R[i*6+4]) + q[:,1]*np.sin(R[i*6+3])*np.sin(R[i*6+4]) + q[:,2]*np.cos(R[i*6+3]) )**2 ) / R[i*6+5] ) ** (- R[i*6+5] )
    
    residuals = data - y
    
    residuals= weight * residuals
    
    return residuals

def diamond_resid_bbq(R, n_fib, b, q, data, weight):
    
    y = (1-R[0:-2:6].sum()) * ( 1 + ( R[-2]*b) / R[-1]  )** (-R[-1])
    
    for i in range(n_fib):
        
        y+= R[i*6]* ( 1 + b * ( R[i*6+2] + (R[i*6+1]-R[i*6+2])* 
             ( q[:,0]*np.sin(R[i*6+3])*np.cos(R[i*6+4]) + q[:,1]*np.sin(R[i*6+3])*np.sin(R[i*6+4]) + q[:,2]*np.cos(R[i*6+3]) )**2 ) / R[i*6+5] ) ** (- R[i*6+5] )
    
    residuals = data - y
    
    residuals= np.sum( (weight**2) * (residuals**2) )
    
    return residuals

def diamond_resid_log(R, n_fib, b, q, data, weight):
    
    y = (1-R[0:-2:6].sum()) * ( 1 + ( R[-2]*b) / R[-1] )** (-R[-1])
    
    for i in range(n_fib):
        
        y+= R[i*6]* ( 1 + b * ( np.exp(R[i*6+2]) + (np.exp(R[i*6+1])-np.exp(R[i*6+2]))* 
             ( q[:,0]*np.sin(R[i*6+3])*np.cos(R[i*6+4]) + q[:,1]*np.sin(R[i*6+3])*np.sin(R[i*6+4]) + q[:,2]*np.cos(R[i*6+3]) )**2 ) / R[i*6+5] ) ** (- R[i*6+5] )
    
    residuals = data - y
    
    residuals= weight * residuals
    
    return residuals



def diamond_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003, kappa=100):
    
    R_0=        np.zeros( 6*n_fib+2 )
    bounds_lo=  np.zeros( 6*n_fib+2 )
    bounds_hi=  np.zeros( 6*n_fib+2 )
    
    R_0[-2]= d_iso
    if n_fib>0:
        bounds_lo[-2]= d_iso*0.95
        bounds_hi[-2]= d_iso*1.05
    else:
        bounds_lo[-2]= d_iso*0.1
        bounds_hi[-2]= d_iso*10
    
    R_0[-1]= kappa
    if n_fib>0:
        bounds_lo[-1]= kappa*0.01
        bounds_hi[-1]= kappa*100
    else:
        bounds_lo[-1]= kappa*0.01
        bounds_hi[-1]= kappa*100
    
    for i in range(n_fib):
        
        R_0[6*i+1]= lam_1
        bounds_lo[6*i+1]= lam_1*0.95
        bounds_hi[6*i+1]= lam_1*1.05
        
        R_0[6*i+2]= lam_2
        bounds_lo[6*i+2]= lam_2*0.95
        bounds_hi[6*i+2]= lam_2*1.05
        
        R_0[6*i+3]= np.pi/10 + np.random.rand() * ( np.pi - np.pi/5)
        bounds_lo[6*i+3]= 0
        bounds_hi[6*i+3]= np.pi
        
        R_0[6*i+4]= np.random.rand() * 2 * np.pi
        bounds_lo[6*i+4]= 0
        bounds_hi[6*i+4]= 2*np.pi
        
        R_0[6*i+5]= kappa
        bounds_lo[6*i+5]= kappa*0.01
        bounds_hi[6*i+5]= kappa*100
    
    if n_fib==1:
        R_0[0]= 0.6
        bounds_lo[0]= 0.05
        bounds_hi[0]= 1.0
    elif n_fib==2:
        R_0[0]= 0.5
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.95
        R_0[6]= 0.5
        bounds_lo[6]= 0.05
        bounds_hi[6]= 0.95
    elif n_fib==3:
        R_0[0]= 0.4
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.85
        R_0[6]= 0.3
        bounds_lo[6]= 0.05
        bounds_hi[6]= 0.85
        R_0[12]= 0.3
        bounds_lo[12]= 0.05
        bounds_hi[12]= 0.85
    elif n_fib==4:
        R_0[0]= 0.35
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.6
        R_0[6]= 0.25
        bounds_lo[6]= 0.05
        bounds_hi[6]= 0.6
        R_0[12]= 0.20
        bounds_lo[12]= 0.05
        bounds_hi[12]= 0.6
        R_0[18]= 0.15
        bounds_lo[18]= 0.05
        bounds_hi[18]= 0.4
    elif n_fib==5:
        R_0[0]= 0.25
        bounds_lo[0]= 0.25
        bounds_hi[0]= 0.5
        R_0[6]= 0.20
        bounds_lo[6]= 0.05
        bounds_hi[6]= 0.5
        R_0[12]= 0.20
        bounds_lo[12]= 0.05
        bounds_hi[12]= 0.5
        R_0[18]= 0.15
        bounds_lo[18]= 0.05
        bounds_hi[18]= 0.5
        R_0[24]= 0.15
        bounds_lo[24]= 0.05
        bounds_hi[24]= 0.5
    
    return R_0, bounds_lo, bounds_hi



def diamond_fibers_from_solution(R_fin, n_fib):
    
    f_fin= np.zeros( (3, n_fib) )
    
    for i in range(n_fib):
        
        f_fin[0,i]= np.sin(R_fin[i*6+3])*np.cos(R_fin[i*6+4]) 
        f_fin[1,i]= np.sin(R_fin[i*6+3])*np.sin(R_fin[i*6+4])
        f_fin[2,i]= np.cos(R_fin[i*6+3])
        f_fin[:,i] /= np.linalg.norm( f_fin[:,i] )
    
    return f_fin









def diamond_init_log(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003, kappa=100):
    
    R_0=        np.zeros( 6*n_fib+2 )
    bounds_lo=  np.zeros( 6*n_fib+2 )
    bounds_hi=  np.zeros( 6*n_fib+2 )
    
    R_0[-2]= d_iso
    if n_fib>0:
        bounds_lo[-2]= d_iso*0.8
        bounds_hi[-2]= d_iso*1.2
    else:
        bounds_lo[-2]= d_iso*0.8
        bounds_hi[-2]= d_iso*1.2
    
    R_0[-1]= kappa
    if n_fib>0:
        bounds_lo[-1]= kappa*0.01
        bounds_hi[-1]= kappa*100
    else:
        bounds_lo[-1]= kappa*0.01
        bounds_hi[-1]= kappa*100
    
    for i in range(n_fib):
        
        R_0[6*i+1]= np.log(lam_1)
        bounds_lo[6*i+1]= np.log(lam_1*0.95)
        bounds_hi[6*i+1]= np.log(lam_1*1.05)
        
        R_0[6*i+2]= np.log(lam_2)
        bounds_lo[6*i+2]= np.log(lam_2*0.95)
        bounds_hi[6*i+2]= np.log(lam_2*1.05)
        
        R_0[6*i+3]= np.pi/10 + np.random.rand() * ( np.pi - np.pi/5)
        bounds_lo[6*i+3]= 0
        bounds_hi[6*i+3]= np.pi
        
        R_0[6*i+4]= np.random.rand() * 2 * np.pi
        bounds_lo[6*i+4]= 0
        bounds_hi[6*i+4]= 2*np.pi
        
        R_0[6*i+5]= kappa
        bounds_lo[6*i+5]= kappa*0.01
        bounds_hi[6*i+5]= kappa*100
    
    if n_fib==1:
        R_0[0]= 0.6
        bounds_lo[0]= 0.05
        bounds_hi[0]= 1.0
    elif n_fib==2:
        R_0[0]= 0.5
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.95
        R_0[6]= 0.5
        bounds_lo[6]= 0.05
        bounds_hi[6]= 0.95
    elif n_fib==3:
        R_0[0]= 0.4
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.85
        R_0[6]= 0.3
        bounds_lo[6]= 0.05
        bounds_hi[6]= 0.85
        R_0[12]= 0.3
        bounds_lo[12]= 0.05
        bounds_hi[12]= 0.85
    elif n_fib==4:
        R_0[0]= 0.35
        bounds_lo[0]= 0.05
        bounds_hi[0]= 0.6
        R_0[6]= 0.25
        bounds_lo[6]= 0.05
        bounds_hi[6]= 0.6
        R_0[12]= 0.20
        bounds_lo[12]= 0.05
        bounds_hi[12]= 0.6
        R_0[18]= 0.15
        bounds_lo[18]= 0.05
        bounds_hi[18]= 0.4
    elif n_fib==5:
        R_0[0]= 0.25
        bounds_lo[0]= 0.25
        bounds_hi[0]= 0.5
        R_0[6]= 0.20
        bounds_lo[6]= 0.05
        bounds_hi[6]= 0.5
        R_0[12]= 0.20
        bounds_lo[12]= 0.05
        bounds_hi[12]= 0.5
        R_0[18]= 0.15
        bounds_lo[18]= 0.05
        bounds_hi[18]= 0.5
        R_0[24]= 0.15
        bounds_lo[24]= 0.05
        bounds_hi[24]= 0.5
    
    return R_0, bounds_lo, bounds_hi
















def prony_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003):
    
    R_0=        np.zeros( 5*n_fib+1 )
    
    R_0[-1]= d_iso
    
    for i in range(n_fib):
        
        R_0[5*i+1]= lam_1
        R_0[5*i+2]= lam_2
        R_0[5*i+3]= np.pi/10 + np.random.rand() * ( np.pi - np.pi/5)
        R_0[5*i+4]= np.random.rand() * np.pi
        
    if n_fib==1:
        R_0[0]= 0.7
    elif n_fib==2:
        R_0[0]= 0.5
        R_0[5]= 0.3
    elif n_fib==3:
        R_0[0]= 0.4
        R_0[5]= 0.3
        R_0[10]= 0.2
    
    return R_0

def prony_simulate(R, n_fib, b, q):
    
    y0 = (1-R[0:-1:5].sum()) * np.exp( - R[-1]*b )
    
    i=0
    y1= R[i*5]* np.exp( -b * ( R[i*5+2] + (R[i*5+1]-R[i*5+2])* 
             ( q[:,0]*np.sin(R[i*5+3])*np.cos(R[i*5+4]) + q[:,1]*np.sin(R[i*5+3])*np.sin(R[i*5+4]) + q[:,2]*np.cos(R[i*5+3]) )**2 ) )
    
    i=1
    y2= R[i*5]* np.exp( -b * ( R[i*5+2] + (R[i*5+1]-R[i*5+2])* 
             ( q[:,0]*np.sin(R[i*5+3])*np.cos(R[i*5+4]) + q[:,1]*np.sin(R[i*5+3])*np.sin(R[i*5+4]) + q[:,2]*np.cos(R[i*5+3]) )**2 ) )
    
    return y0+y1+y2, y0, y1, y2


def prony_simulate3(R, n_fib, b, q):
    
    y0 = (1-R[0:-1:5].sum()) * np.exp( - R[-1]*b )
    
    i=0
    y1= R[i*5]* np.exp( -b * ( R[i*5+2] + (R[i*5+1]-R[i*5+2])* 
             ( q[:,0]*np.sin(R[i*5+3])*np.cos(R[i*5+4]) + q[:,1]*np.sin(R[i*5+3])*np.sin(R[i*5+4]) + q[:,2]*np.cos(R[i*5+3]) )**2 ) )
    
    i=1
    y2= R[i*5]* np.exp( -b * ( R[i*5+2] + (R[i*5+1]-R[i*5+2])* 
             ( q[:,0]*np.sin(R[i*5+3])*np.cos(R[i*5+4]) + q[:,1]*np.sin(R[i*5+3])*np.sin(R[i*5+4]) + q[:,2]*np.cos(R[i*5+3]) )**2 ) )
    
    i=2
    y3= R[i*5]* np.exp( -b * ( R[i*5+2] + (R[i*5+1]-R[i*5+2])* 
             ( q[:,0]*np.sin(R[i*5+3])*np.cos(R[i*5+4]) + q[:,1]*np.sin(R[i*5+3])*np.sin(R[i*5+4]) + q[:,2]*np.cos(R[i*5+3]) )**2 ) )
    
    return y0+y1+y2+y3, y0, y1, y2, y3



def prony_init_final(lam_1=0.0019, lam_2=0.0004):
    
    R_0=        np.zeros( 4 )
    bounds_lo=  np.zeros( 4 )
    bounds_hi=  np.zeros( 4 )
    
    R_0[0]= lam_1
    bounds_lo[0]= lam_1*0.8
    bounds_hi[0]= lam_1*1.2
    
    R_0[1]= lam_2
    bounds_lo[1]= lam_2*0.8
    bounds_hi[1]= lam_2*1.2
    
    R_0[2]= np.pi/10 + np.random.rand() * ( np.pi - np.pi/5)
    bounds_lo[2]= 0
    bounds_hi[2]= np.pi
    
    R_0[3]= np.random.rand() * np.pi
    bounds_lo[3]= 0
    bounds_hi[3]= np.pi
    
    return R_0, bounds_lo, bounds_hi


def prony_resid_final(R, q, data):
    
    y = R[1] + ( R[0]-R[1] ) * ( q[:,0]*np.sin(R[2])*np.cos(R[3]) + q[:,1]*np.sin(R[2])*np.sin(R[3]) + q[:,2]*np.cos(R[2]) )**2
    
    residuals = data - y
    
    return residuals


def prony_simulate_final(R, q):
    
    y = R[1] + ( R[0]-R[1] ) * ( q[:,0]*np.sin(R[2])*np.cos(R[3]) + q[:,1]*np.sin(R[2])*np.sin(R[3]) + q[:,2]*np.cos(R[2]) )**2
    
    return y








def prony(t, F, m):
    
    N    = len(t)
    Amat = np.zeros((N-m, m))
    bmat = F[m:N]
    
    for jcol in range(m):
    		Amat[:, jcol] = F[m-jcol-1:N-1-jcol]
    		
    sol = np.linalg.lstsq(Amat, bmat)
    d = sol[0]
    
    c = np.zeros(m+1)
    c[m] = 1.
    for i in range(1,m+1):
        c[m-i] = -d[i-1]
    
    u = poly.polyroots(c)
    b_est = np.log(u)/(t[1] - t[0])
    
    # Set up LLS problem to find the "a"s in step 3
    Amat = np.zeros((N, m))
    bmat = F
    
    for irow in range(N):
        Amat[irow, :] = u**irow
    		
    sol = np.linalg.lstsq(Amat, bmat, rcond=None)
    a_est = sol[0]
    
    return a_est, b_est


def MatPen(t, x, M=3, L= 10):
    
    N = len(x)
#    L = int(N/3)
    dt= t[1]- t[0]
    Y = linalg.hankel(x[:N-L], x[N-L-1:])
#    Y1= Y[:,:-1]
#    Y2= Y[:,1:]
    
#    Y1_dagger= np.linalg.pinv(Y1)
#    Y1Y2= np.matmul( Y1_dagger, Y2 )
#    W, _= linalg.eig(  Y1Y2 )  
    
    U, S, V = linalg.svd(Y, full_matrices=False)
    V=  V.T
    #S= np.diag(S)
    
    V=  V[:,:M]
    #S=  S[:,:M]
    
    V1= V[:-1,:]
    V2= V[1:,:]
    
    V1_dagger= np.linalg.pinv(V1)
    V1V2= np.matmul( V1_dagger, V2 )
    WW, _= linalg.eig(  V1V2 )
#    WW = linalg.eigvals(  V1V2 )
    
    z= np.real(WW[:M])
    
    Z= np.zeros((N,M))
    for i in range(N):
        Z[i,:]= z**(t[i]/dt)
    
    r, _, _, _= np.linalg.lstsq(Z, x)
    
    return r, -np.log(z)/dt






#def MatPen_signal_simulate(a_est, b_est, b_vals):
#    
#    s_est= np.zeros(b_vals.shape)
#    
#    for i in range(len(a_est)):
#        
#        s_est= s_est + a_est[i] * np.exp(-b_est[i]*b_vals)
#        
#    return s_est

def MatPen_signal_simulate(a_est, b_est, b_vals):
    
    s0= a_est[0] * np.exp(-b_est[0]*b_vals)
    s1= a_est[1] * np.exp(-b_est[1]*b_vals)
    s2= a_est[2] * np.exp(-b_est[2]*b_vals)
    
    s_est= s0+s1+s2
    
    return s_est, s0, s1, s2




def MatPenV2(t, x, M=3, L= 10):
    
    N = len(x)
#    L = int(N/3)
    dt= t[1]- t[0]
    Y = linalg.hankel(x[:N-L], x[N-L-1:])
#    Y1= Y[:,:-1]
#    Y2= Y[:,1:]
    
#    Y1_dagger= np.linalg.pinv(Y1)
#    Y1Y2= np.matmul( Y1_dagger, Y2 )
#    W, _= linalg.eig(  Y1Y2 )  
    
    U, S, V = linalg.svd(Y, full_matrices=False)
    V=  V.T
    #S= np.diag(S)
    
    V=  V[:,:M]
    #S=  S[:,:M]
    
    V1= V[:-1,:]
    V2= V[1:,:]
    
    S1S2= np.concatenate( (V2,V1 ) , axis=-1)
    U, S, V = linalg.svd(S1S2, full_matrices=False)
    V=  V.T
    
    V2= V[:M,:]
    V1= V[M:,:]
    
    V1_dagger= np.linalg.pinv(V1)
    V1V2= np.matmul( V1_dagger, V2 )
    WW, _= linalg.eig(  V1V2 )
#    WW = linalg.eigvals(  V1V2 )
    
    z= np.real(WW[:M])
    
    Z= np.zeros((N,M))
    for i in range(N):
        Z[i,:]= z**(t[i]/dt)
    
    r, _, _, _= np.linalg.lstsq(Z, x)
    
    return r, -np.log(z)/dt





def Cadzow(x, M=3, L= 10, n_iter= 1, verbose=False, x_clean=-1):
    
    N = len(x)
    
    if verbose:
        print('starting error norm: ' , np.linalg.norm(x-x_clean))
    
    for i_iter in range(n_iter):
        
        Y = linalg.hankel(x[:N-L], x[N-L-1:])
        
        U, S, V = linalg.svd(Y, full_matrices=False)
        V=  V.T
        
        YY= np.matmul( np.matmul(U[:,:M], np.diag(S[:M])), V[:,:M].T)
        
        xx=   np.zeros(len(x))
        xx_c= np.zeros(len(x))
        
        for i in range(YY.shape[1]):
            xx[i:i+YY.shape[0]]   = xx[i:i+YY.shape[0]] + YY[:,i]
            xx_c[i:i+YY.shape[0]]+= 1
        
        x= xx/xx_c
        
        if verbose:
            print('error norm iteration: ' + str(i_iter) + '  ' , np.linalg.norm(x-x_clean))
    
    return x










def variable_projection_init(n_fib, lam_1=0.0019, lam_2=0.0004, d_iso=0.003):
    
    R_0=        np.zeros( 5*n_fib+1 )
    R_L=        np.zeros( 5*n_fib+1 )
    R_H=        np.zeros( 5*n_fib+1 )
    
    R_0[-1]= d_iso
    R_L[-1]= d_iso*0.5
    R_H[-1]= d_iso*2.0
    
    for i in range(n_fib):
        
        R_0[5*i+1]= lam_1
        R_0[5*i+2]= lam_2
        R_0[5*i+3]= np.pi/10 + np.random.rand() * ( np.pi - np.pi/5)
        R_0[5*i+4]= np.random.rand() * np.pi
        
        R_L[5*i+1]= lam_1*0.5
        R_L[5*i+2]= lam_2*0.5
        R_L[5*i+3]= 0
        R_L[5*i+4]= 0
        
        R_H[5*i+1]= lam_1*2.0
        R_H[5*i+2]= lam_2*2.0
        R_H[5*i+3]= np.pi
        R_H[5*i+4]= np.pi
        
    R_0[0]= 0.5
    R_0[5]= 0.3
    R_L[0]= 0.0
    R_L[5]= 0.0
    R_H[0]= 0.8
    R_H[5]= 0.5
    
    return R_0, R_L, R_H

def variable_projection_resid(R, b, q, data):
    
    y0 = np.exp( - R[-1]*b )
    
    i=0
    y1= np.exp( -b * ( R[i*4+1] + (R[i*4]-R[i*4+1])* 
             ( q[:,0]*np.sin(R[i*4+2])*np.cos(R[i*4+3]) + q[:,1]*np.sin(R[i*4+2])*np.sin(R[i*4+3]) + q[:,2]*np.cos(R[i*4+2]) )**2 ) )
    
    i=1
    y2= np.exp( -b * ( R[i*4+1] + (R[i*4]-R[i*4+1])* 
             ( q[:,0]*np.sin(R[i*4+2])*np.cos(R[i*4+3]) + q[:,1]*np.sin(R[i*4+2])*np.sin(R[i*4+3]) + q[:,2]*np.cos(R[i*4+2]) )**2 ) )
    
    Phi= np.vstack( ( y0, y1, y2 ) ).T
    
    Phi_dagger= np.linalg.pinv(Phi)
    
    resid= np.matmul( ( np.eye(len(b)) - np.matmul( Phi, Phi_dagger ) ), data )
    
    return resid



def variable_projection_Phi_matrix(R, b, q):
    
    y0 = np.exp( - R[-1]*b )
    
    i=0
    y1= np.exp( -b * ( R[i*4+1] + (R[i*4]-R[i*4+1])* 
             ( q[:,0]*np.sin(R[i*4+2])*np.cos(R[i*4+3]) + q[:,1]*np.sin(R[i*4+2])*np.sin(R[i*4+3]) + q[:,2]*np.cos(R[i*4+2]) )**2 ) )
    
    i=1
    y2= np.exp( -b * ( R[i*4+1] + (R[i*4]-R[i*4+1])* 
             ( q[:,0]*np.sin(R[i*4+2])*np.cos(R[i*4+3]) + q[:,1]*np.sin(R[i*4+2])*np.sin(R[i*4+3]) + q[:,2]*np.cos(R[i*4+2]) )**2 ) )
    
    Phi= np.vstack( ( y0, y1, y2 ) ).T
    
    return Phi







def model_selection_bootstrap(s, b_vals, b_vecs, n_bs, m_max= 3, threshold= 0.5, model='DIAMOND', delta_mthod= False):
    
    m_selected= False
    m_opt= -1
    m= 0
    wt_temp= s**2
    
    while not m_selected:
        
        true_numbers= m
        
        if model=='ball_n_sticks':
            R_init, bounds_lo, bounds_up = polar_fibers_and_iso_init(true_numbers, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
        elif model=='DIAMOND':
            #R_init, bounds_lo, bounds_up = diamond_init(true_numbers, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
            R_init, bounds_lo, bounds_up = diamond_init_log(true_numbers, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
        else:
            print('Model type unidentified.')
            return np.nan
        
        ss_bs=       np.zeros(len(b_vals))
        ss_bs_count= np.zeros(len(b_vals))
        
        ss_bs_matrix=       np.zeros( (len(b_vals), n_bs) )
        ss_bs_count_matrix= np.zeros( (len(b_vals), n_bs) )
        
        for i_bs in range(n_bs):
            
            np.random.seed(i_bs)
            
            ind_bs_train= np.random.randint(0,len(b_vals), b_vals.shape)
            ind_bs_test= [i not in ind_bs_train   for i in range(len(b_vals))]
            ind_counts = [np.sum(ind_bs_train==i) for i in range(len(b_vals))]
            
            if model=='ball_n_sticks':
                
                solution = opt.least_squares(polar_fibers_and_iso_resid, R_init,
                                                bounds=(bounds_lo,bounds_up),
                                                args=( true_numbers, b_vals[ind_bs_train], b_vecs[ind_bs_train,:],
                                                s[ind_bs_train],
                                                wt_temp[ind_bs_train]**0.0))
            elif model=='DIAMOND':
                
                '''solution = opt.least_squares(diamond_resid, R_init,
                                                bounds=(bounds_lo,bounds_up),
                                                args=( true_numbers, b_vals[ind_bs_train], b_vecs[ind_bs_train,:],
                                                s[ind_bs_train],
                                                wt_temp[ind_bs_train]**0.0))'''
                
                solution = opt.least_squares(diamond_resid_log, R_init,
                                                bounds=(bounds_lo,bounds_up),
                                                args=( true_numbers, b_vals[ind_bs_train], b_vecs[ind_bs_train,:],
                                                s[ind_bs_train],
                                                wt_temp[ind_bs_train]**0.0))
                
                '''solution = pybobyqa.solve(polar_fibers_and_iso_resid_bbq, R_init,
                                                bounds=(bounds_lo,bounds_up),
                                                args=( true_numbers, b_vals[ind_bs_train], b_vecs[ind_bs_train,:],
                                                s[ind_bs_train],
                                                wt_temp[ind_bs_train]**0.5),
                                                rhobeg= 0.002, scaling_within_bounds= True, seek_global_minimum=False)'''
            
            if model=='ball_n_sticks':
                ss= polar_fibers_and_iso_simulate(solution.x, true_numbers, b_vals, b_vecs)
            elif model=='DIAMOND':
                #ss= diamond_simulate(solution.x, true_numbers, b_vals, b_vecs)
                ss= diamond_simulate_log(solution.x, true_numbers, b_vals, b_vecs)
            
            ss_bs[ind_bs_test]+= ( ss[ind_bs_test]-s[ind_bs_test] )**2
            ss_bs_count[ind_bs_test]+= 1
            
            ss_bs_matrix[ind_bs_test,i_bs]= ( ss[ind_bs_test]-s[ind_bs_test] )**2
            ss_bs_count_matrix[:,i_bs]= ind_counts
            
        ss_bs[ss_bs_count>0]= ss_bs[ss_bs_count>0]/ ss_bs_count[ss_bs_count>0]
        #ss_bs= ss_bs[ss_bs_count>0]
        E_bs= ss_bs.mean()
        
        if model=='ball_n_sticks':
            solution = opt.least_squares(polar_fibers_and_iso_resid, R_init,
                                                bounds=(bounds_lo,bounds_up),
                                                args=( true_numbers, b_vals, b_vecs,
                                                s,
                                                wt_temp**0.0))
            ss= polar_fibers_and_iso_simulate(solution.x, true_numbers, b_vals, b_vecs)
        elif model=='DIAMOND':
            '''solution = opt.least_squares(diamond_resid, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( true_numbers, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
            ss= diamond_simulate(solution.x, true_numbers, b_vals, b_vecs)'''
            solution = opt.least_squares(diamond_resid_log, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( true_numbers, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
            '''solution = pybobyqa.solve(polar_fibers_and_iso_resid_bbq, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( true_numbers, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.5),
                                    rhobeg= 0.002, scaling_within_bounds= True, seek_global_minimum=False)'''

            ss= diamond_simulate_log(solution.x, true_numbers, b_vals, b_vecs)
        
        ss_fit= ( ss-s )**2
        E_fit= ss_fit.mean()
        
        E_632= 0.368*E_fit + 0.632*E_bs
        
        if m==0:
            
            E_632_m_n_1= E_632
            #E_fit_m_n_1= E_fit
            E_bs_m_n_1 = E_bs
            ss_bs_m_n_1= ss_bs.copy()
            ss_bs_matrix_m_n_1= ss_bs_matrix.copy()
            #ss_bs_count_matrix_m_n_1= ss_bs_count_matrix.copy()
            
        else:
            
            del_632= E_632_m_n_1- E_632
            #del_fit= E_fit_m_n_1- E_fit
            del_bs = E_bs_m_n_1 - E_bs
            
            del_bs_ss= ss_bs_m_n_1- ss_bs
            
            if delta_mthod:
                
                del_ss_bs_matrix= ss_bs_matrix_m_n_1- ss_bs_matrix
                
                q_hat= np.sum(del_ss_bs_matrix, axis=0)
                
                N_hat= np.mean( ss_bs_count_matrix, axis=1 )
                
                numer= ss_bs_count_matrix - N_hat[:,np.newaxis]
                
                numer= np.matmul(numer, q_hat)
                
                denom= np.sum( ss_bs_count_matrix==0, axis=1 )
                
                D= (2+ 1/(len(b_vals)-1)) * ( del_bs_ss - del_bs_ss.mean() )/ len(b_vals) + (denom>0) * numer/ (denom+1e-7)
                
                SE_BS= np.linalg.norm(D)
                
            else:
                
                SE_BS= np.sqrt( np.sum( ( del_bs_ss - del_bs_ss.mean() )**2 ) / ( len(b_vals)**2 ) )
            
            SE_632= del_632/del_bs * SE_BS
            
            if del_632 < threshold*SE_632 or m==m_max:
                
                m_opt= m-1
                m_selected= True
                
            else:
                
                E_632_m_n_1= E_632
                #E_fit_m_n_1= E_fit
                E_bs_m_n_1 = E_bs
                ss_bs_m_n_1= ss_bs.copy()
                ss_bs_matrix_m_n_1= ss_bs_matrix.copy()
                #ss_bs_count_matrix_m_n_1= ss_bs_count_matrix.copy()
        
        m+= 1
    
    
    return m_opt






def model_selection_f_test(s, b_vals, b_vecs, m_max= 3, threshold= 20, condition_mode= 'F_val', model='DIAMOND'):
    
    m_selected= False
    m_opt= -1
    m= 0
    wt_temp= s**2
    
    while not m_selected:
        
        true_numbers= m
        if model=='ball_n_sticks':
            R_init, bounds_lo, bounds_up = polar_fibers_and_iso_init(true_numbers, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
        elif model=='DIAMOND':
            #R_init, bounds_lo, bounds_up = diamond_init(true_numbers, lam_1=0.0019, lam_2=0.0004, d_iso=0.003)
            R_init, bounds_lo, bounds_up = diamond_init_log(true_numbers, lam_1=0.0020, lam_2=0.0004, d_iso=0.003)
        else:
            print('Model type unidentified.')
            return np.nan
        
        if model=='ball_n_sticks':
            solution = opt.least_squares(polar_fibers_and_iso_resid, R_init,
                                                bounds=(bounds_lo,bounds_up),
                                                args=( true_numbers, b_vals, b_vecs,
                                                s,
                                                wt_temp**0.0))
            ss= polar_fibers_and_iso_simulate(solution.x, true_numbers, b_vals, b_vecs)
        elif model=='DIAMOND':
            '''solution = opt.least_squares(diamond_resid, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( true_numbers, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
            ss= diamond_simulate(solution.x, true_numbers, b_vals, b_vecs)'''
            solution = opt.least_squares(diamond_resid_log, R_init,
                                    bounds=(bounds_lo,bounds_up),
                                    args=( true_numbers, b_vals, b_vecs,
                                    s,
                                    wt_temp**0.0))
            ss= diamond_simulate_log(solution.x, true_numbers, b_vals, b_vecs)
            
        ss_fit= ( ss-s )**2
        
        if m==0:
            
            RSS1= ss_fit.sum()
            p1= len(R_init)
            
        else:
            
            RSS2= ss_fit.sum()
            p2= len(R_init)
            
            f_val= ( (RSS1-RSS2) / (p2-p1) )   /   ( RSS2 / ( len(b_vals)-p2 ) )
            
            if condition_mode== 'F_prob':
                f_stat= f.cdf( f_val , p2-p1, len(b_vals)-p2 )
                cond= f_stat< 1-threshold or m==m_max
            elif condition_mode== 'F_val':
                cond= f_val< threshold or m==m_max
            else:
                print('Condition mode unidentified!')
                return np.nan
            
            if cond:
                
                m_opt= m-1
                m_selected= True
                
            else:
                
                RSS1= ss_fit.sum()
                p1= len(R_init)
                    
        
        m+= 1
        
    
    return m_opt
                











def model_selection_f_test_shore(s, sph, b_vals, b_vecs, m_max= 8, threshold= 20, condition_mode= 'F_val'):
    
    v = sph.vertices
    
    m_selected= False
    m_opt= -1
    m= 0
    
    while not m_selected:
        
        L= m
        Zeta= 700
        b0_thresh= 50
        
        SHORE_Phi= shore_matrix(b_vals, b0_thresh, b_vecs, L=L, Zeta=Zeta)
        
        SHORE_Psi= shore_matrix_odf(v, L=L, Zeta=Zeta)
        
        f_n_0, _ = shore_odf(s, SHORE_Phi, SHORE_Psi, L=L, Zeta=Zeta)
        
        ss= np.matmul(SHORE_Phi, f_n_0)
            
        ss_fit= ( ss-s )**2
        
        if m==0:
            
            RSS1= ss_fit.sum()
            p1= SHORE_Phi.shape[-1]
            
        else:
            
            RSS2= ss_fit.sum()
            p2= SHORE_Phi.shape[-1]
            
            f_val= ( (RSS1-RSS2) / (p2-p1) )   /   ( RSS2 / ( len(b_vals)-p2 ) )
            
            if condition_mode== 'F_prob':
                f_stat= f.cdf( f_val , p2-p1, len(b_vals)-p2 )
                cond= f_stat< 1-threshold or m==m_max
            elif condition_mode== 'F_val':
                cond= f_val< threshold or m==m_max
            else:
                print('Condition mode unidentified!')
                return np.nan
            
            if cond:
                
                m_opt= m-2
                m_selected= True
                
            else:
                
                RSS1= ss_fit.sum()
                p1= SHORE_Phi.shape[-1]
                
        m+= 2
    
    SHORE_Phi= shore_matrix(b_vals, b0_thresh, b_vecs, L=m_opt, Zeta=Zeta)
    
    SHORE_Psi= shore_matrix_odf(v, L=m_opt, Zeta=Zeta)
    
    _, f_n = shore_odf(s, SHORE_Phi, SHORE_Psi, L=m_opt, Zeta=Zeta)
    
    fibers , responses = find_dominant_fibers_dipy_way(sph, f_n[:len(v)], min_angle= 30, n_fib=None, \
                                                                                       peak_thr=None, optimize=False, Psi= SHORE_Psi, opt_tol=1e-7)
    
    m_opt_2= len(responses)
    
    return m_opt, m_opt_2
                



















def spherical_harmonics(m, n, theta, phi):
    x = np.cos(phi)
    val = lpmv(m, n, x).astype(complex)
    val *= np.sqrt((2 * n + 1) / 4.0 / np.pi)
    val *= np.exp(0.5 * (gammaln(n - m + 1) - gammaln(n + m + 1)))
    val = val * np.exp(1j * m * theta)
    return val

def real_sph_harm(m, n, theta, phi):
    
    sh = spherical_harmonics(np.abs(m), n, phi, theta)
    
    real_sh = np.where(m > 0, sh.imag, sh.real)
    real_sh *= np.where(m == 0, 1., np.sqrt(2))
    
    return real_sh

def shore_matrix(b_vals, b0_thresh, b_vecs, L=6, Zeta=700):
    
    b_vals[b_vals<b0_thresh]= 0
    b_vecs_scaled = np.sqrt(b_vals[:,np.newaxis]) * b_vecs
    
    r= np.linalg.norm(b_vecs_scaled, axis=1)
    phi = np.arctan2(b_vecs_scaled[:,1], b_vecs_scaled[:,0])
    theta = np.arccos(b_vecs_scaled[:,2]/(r+1e-17))
    theta[np.logical_or( np.isnan(theta), r==0 )]= 0
    
    n_c = int(np.round(1 / 6.0 * (L / 2 + 1) * (L / 2 + 2) * (2 * L + 3)))
    
    M = np.zeros((len(b_vals), n_c))
    ind = -1
    
    for l in range(0, L + 1, 2):
        for n in range(l, int((L + l) / 2) + 1):
            for m in range(-l, l + 1):
                ind+= 1
                M[:, ind] = real_sph_harm(m, l, theta, phi) * \
                    genlaguerre(n - l, l + 0.5)(r ** 2 / Zeta) * \
                    np.exp(- r ** 2 / (2.0 * Zeta)) * \
                    np.sqrt((2 * factorial(n - l)) / (Zeta ** 1.5 * gamma(n + 1.5))) * \
                    (r ** 2 / Zeta) ** (l / 2)
    
    return M


def shore_matrix_odf(v, L=6, Zeta=700):
    
    r= np.linalg.norm(v, axis=1)
    phi = np.arctan2(v[:,1], v[:,0])
    theta = np.arccos(v[:,2]/(r+1e-17))
    theta[np.logical_or( np.isnan(theta), r==0 )]= 0
    
    n_c = int(np.round(1 / 6.0 * (L/2 + 1) * (L/2 + 2) * (2 * L + 3)))
    
    upsilon = np.zeros((len(v), n_c))
    ind= -1
    
    for l in range(0, L + 1, 2):
        for n in range(l, int((L + l) / 2) + 1):
            for m in range(-l, l + 1):
                ind+= 1
                upsilon[:, ind] = (-1) ** (n - l / 2.0) * \
                    np.sqrt((gamma(l / 2.0 + 1.5) ** 2 *
                    gamma(n + 1.5) * 2 ** (l + 3)) /
                   (16 * np.pi ** 3 * (Zeta) ** 1.5 * factorial(n - l) *
                    gamma(l + 1.5) ** 2)) * \
                    hyp2f1(l - n, l / 2.0 + 1.5, l + 1.5, 2.0) * \
                    real_sph_harm(m, l, theta, phi)
    
    return upsilon

def shore_odf(s, SHORE_Phi, SHORE_Psi, L=6, Zeta=700):
    
    PhiInv =   np.dot( np.linalg.inv(np.dot(SHORE_Phi.T, SHORE_Phi) ), SHORE_Phi.T)
    
    f_n_0 = np.dot(PhiInv, s)
                
    signal_0 = 0
    
    for n in range(int(L / 2) + 1):
        signal_0 += (
            f_n_0[n] * (genlaguerre(n, 0.5)(0) * (
                (factorial(n)) /
                (2 * np.pi * (Zeta ** 1.5) * gamma(n + 1.5))
            ) ** 0.5)
        )
    
    f_n_0 = f_n_0 / signal_0
    
    f_n = np.dot(SHORE_Psi, f_n_0)
    
    return f_n_0, f_n






def MCMC_prob(R, sigma, b_vals, b_vecs, s, n_fib, sample_stride):
    
    '''prob= 1
    
    for i in range(n_fib):
        
        prob*= -1/( (1-R[5*i]+0.001) * np.log(1-R[5*i]+0.001) )
        
    for i in range(n_fib):
        
        prob*= np.sin( R[5*i+3] )
    
    prob*= 1/ sigma
    
    s_prd= polar_fibers_and_iso_simulate(R, n_fib, b_vals, b_vecs)
    
    prob*= np.prod( np.exp( - (s_prd[::sample_stride]-s[::sample_stride])**2/sigma**2 ) )'''
    
    prob1= 0
    
    for i in range(n_fib):
        
        prob1+= np.log( -1/( (1-(R[5*i]+0.001)) * np.log(1-(R[5*i]+0.001)) ) )
        
    for i in range(n_fib):
        
        prob1+= np.log( np.sin( R[5*i+3] ) )
    
    prob1+= np.log( 1/ sigma )
    
    s_prd= polar_fibers_and_iso_simulate(R, n_fib, b_vals, b_vecs)
    
    prob2= np.sum( - (s_prd[::sample_stride]-s[::sample_stride])**2/sigma**2 )
    
    prob= prob1 + prob2
    
    
    return prob, prob1, prob2



def MCMC(R_inter, b_vals, b_vecs, s, n_fib, sigma, step_f= 0.02, step_lam= 0.0001, step_ang= 0.2, step_sigma= 0.001, 
         aneal_f= 1.2, aneal_lam= 1.2, aneal_ang= 1.2, aneal_sigma= 1.2, N_mcmc=10000, N_aneal= 1000, sample_stride=50):
    
    R_old=     R_inter.copy()
    sigma_old= sigma
    
    prob_track= np.zeros((N_mcmc,4))
    prob_count= -1
    
    for i_mcmc in range(N_mcmc):
        
        prob_old, _, _ = MCMC_prob(R_old, sigma_old, b_vals, b_vecs, s, n_fib, sample_stride)
        
        R_c=     R_old.copy()
        sigma_c= sigma_old
        
        
        for i in range(n_fib):
            
            R_c[5*i+3]+= np.random.randn()*step_ang
            if R_c[5*i+3]>np.pi:
                R_c[5*i+3]= np.pi
            elif R_c[5*i+3]<0.0:
                R_c[5*i+3]= 0.0
            
            R_c[5*i+4]+= np.random.randn()*step_ang
            if R_c[5*i+4]>2*np.pi:
                R_c[5*i+4]= 2*np.pi
            elif R_c[5*i+4]<0.0:
                R_c[5*i+4]= 0.0
        
        if compute_min_angle_between_fibers(R_c, n_fib)<30:
            
            R_c=     R_old.copy()
        
        good_steps= False
        steps_tried= 0
        
        while not good_steps:
            
            for i in range(n_fib):
                
#                R_c[5*i]= R_old[5*i] + np.random.randn()*step_f
                R_c[5*i]+= np.random.randn()*step_f
                
            if np.all(R_c[0:-1:5]>0) and np.all(R_c[0:-1:5]<0.998) and np.sum(R_c[0:-1:5])<=1:
                
                good_steps= True
            
            steps_tried+= 1
            
            if steps_tried>100:
                
                good_steps= True
                
                R_c[0:-1:5]= R_old[0:-1:5].copy()
                
                #print('steps failed')
                
        
        sigma_c+= np.random.randn()*step_sigma
        
        for i in range(n_fib):
            
            R_c[5*i+1]+= np.random.randn()*step_lam
            if R_c[5*i+1]>0.003:
                R_c[5*i+1]= 0.003
            elif R_c[5*i+1]<0.001:
                R_c[5*i+1]= 0.001
                
            R_c[5*i+2]+= np.random.randn()*step_lam
            if R_c[5*i+2]>0.001:
                R_c[5*i+2]= 0.001
            elif R_c[5*i+2]<0.0002:
                R_c[5*i+2]= 0.0002
        
        R_c[-1]+= np.random.randn()*step_lam
        if R_c[-1]>0.0035:
            R_c[-1]= 0.0035
        elif R_c[-1]<0.0025:
            R_c[-1]= 0.0025
        
        prob_c, prob1, prob2= MCMC_prob(R_c, sigma_c, b_vals, b_vecs, s, n_fib, sample_stride)
        
        alpha= prob_c/prob_old
        
        r= min(alpha, 1)
        
        u= np.random.rand()
        
        if prob_c>prob_old:
#        if alpha>1:
#        if u<r:
            R_old=     R_c.copy()
            sigma_old= sigma_c
            prob_count+= 1
            prob_track[prob_count,:]= i_mcmc, prob_c, prob1, prob2
            #print('updated ', prob_c, prob_old)
            
        
        if i_mcmc>0 and i_mcmc%N_aneal==0:
            
            step_f/= aneal_f
            step_lam/= aneal_lam
            step_ang/= aneal_ang
            step_sigma/= aneal_sigma
    
    '''print('number of updates:  ', prob_count+1)
    prob_track= prob_track[:prob_count,:]
    plt.figure(), 
    plt.plot(prob_track[:,0], prob_track[:,1], '.b')
    plt.plot(prob_track[:,0], prob_track[:,2], '.r')
    plt.plot(prob_track[:,0], prob_track[:,3], '.k')'''
    
    return R_old, prob_track
    
    














step_f= 0.02
step_lam= 0.00001
step_ang= 0.2
#step_sigma= 0.001*sigma
aneal_f= 1.2
aneal_lam= 1.2
aneal_ang= 1.2
aneal_sigma= 1.2
N_mcmc=20000
N_aneal= 3000
sample_stride=10









#def compute_angular_feature_vector(sig_orig, b_vecs, dir_orig, N= 20):
#    
#    s= sig_orig.copy()
#    d= dir_orig.copy()
#    d/= np.linalg.norm(d)
#    #Theta= np.zeros(len(s))
#    
#    f_vec= np.zeros(N)
#    c_vec= np.zeros(N, np.int)
#    f_mat= np.zeros((N,len(s)))
#    
#    f_vec_full= np.zeros(len(s))
#    a_vec_full= np.zeros(len(s))
#
#    for i in range(len(s)):
#        
#        theta= np.arccos( np.clip( np.abs( np.dot( d, b_vecs[:,i] ) ), 0, 1) )
#        ind=  min( int( np.floor( (theta-0) / ( np.pi/2 ) * N  ) ), N-1)
#        
#        f_vec[ind]+= s[i]
#        f_mat[ind,c_vec[ind]]= s[i]
#        c_vec[ind]+= 1
#        #Theta[i]=    theta
#    
#        a_vec_full[i]= theta
#        f_vec_full[i]= s[i]
#
#    f_avg= f_vec / ( c_vec + 1e-7)
#    f_std= np.zeros(N)
#    
#    for i in range(N):
#        f_avg[i]= f_mat[i,:c_vec[i]].mean()
#    
#    return f_vec, c_vec, f_avg, f_std, f_mat, a_vec_full, f_vec_full
    








def compute_angular_feature_vector_cont(sig_orig, b_vecs, dir_orig, N= 10, M= 20, full_circ= False):
    
    s= sig_orig.copy()
    d= dir_orig.copy()
    d/= np.linalg.norm(d)
    
    f_vec= np.zeros((N+1,2))
    
    if full_circ:
        theta= np.arccos( np.clip( np.dot( d, b_vecs ) , -1, 1) )
    else:
        theta= np.arccos( np.clip( np.abs( np.dot( d, b_vecs ) ), 0, 1) )
        
    for i in range(N+1):
        
        if full_circ:
            theta_i= i/N * np.pi
        else:
            theta_i= i/N * np.pi/2
            
        diff= 1/ ( np.abs(theta- theta_i) + 0.2 )
        
        if M>0:
            arg= np.argsort(diff)[-M:]
            mean= np.sum( diff[arg]*s[arg] )/ diff[arg].sum()
            var = np.sum( diff[arg]*((s[arg]-mean)**2) )/ diff[arg].sum()
        else:
            mean= np.sum( diff*s )/ diff.sum()
            var = np.sum( diff*((s-mean)**2) )/ diff.sum()
        
        f_vec[i,:]= mean, var
        
    return f_vec








def compute_angular_feature_vector_cont_full_sphere(sig_orig, b_vecs, v, N= 10, M= 20, full_circ= False):
    
    assert( len(sig_orig)==b_vecs.shape[1] )
    
    s= sig_orig.copy()
    
    f_vec= np.zeros((v.shape[0], 2*N+2))
    
    if full_circ:
        theta= np.arccos( np.clip( np.dot( v, b_vecs ) , -1, 1) )
    else:
        theta= np.arccos( np.clip( np.abs( np.dot( v, b_vecs ) ), 0, 1) )
        
    for i in range(N+1):
        
        if full_circ:
            theta_i= i/N * np.pi
        else:
            theta_i= i/N * np.pi/2
            
        diff= 1/ ( np.abs(theta- theta_i) + 0.2 )
        arg1= np.argsort(diff)[:,-M:]
        
        arg2= [ np.arange(len(v))[:,np.newaxis],arg1 ]
        
        mean= np.sum( diff[arg2]*s[arg1] , axis=1)/ np.sum( diff[arg2], axis=1 )
        var = np.sum( diff[arg2]*((s[arg1]-mean[:,np.newaxis])**2) , axis=1 )/ np.sum( diff[arg2], axis=1 )
        
        f_vec[:, i]=     mean
        f_vec[:, i+N+1]= var
        
    return f_vec






def compute_angular_feature_vector_cont_full_sphere_fast(sig_orig, b_vecs, v, N= 10, M= 20, full_circ= False):
    
    assert( len(sig_orig)==b_vecs.shape[1] )
    
    s= sig_orig.copy()
    
    if full_circ:
        theta= np.arccos( np.clip( np.dot( v, b_vecs ) , -1, 1) )
    else:
        theta= np.arccos( np.clip( np.abs( np.dot( v, b_vecs ) ), 0, 1) )
        
    if full_circ:
        theta_i= np.arange(N+1)/N * np.pi
    else:
        theta_i= np.arange(N+1)/N * np.pi/2
    
    theta_i_m= np.tile(theta_i, [len(v),b_vecs.shape[1],1])
    theta_m= np.tile(theta[:,:,np.newaxis], [1,1,len(theta_i)])
    
    diff= 1/ ( np.abs(theta_m- theta_i_m) + 0.2 )
    diff= np.transpose(diff,[0,2,1])
    Arg1= np.argsort(diff)[:,:,-M:]
    
    Arg2= [ np.arange(len(v))[:,np.newaxis,np.newaxis], np.arange(len(theta_i))[np.newaxis,:,np.newaxis], Arg1]
    
    mean= np.sum( diff[Arg2]*s[Arg1] , axis=-1)/ np.sum( diff[Arg2], axis=-1 )
    var = np.sum( diff[Arg2]*((s[Arg1]-mean[:,:,np.newaxis])**2) , axis=-1 )/ np.sum( diff[Arg2], axis=-1 )
    
    f_vec= np.hstack( (mean, var) )
    
    return f_vec




def compute_angular_feature_vector_cont_full_sphere_faster(sig_orig, b_vecs, v, N= 10, M= 20, full_circ= False):
    
    s= sig_orig.copy()
    
    if full_circ:
        theta= np.arccos( np.clip( np.dot( v, b_vecs ) , -1, 1) )
    else:
        theta= np.arccos( np.clip( np.abs( np.dot( v, b_vecs ) ), 0, 1) )
        
    if full_circ:
        theta_i= np.arange(N+1)/N * np.pi
    else:
        theta_i= np.arange(N+1)/N * np.pi/2
    
    theta_i_m= np.tile(theta_i, [len(v),b_vecs.shape[1],1])
    theta_m= np.tile(theta[:,:,np.newaxis], [1,1,len(theta_i)])
    
    diff= 1/ ( np.abs(theta_m- theta_i_m) + 0.2 )
    diff= np.transpose(diff,[0,2,1])
    Arg1= np.argsort(diff)[:,:,-M:]
    Arg11= [ np.arange(s.shape[0])[:,np.newaxis,np.newaxis,np.newaxis] , Arg1]
    
    Arg2= [ np.arange(len(v))[:,np.newaxis,np.newaxis], np.arange(len(theta_i))[np.newaxis,:,np.newaxis], Arg1]
    
    mean= np.sum( diff[Arg2]*s[Arg11] , axis=-1)/ np.sum( diff[Arg2], axis=-1 )
    var = np.sum( diff[Arg2]*((s[Arg11]-mean[:,:,:,np.newaxis])**2) , axis=-1 )/ np.sum( diff[Arg2], axis=-1 )
    
    f_vec= np.concatenate((mean, var), axis=-1)
    
    return f_vec














def random_point_on_sphere(n_fib):
    
    fib = np.random.randn(3, n_fib)
    
    fib /= np.linalg.norm(fib, axis=0)
    
    return fib




def random_points_on_sphere(n_fib, min_separation= 45, n_try_angle= 100):
    
    good_separation= False
    
    i_try_angle= 0
    
    while not good_separation:
        
        fib_trial= random_point_on_sphere(n_fib)
        
        i_try_angle+= 1
        separation_temp= compute_min_angle_between_fiberset(fib_trial)
        
        if separation_temp>min_separation:
            good_separation= True
            
        if i_try_angle==n_try_angle:
            good_separation= True
            print('Fibers are not well-separated!')
    
    return fib_trial






