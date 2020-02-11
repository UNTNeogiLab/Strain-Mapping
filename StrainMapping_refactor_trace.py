#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 21:16:56 2020

@author: tracebivens
"""

import numpy as np
import glob
from scipy import optimize
import itertools
from  joblib import Parallel, delayed
import multiprocessing as mP
import time
import os
from numba import jit, prange
import dask as da
nCores = mP.cpu_count()

directory =os.getcwd()

#%%
#@jit(cache=True)
def LoadFiles(directory, sample_step):
    filelist = glob.glob(str(directory) + '/halfwave*.npy')
    filelist = sorted(filelist)
    data = np.array([np.load(fname) for fname in filelist])
    N = sample_step
    data = data[:,0:2039:N,0:2039:N]
    threshold = np.mean(data)*1.2
    phi = np.arange(0, 2*data.shape[0], 2) * np.pi/180
    return data, threshold, phi
#%%
@jit(cache=True)
def StrainModel(phi, a, b, delta, theta):
    return 0.25*(a * np.cos(3*phi - 3*delta) + b * np.cos(2 * theta + phi -
                 3 * delta))**2
#%%
@jit(cache=True)
def Normalize(zstack):
    norm = (zstack - zstack.min()) / (np.ptp(zstack))
    return norm

#%%
#@jit(cache=True, parallel=True)
def StrainFit(data, threshold, phi):
    paramsarray = np.zeros((data.shape[1], data.shape[2], 4))
    params_cov_array = np.zeros((data.shape[1], data.shape[2], 4))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if data[:,i,j].mean() > threshold:
                data[:,i,j] = Normalize(data[:,i,j])
                params, pcov = optimize.curve_fit(StrainModel, phi, data[:,i,j],
                                                  maxfev = 1000000,
                                                  p0 = [1,1,1,1],
                                                  method='lm',
                                                  xtol = 1e-9,
                                                  ftol = 1e-9)
                paramsarray[i,j,:] = params
                params_cov_array[i,j,:] = pcov.diagonal()
            else:
                pass
            
    return paramsarray, params_cov_array
#%%
            
filelist = glob.glob(str(directory) + '/halfwave*.npy')
filelist = sorted(filelist)
x = np.array([np.load(fname) for fname in filelist])
N = 10
x = x[:,0:2039:N,0:2039:N]    #selects every Nth stack for debugging
print("Files uploaded")

threshold = np.average(x)*1.2
phi = np.arange(0, 2*x.shape[0], 2) * np.pi/180         
            
#%%
@da.delayed
def fitter(data):
    params, pcov = optimize.curve_fit(StrainModel, phi, data[:,1000,1300],
                                                  maxfev = 1000000,
                                                  p0 = [1,1,1,1],
                                                  method='lm',
                                                  xtol = 1e-9,
                                                  ftol = 1e-9)
    return params, pcov

#%%
def f(v):
    i = v[0]
    j = v[1]
    t = [v[0],v[1], 0, 0, 0, 0 ]
    q = [v[0],v[1], 0, 0, 0, 0 ]
    if x[:,i,j].mean() > threshold:
        xx = Normalize(x[:,i,j])
        params, params_covariance = optimize.curve_fit(StrainModel, phi, xx,
                                                       maxfev = 1000000,
                                                       p0 = [1,1,1,1],
                                                       method='lm',
                                                       xtol = 1e-9,
                                                       ftol = 1e-9)
        for k in range(4):
            t[k+2] = params[k]
            q[k+2] = params_covariance.diagonal()[k]
       # print(str(i) + ',' + str(j))
    return t, q

tic = time.time()
print('Start time=' + str(tic))

print('Parallelizing')
paramsarray = np.zeros((x.shape[1], x.shape[2], 4))
params_cov_array = np.zeros((x.shape[1], x.shape[2], 4))
V = itertools.product(range(x.shape[1]), range(x.shape[2]) )
A = Parallel(n_jobs=nCores)( delayed(f)(v) for v in V )
print('Collecting')
for a in A:
    for k in range(4):
        paramsarray[a[0],a[1],k] = a[k+2]
        params_cov_array[a[0],a[1],k] = a[k+2]
print('Saving')
np.save(directory + '/outfile', paramsarray)
#np.save('outfile', paramsarray)
outfile = np.load(directory + '/outfile.npy')
toc = time.time()
print('Run Time =' + str(toc - tic) + ' seconds')



