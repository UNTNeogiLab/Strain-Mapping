#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import glob
from scipy import optimize
import itertools
from  joblib import Parallel, delayed
import multiprocessing as mP
import time
import os

nCores = mP.cpu_count()

directory =os.getcwd()

#%% Import Files

#If outfile exists, delete it

filelist = glob.glob(str(directory) + '/halfwave*.npy')
filelist = sorted(filelist)
x = np.array([np.load(fname) for fname in filelist])

print("Files uploaded")


#%%
tic = time.time()
def strainfit(phi, a, b, delta, theta):
    return 0.25*(a * np.cos(3*phi- 3*delta) + b * np.cos(2 * theta + phi -
                 3*delta))**2
N = 1
print('Start time=' + str(tic))
threshold = np.average(x)*1.2
phi = np.arange(0,2*x.shape[0],2)*np.pi/180
def f(v):
    i = v[0]*N
    j = v[1]*N
    t = [v[0],v[1], 0, 0, 0, 0 ]
    if x[:,i,j].mean() > threshold:
        xx = (x[:,i,j]-np.amin(x[:,i,j]))/np.amax(x[:,i,j]-np.amin(x[:,i,j]))
        params, params_covariance = optimize.curve_fit(strainfit, phi, xx,
                                                       maxfev = 1000000,
                                                       p0 = [1,1,1,1],
                                                       method='lm',
                                                       xtol = 1e-9,
                                                       ftol = 1e-9)
        for k in range(4):
            t[k+2] = params[k]
        print(str(i) + ',' + str(j))
    return t

print('Parallelizing')
paramsarray = np.zeros((x.shape[1]//N, x.shape[2]//N, 4))
V = itertools.product(range(x.shape[1]//N), range(x.shape[2]//N) )
A = Parallel(n_jobs=nCores)( delayed(f)(v) for v in V )
print('Collecting')
for a in A:
    for k in range(4):
        paramsarray[a[0],a[1],k] = a[k+2]
print('Saving')
np.save(directory + '/outfile', paramsarray)
#np.save('outfile', paramsarray)
outfile = np.load(directory + '/outfile.npy')
toc = time.time()
print('Run Time =' + str(toc - tic) + ' seconds')
#%%
import numpy as np
import os
import matplotlib.pyplot as plt
directory =os.getcwd()
outfile = np.load(directory + '/outfile.npy')
def pplot(paramsarray):
    f , ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,
        sharex=True,sharey=True,dpi=200)
    A = paramsarray[:,:,0]
    B = paramsarray[:,:,1]
    Delta = np.mod(paramsarray[:,:,2]*60/(np.pi),60)
    Theta = np.mod(paramsarray[:,:,3]*60/(np.pi),60)
    
    
    #constants from Mennel et al. 2018 Nat. Comm.
    chi = 4.5
    nu = 0.29
    p1 = -0.68
    p2 = -2.35

    exx = 0.5*((A-2*chi)/((1-nu)*(p1+p2)) + B/((1+nu)*(p1-p2)))
    eyy = 0.5*((A-2*chi)/((1-nu)*(p1+p2)) - B/((1+nu)*(p1-p2)))


    f1 = ax1.imshow(A, cmap='jet')
    f2 = ax2.imshow(B, cmap='jet')
    f3 = ax3.imshow(Delta,cmap='jet')
    f4 = ax4.imshow(Theta,cmap='jet')
    f5 = ax5.imshow(exx,cmap='jet')
    f6 = ax6.imshow(eyy,cmap='jet')
    
    fs = 10
    
    ax1.set_title('A', fontsize= fs)
    f.colorbar(f1, ax=ax1)
    ax2.set_title('B', fontsize= fs)
    f.colorbar(f2, ax=ax2)
    ax3.set_title(r'$\delta (Orientation)$', fontsize= fs)
    f.colorbar(f3, ax=ax3)
    ax4.set_title(r'$\theta (Strain angle)$', fontsize= fs)
    f.colorbar(f4, ax=ax4)
    ax5.set_title(r'$e_x$$_x $', fontsize= fs)
    f.colorbar(f5, ax=ax5)
    ax6.set_title(r'$e_y$$_y $', fontsize= fs)
    f.colorbar(f6, ax=ax6)


    plt.tight_layout()
    plt.savefig(str(directory) + '.png', dpi=2000)
pplot(outfile)
plt.show()



