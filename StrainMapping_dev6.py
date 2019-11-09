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

filelist = glob.glob(str(directory) + '/*.npy')
filelist = sorted(filelist)
x = np.array([np.load(fname) for fname in filelist])




#%%
tic = time.time()
def strainfit(phi, a, b, delta, theta):
    return 0.25*(a * np.cos(3*phi- 3*delta) + b * np.cos(2 * theta + phi -
                 3*delta))**2
N = 1
threshold = np.average(x)*1.2
phi = np.arange(0,2*x.shape[0],2)*np.pi/180
def f(v):
    i = v[0]*N
    j = v[1]*N
    t = [v[0],v[1], 0, 0, 0, 0 ]
    if x[:,i,j].mean() > threshold:
        xx = (x[:,i,j]-np.amin(x[:,i,j]))/np.amax(x[:,i,j]-np.amin(x[:,i,j]))
        params, params_covariance = optimize.curve_fit(strainfit, phi, xx,
                                                       maxfev=1000000,
                                                       method='lm')
        for k in range(4):
            t[k+2] = params[k]
    return t
paramsarray = np.zeros((x.shape[1]//N, x.shape[2]//N, 4))
V = itertools.product(range(x.shape[1]//N), range(x.shape[2]//N) )
A = Parallel(n_jobs=nCores)( delayed(f)(v) for v in V )
for a in A:
    for k in range(4):
        paramsarray[a[0],a[1],k] = a[k+2]
np.save(directory + '/outfile', paramsarray)
#np.save('outfile', paramsarray)
outfile = np.load(directory + '/outfile.npy')
toc = time.time()
print(toc - tic)
#%%
import numpy as np
import matplotlib.pyplot as plt
directory =os.getcwd()
outfile = np.load(directory + '/outfile.npy')
def pplot(paramsarray):
    f , ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,
        sharex=True,sharey=True,dpi=200)
    f1 = ax1.imshow(paramsarray[:,:,0], cmap='jet')
    f2 = ax2.imshow(paramsarray[:,:,1], cmap='jet')
    f3 = ax3.imshow(np.mod(paramsarray[:,:,2]*60/(np.pi),60),
                    cmap='jet')
    f4 = ax4.imshow(np.mod(paramsarray[:,:,3]*60/(np.pi),60),
                    cmap='jet')
    ax1.set_title('A')
    f.colorbar(f1, ax=ax1)
    ax2.set_title('B')
    f.colorbar(f2, ax=ax2)
    ax3.set_title(r'$\delta (Orientation)$')
    f.colorbar(f3, ax=ax3)
    ax4.set_title(r'$\theta (Strain angle)$')
    f.colorbar(f4, ax=ax4)
    #f4.clim(0,60)
    plt.show()
pplot(outfile)


