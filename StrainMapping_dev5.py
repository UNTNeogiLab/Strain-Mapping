# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
#import matplotlib.pyplot as plt
import glob
from scipy import optimize
import itertools
from  joblib import Parallel, delayed
import multiprocessing as mP
import time

nCores = mP.cpu_count()

def pplot(paramsarray):

    f , ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharex=True, sharey=True, dpi=200)
    ax1.imshow(paramsarray[:,:,0], cmap='jet')
    ax2.imshow(paramsarray[:,:,1], cmap='jet')
    ax3.imshow(np.mod(paramsarray[:,:,2], 2*np.pi/6),cmap='jet')
    ax4.imshow(paramsarray[:,:,3], cmap='jet')
    ax1.set_title('A')
    ax2.set_title('B')
    ax3.set_title(r'$\delta (Orientation)$')
    ax4.set_title(r'$\theta (Strain angle)$')
    plt.show()


#%% Import Files
filelist = glob.glob('/storage/scratch2/share/pi_an0047/191009/Imaging/2019-10-16/*.npy')
filelist = sorted(filelist)
x = np.array([np.load(fname) for fname in filelist])
#%%Define Variables
phi = np.arange(0,2*x.shape[0],2)*np.pi/180
cyc = phi/(2*np.pi)

#%%
tic = time.time()
def strainfit(phi, a, b, delta, theta):
    return 0.25*(a * np.cos(3*phi- 3*delta) + b * np.cos(2 * theta + phi - 3*delta))**2

N = 1
threshold = 3500

def f(v):
    i = v[0]*N
    j = v[1]*N
    t = [v[0],v[1], 0, 0, 0, 0 ]
    if x[:,i,j].mean() > threshold:
        xx = (x[:,i,j]-np.amin(x[:,i,j]))/np.amax(x[:,i,j]-np.amin(x[:,i,j]))
        params, params_covariance = optimize.curve_fit(strainfit, phi, xx,
                                                       maxfev=10000000,
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


np.save('outfile', paramsarray)
outfile = np.load('outfile.npy')

toc = time.time()
print(toc - tic)

pplot(paramsarray)


