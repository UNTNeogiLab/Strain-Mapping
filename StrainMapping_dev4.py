# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import optimize
import itertools
from  joblib import Parallel, delayed
import multiprocessing as mP

nCores = mP.cpu_count()
#%% Import Files
filelist = glob.glob('/storage/scratch2/share/pi_an0047/191009/Imaging/2019-10-16/*.npy')
filelist = sorted(filelist)
x = np.array([np.load(fname) for fname in filelist])
#%%Define Variables
phi = np.arange(0,2*x.shape[0],2)*np.pi/180
cyc = phi/(2*np.pi)
#%%Fit to Function
def strainfit(phi, a, b, delta, theta):
    return 0.25*(a * np.cos(3*phi- 3*delta) + b * np.cos(2 * theta + phi - 3*delta))**2

threshold = 4000
paramsimg2d = []

N = 128

paramsarray = np.zeros((x.shape[1]//N, x.shape[2]//N, 4))

emptyparams = np.zeros(4)
for iN in range(x.shape[1]//N):
    paramsimg = []
    i = N*iN
    for jN in range(x.shape[2]//N):
        j = N*jN
        if x[:,i,j].mean() > threshold:
            xx = (x[:,i,j]-np.amin(x[:,i,j]))/np.amax(x[:,i,j]-np.amin(x[:,i,j]))
            params, params_covariance = optimize.curve_fit(strainfit, phi, xx, maxfev=10000000,
                                                method='lm')
            paramsimg.append(params)            # original
            #paramsimg2d.append(params)
            for k in range(4):
                paramsarray[iN, jN, k] = params[k]
        #else:
           #paramsimg.append(emptyparams)        # original
           #paramsimg2d.append(emptyparams)
    #paramsimg2d.append(paramsimg)               # original

#paramsarray = np.array(paramsimg2d)
np.save('outfile', paramsarray)
outfile = np.load('outfile.npy')


#%%

def strainfit(phi, a, b, delta, theta):
    return 0.25*(a * np.cos(3*phi- 3*delta) + b * np.cos(2 * theta + phi - 3*delta))**2

N = 1
threshold = 4000

def f(v, p):
    i = v[0]*N
    j = v[1]*N
    if x[:,i,j].mean() > threshold:
        xx = (x[:,i,j]-np.amin(x[:,i,j]))/np.amax(x[:,i,j]-np.amin(x[:,i,j]))
        params, params_covariance = optimize.curve_fit(strainfit, phi, xx,
                                                       maxfev=10000000,
                                                       method='lm')
        for k in range(4):
            p[v[0], v[1], k] = params[k]


paramsarray = np.zeros((x.shape[1]//N, x.shape[2]//N, 4))


V = itertools.product(range(x.shape[1]//N), range(x.shape[2]//N) )

Parallel(n_jobs=nCores)( delayed(f)(v,paramsarray) for v in V )
np.save('outfile', paramsarray)
outfile = np.load('outfile.npy')




#%%Plot
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

#%%
# =============================================================================
# 
# fig = plt.figure()
# ax1 = plt.subplot(211)
# 
# ax2 = plt.subplot(212, projection='polar')
# =============================================================================


# =============================================================================
# f , ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey = False)
# ax1.plot(theta, x[:,1300,1300])
# ax2.plot(theta, x[:,1300,1300])
# ax3.plot(xfftfreq, xfft.real[:,1000,1000],)
# ax3.set_title('Re{FFT}')
# ax4.plot(xfftfreq, xfft.imag[:,1000,1000])
# ax4.set_title('Im{FFT}')
# ax3.set_xlim(-.05,.05)
# ax4.set_xlim(-.05,.05)
# 
# =============================================================================




