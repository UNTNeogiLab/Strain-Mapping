# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import interpolate
from scipy import optimize
from tempfile import TemporaryFile
import ray

ray.init()

@ray.remote
def strainfit(phi, a, b, delta, theta):
    return 0.25*(a * np.cos(3*phi- 3*delta) + b * np.cos(2 * theta + phi - 3*delta))**2


#%% Import Files
filelist = glob.glob('/storage/scratch2/share/pi_an0047/191009/Imaging/2019-10-16/*.npy')
filelist = sorted(filelist)

x = ray.put(np.array([np.load(fname) for fname in filelist]))
#%%Define Variables

phi = np.arange(0,2*x.shape[0],2)*np.pi/180
cyc = phi/(2*np.pi)


#%%
threshold = 4000
paramsimg2d = []
emptyparams = np.zeros(4)
for i in range(x.shape[1]):
    paramsimg = []
    for j in range(x.shape[2]):
        if x[:,i,j].mean() > threshold:
            xx = (x[:,i,j]-np.amin(x[:,i,j]))/np.amax(x[:,i,j]-np.amin(x[:,i,j]))
            params, params_covariance = optimize.curve_fit(strainfit.remote(), phi, xx, maxfev=10000000,
                                                method='lm')
            paramsimg.append(params)
        else:
           paramsimg.append(emptyparams)
    paramsimg2d.append(paramsimg)

paramsarray = np.array(paramsimg2d)



np.save('outfile', paramsarray)
outfile = np.load('outfile.npy')

#%%
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


f , ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey = False)
ax1.plot(theta, x[:,1300,1300])
ax2.plot(theta, x[:,1300,1300])
ax3.plot(xfftfreq, xfft.real[:,1000,1000],)
ax3.set_title('Re{FFT}')
ax4.plot(xfftfreq, xfft.imag[:,1000,1000])
ax4.set_title('Im{FFT}')
ax3.set_xlim(-.05,.05)
ax4.set_xlim(-.05,.05)





