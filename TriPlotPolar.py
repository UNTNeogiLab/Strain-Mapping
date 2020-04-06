import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs

def TriPolar(xcoord,ycoord):
    fig, axes = plt.subplots(8, 9, subplot_kw=dict(projection='polar'), 
                             sharex='all', sharey='all',
                             tight_layout=True)
    
    
    theta = np.radians(np.arange(0,360,2))
    
        
    Q = [f for f in s.data[0,:,ycoord,xcoord,:]]
    R = [f for f in s.data[1,:,ycoord,xcoord,:]]


    axes = axes.ravel()
    for i in range(axes.size):
        w =  np.arange(780,922,2)
        w = w[i]
        l1, = axes[i].plot(theta,Q[i], color='tab:green')
        l2, = axes[i].plot(theta,R[i], color='tab:blue')
        l3, = axes[i].plot(theta,Q[i]+R[i], color='tab:red')

    fig.legend((l1, l2,l3), (r'$\parallel$', r'$\perp$',r'$\parallel + \perp$'),
               'upper left',prop={'size': 15})
       
    
        
    [axi.set_rticks([]) for axi in axes.ravel()]
    [axi.set_xticks([0,np.pi/3,2*np.pi/3,np.pi,4*np.pi/3,5*np.pi/3]) for axi in axes.ravel()]
    
    fig.suptitle(f' \n X = {xcoord} \n Y = {ycoord}', fontsize=24)
    fig.tight_layout()
    
#%%
    
    
def TriPolarNorm(xcoord,ycoord):
    fig, axes = plt.subplots(8, 9, subplot_kw=dict(projection='polar'), 
                             sharex='all', sharey='all',
                             tight_layout=True)
    
    
    theta = np.radians(np.arange(0,360,2))
    
        
    Q = [f for f in s.data[0,:,ycoord,xcoord,:]]
    R = [f for f in s.data[1,:,ycoord,xcoord,:]]
    S = []

    axes = axes.ravel()
    for i in range(axes.size):
        w =  np.arange(780,922,2)
        w = w[i]
        l1, = axes[i].plot(theta,Q[i]/Q[i].max(), color='tab:green')
        l2, = axes[i].plot(theta,R[i]/R[i].max(), color='tab:blue')
        l3, = axes[i].plot(theta,(Q[i]+R[i])/(Q[i]+R[i]).max(), color='tab:red')
        S.append(np.sum(Q[i]+R[i]))
        axes[i].set_title(f'{w}nm')
    fig.legend((l1, l2,l3), (r'$\parallel$', r'$\perp$',r'$\parallel + \perp$'),
               'upper left',prop={'size': 15})
       
    
        
    [axi.set_rticks([]) for axi in axes.ravel()]
    [axi.set_xticks([0,np.pi/3,2*np.pi/3,np.pi,4*np.pi/3,5*np.pi/3]) for axi in axes.ravel()]
    fig.suptitle(f' \n X = {xcoord} \n Y = {ycoord}', fontsize=24)
    fig.tight_layout()
    
