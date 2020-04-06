#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 16:31:54 2020

@author: briansquires
"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
import hyperspy.api as hs
#%%



def Polar(xcoord,ycoord, orientation, norm, file):
    s = hs.load(file, lazy=True)
    fig, axes = plt.subplots(int(np.floor(np.sqrt(s.axes_manager[2].size))), 
                             int(np.ceil(np.sqrt(s.axes_manager[2].size))), 
                             subplot_kw=dict(projection='polar'), 
                             sharex='all', sharey='all',
                             tight_layout=True)
    
    theta = np.radians(np.arange(0,360,2))
    if orientation <=1:
        
        Q = [f for f in s.data[orientation,:,ycoord,xcoord,:]]
        
        axes = axes.ravel()
        for i in range(axes.size-1):
            w =  np.arange(780,922,2)
            w = w[i]
            
            if norm==False:
                axes[i].plot(theta,Q[i])
            elif norm==True:
                axes[i].plot(theta,Q[i]/Q[i].max())
            else:
                print('Identify Norm')
            axes[i].set_title(f'{w}nm')
    elif orientation == 2:
        
        Q = [f for f in s.data[0,:,ycoord,xcoord,:]]
        R = [f for f in s.data[1,:,ycoord,xcoord,:]]
        axes = axes.ravel()
        for i in range(axes.size):
            w =  np.arange(780,922,2)
            w = w[i]
            axes[i].plot(theta,Q[i]+R[i])
            axes[i].set_title(f'{w}nm')
       


    [axi.set_rticks([]) for axi in axes.ravel()]
    [axi.set_xticks([0,np.pi/3,2*np.pi/3,np.pi,4*np.pi/3,5*np.pi/3]) for axi in axes.ravel()]
    O = ['Parallel','Perpendicular', 'Parallel \n + \n Perpendicular']
    fig.suptitle(O[orientation]+f' \n X = {xcoord} \n Y = {ycoord}')



#%%
    
    
def TriPolar(xcoord,ycoord):
    fig, axes = plt.subplots(int(np.floor(np.sqrt(s.axes_manager[2].size))), 
                             int(np.ceil(np.sqrt(s.axes_manager[2].size))), 
                             subplot_kw=dict(projection='polar'), 
                             sharex='all', sharey='all',
                             tight_layout=True)
    
    
    theta = np.radians(np.arange(0,360,2))
    
        
    Q = [f for f in s.data[0,:,ycoord,xcoord,:]]
    R = [f for f in s.data[1,:,ycoord,xcoord,:]]


    axes = axes.ravel()
    for i in range(axes.size-1):
        w =  np.arange(780,922,2)
        w = w[i]
        l1, = axes[i].plot(theta,Q[i], color='tab:green')
        l2, = axes[i].plot(theta,R[i], color='tab:blue')
        l3, = axes[i].plot(theta,Q[i]+R[i], color='tab:red')
        axes[i].set_title(f'{w}nm')
    fig.legend((l1, l2,l3), (r'$\parallel$', r'$\perp$',r'$\parallel + \perp$'),
               'upper left',prop={'size': 15})
       
    
        
    [axi.set_rticks([]) for axi in axes.ravel()]
    [axi.set_xticks([0,np.pi/3,2*np.pi/3,np.pi,4*np.pi/3,5*np.pi/3]) for axi in axes.ravel()]
    
    fig.suptitle(f' \n X = {xcoord} \n Y = {ycoord}', fontsize=24)
    fig.tight_layout()
    
#%%
    
    
def TriPolarNorm(xcoord,ycoord):
    fig, axes = plt.subplots(int(np.floor(np.sqrt(s.axes_manager[2].size))), 
                             int(np.ceil(np.sqrt(s.axes_manager[2].size))), 
                             subplot_kw=dict(projection='polar'), 
                             sharex='all', sharey='all',
                             tight_layout=True)
    
    
    theta = np.radians(np.arange(0,360,2))
    
        
    Q = [f for f in s.data[0,:,ycoord,xcoord,:]]
    R = [f for f in s.data[1,:,ycoord,xcoord,:]]
    S = []

    axes = axes.ravel()
    for i in range(axes.size-1):
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
    
