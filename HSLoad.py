#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:09:13 2020

@author: briansquires
"""
#%%
    

import sys
 
filename = sys.argv[1] 
wavstart = sys.argv[2] 
wavend   = sys.argv[3]
wavres   = sys.argv[4]


def HSLoad(filename,wavstart, wavend, wavres):
    """Loads and saves 5D hyperspy file.
    
    Currently assumes 2deg polarization resolution"""
    
    
    import numpy as np
    import matplotlib 
    matplotlib.rcParams["backend"] = "Agg"
    import glob
    import hyperspy.api as hs
    
    directories = [f for f in glob.glob('*_*nm')]
    directories.sort()
    def ParaLoad(directory):
        A =[]
        
        files = [f for f in glob.glob(str(directory)+'/parallel/*0.npy')]
        files = np.sort(files)
        
        for f in files:
            a = np.load(f)
            A.append(a)
        return A

    def PerpLoad(directory):
        B =[]
    
        files = [f for f in glob.glob(str(directory)+'/perpendicular/*0.npy')]
        files = np.sort(files)
        
        for f in files:
            a = np.load(f)
            B.append(a)
        return B
    
    
    LA = [ParaLoad(f) for f in directories]
    LB = [PerpLoad(f) for f in directories]
    s = hs.signals.Signal1D([LA,LB], lazy=True)
    s.axes_manager[1].name = "Polarization"
    s.axes_manager['Polarization'].units = 'degrees'
    s = s.as_signal1D('Polarization')  
    s.axes_manager['Polarization'].scale = 2
    s.axes_manager[0].name = "X"
    s.axes_manager[1].name = "Y"    
    s.axes_manager[2].name = "Wavelength"
    s.axes_manager['Wavelength'].offset = wavstart
    s.axes_manager['Wavelength'].scale = wavres
    s.axes_manager['X'].units = 'pixels'
    s.axes_manager['Y'].units = 'pixels'  
    s.axes_manager['Wavelength'].units = 'nm'
    s.axes_manager[3].name = 'Orientation'
    s.axes_manager[3].units = 'Para/Perp'
    s.save(filename)
    return s

    
#%%Parallel Implementation
    """Under Construction"""
    

def HSLoadMap(filename):
    
    import ipyparallel as ipp
    import glob
    rc = ipp.Client()
    v = rc[:]
    lv = rc.load_balanced_view()
    
    directories = [f for f in glob.glob('*_*nm')]
    directories.sort()
    def ParaLoad(directory):
        import glob
        import numpy as np
        A =[]
        
        files = [f for f in glob.glob(str(directory)+'/parallel/*0.npy')]
        files = np.sort(files)
        
        for f in files:
            a = np.load(f)
            A.append(a)
        return A

    def PerpLoad(directory):
        import glob
        import numpy as np
        B =[]
    
        files = [f for f in glob.glob(str(directory)+'/perpendicular/*0.npy')]
        files = np.sort(files)
        
        for f in files:
            a = np.load(f)
            B.append(a)
        return B
    
    import hyperspy.api as hs

    LA = v.map(ParaLoad,directories)
    LB = v.map(PerpLoad,directories)
    s = hs.signals.Signal1D([LA,LB], lazy=True)
    s.axes_manager[1].name = "Polarization"
    s.axes_manager['Polarization'].units = 'degrees'
    s = s.as_signal1D('Polarization')  
    s.axes_manager['Polarization'].scale = 2
    s.axes_manager[0].name = "X"
    s.axes_manager[1].name = "Y"    
    s.axes_manager[2].name = "Wavelength" 
    s.axes_manager['X'].units = 'pixels'
    s.axes_manager['Y'].units = 'pixels'  
    s.axes_manager['Wavelength'].units = 'nm'
    s.axes_manager[3].name = 'Orientation'
    s.axes_manager[3].units = 'Para/Perp'
    s.save(filename)
    return s

    







