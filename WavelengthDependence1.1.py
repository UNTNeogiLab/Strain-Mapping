# -*- coding: utf-8 -*-
"""
Spyder Editor

This script defines functions for initializing data structures
and instruments used for SHG strain imaging microscopy.

"""

from instrumental import instrument, u
from pyvcam import pvc
from pyvcam.camera import Camera
import pyvisa
import numpy as np
from time import sleep
from datetime import datetime
import os
import tifffile
import glob
from tqdm import tqdm
import subprocess
import h5py
import time

#%%
def InitializeInstruments():
    """
    Initializes the camera and rotators to the desired names.
    TODO: Figure out how to set the camera to 'quantview' mode.

    Parameters
    ----------
    none
    
    Returns
    -------
    cam : object
        Named pyvcam camera object.
    A : object
        Named Instrumental instrument object.
    B : object
        Named Instrumental instrument object.
    C : object
        Named Instrumental instrument object.

    """
    pvc.init_pvcam()    # Initialize PVCAM
    cam = next(Camera.detect_camera()) # Use generator to find first camera
    cam.open()                         # Open the camera.
    if cam.is_open == True:
        print("Camera open")
    else:
        print("Error: camera not found")
    try:
        A = instrument('A')    # try/except is used here to handle
    except:                    # a bug in instrumental that requires
        A = instrument('A')    # this line to be run twice
    print("A.serial = " + A.serial)
    try:
        B = instrument('B')
    except:
        B = instrument('B')
    print("B.serial = " + B.serial)
    try:
        C = instrument('C')
    except:
        C = instrument('C')
    print("C.serial = " + C.serial)
    
    return cam, A, B, C


cam, rotator_bottom, rotator_top, C = InitializeInstruments()
#%%

rm = pyvisa.ResourceManager()
Pmeter = rm.open_resource('ASRL3::INSTR')
MaiTai = rm.open_resource('ASRL1::INSTR')

#%%
def InitializeDateFolder(location='E:\\Imaging'):
    """
    Creates a folder named by current date and changes the working directory
    to match. Should be called first.

    Parameters
    ----------
    location : string
        The path of the parent folder the new data folder should be
        generated in.
    Returns
    -------
    newfolder : pathstring
        The path of the generated folder.

    """
    today = datetime.today()
    datefolder = location + '\\' + str(today.date())
    if os.path.exists(datefolder):
        os.chdir(datefolder)
        pass
    else:
        os.mkdir(datefolder)
        os.chdir(datefolder)
        
    return datefolder
#%%
def InitializeRunFolder(sample, sample_origin, wavelength,
                        datefolder, circ_pol=None):
    """
    Creates a folder for an individual strain mapping run.
    TODO: figure out a better way to save the timestamp
    
    Parameters
    ----------
    sample : string
        Identifies the sample.
    sample_origin : string
        Identifies the maker of the sample
    circ_pol : numeric, optional
        Identifies the circular polarization that HalfWaveLoop() will be
        run at.
    datefolder : pathstring
        The parent directory to create runfolder in.

    Returns
    -------
    runfolder : pathstring
        The path of the generated folder.

    """
    now = datetime.now()
    #time = now.strftime("%H.%M")
    if circ_pol == None:
        runfolder = datefolder + '\\' + sample + '_' + str(wavelength) + 'nm'
    else:
        cp_folder = datefolder + '\\' + sample + '_' + circ_pol
        os.mkdir(cp_folder)
        os.chdir(cp_folder)
    os.mkdir(runfolder)
    os.chdir(runfolder)
    
    return runfolder, sample
#%%

#%%
def CloseInstruments(cam, A, B, C):
    cam.close()
    pvc.uninit_pvcam()
    A.close()
    B.close()
    C.close()
#%%
def HalfWaveLoop(orientation ,datefolder, wavelength, sample, sample_origin, cam, rotator_top,
                 rotator_bottom, runfolder, start, stop, step, zfill,
                 delay, exp_time, circ_pol):
    """
    Main strain-mapping loop. Rotates pre- and post-sample halfwave plates
    in tandem, taking camera images at each polarization step. Also saves
    metadata.txt in the data folder.

    Parameters
    ----------
    wavelength : numeric
        Measurement wavelength. Metadata parameter.
    sample : string
        Sample name. Metadata parameter.
    sample_origin : string
        Where the sample came from. Metadata parameter.
    cam : object
        pvam Camera object.
    rotator_top : object
        Instrumental K10CR1 object.
    rotator_bottom : object
        Instrumental K10CR1 object.
    start : numeric, optional
        Polarization range start point. For a calibrated run, set to
        the desired offset. Must be a positive value.
    stop : numeric, optional
        Polarization range stop point. 
    step : numeric, optional
        Polarization step size. Lower values yield higher resolution.
    zfill : numeric, optional
        Number of zeros to pad the output filenames
        with. Should be increased for high resolution
        datasets. The default is 3.
    delay : numeric, optional
        The pre-acquisition delay time in seconds.
        Used to give time to turn off light sources
        and vacate the lab. The default is 180.
    exp_time : numeric, optional
        Camera exposure time in ms. The default is 10000.
    circ_pol : numeric, optional
        The circular polarization this run was acquired at. 

    Returns
    -------
    None.

    """
    if orientation == 'parallel':
        sys_offset = 0 * u.degree
        folder = runfolder + '\\' + 'parallel'
        os.mkdir(folder)
        os.chdir(folder)
    elif orientation == 'perpendicular':
        sys_offset = 45 * u.degree
        folder = runfolder + '\\' + 'perpendicular'
        os.mkdir(folder)
        os.chdir(folder)
    #tqdm.write('Homing bottom rotator')
    rotator_bottom.home(wait=True)
    #tqdm.write('Homing top rotator')
    rotator_top.home(wait=True)
    tick = datetime.now()
    stop = start + stop
    sleep(delay)
    step = step * 0.5
    Data = []

    R = tqdm(np.arange(start, stop, step),
                  desc=f'{orientation} at {wavelength} nm',
                  position=0, leave=True)
                 
    for i in R:
        
        
        position = i * u.degree
        position_top = position - rotator_top.offset + sys_offset
        position_bottom = position - rotator_bottom.offset
        #position_bottom = (360*u.degree)-(position - rotator_bottom.offset)
        strpos = str(2 * i)
        padded = strpos.zfill(zfill)
        name = 'halfwave' + padded
        rotator_top.move_to(position_top, wait=True)
        rotator_bottom.move_to(position_bottom, wait=True)
        frame = cam.get_frame(exp_time=exp_time)
        np.save(name, frame, allow_pickle=False)
        with h5py.File(datefolder+'\\'+sample+'.h5','a') as hdf: 
            hdf.create_dataset(f'{wavelength}/{orientation}/{padded}',data=Data)
    tock = datetime.now()
    delta = tock - tick
    wavelength = str(wavelength) + ' nm'
    with open('metadata.txt', mode='w') as f:
        print('Start time: ' + str(tick), file=f)
        print('End time: ' + str(tock), file=f)
        print('Total Acquisition time: ' + str(delta), file=f)
        print('Wavelength: ' + str(wavelength), file=f)
        print('Exposure time: ' + str(exp_time) + 'ms', file=f)
        print('Sample: ' + str(sample), file=f)
        print('Sample Origin: ' + str(sample_origin), file=f)
        print('Polarization range: ' + str(2 * start) +
              ' to ' + str(2 * stop) + 'deg', file=f)
        print('Polarization resolution: ' + str(2 * step) + 'deg', file=f)
        if circ_pol is not None:
            print('Circular Polarization: ' + str(circ_pol) + 'deg', file=f)
        else:
            pass
    os.chdir(runfolder)

    return folder

#%%
def ClusterSync():
    """
    Syncs data to the cluster.
    TODO: figure out how to make the 'rsync' string more readable/maintainable,
    and how to pass the desired data location.

    Parameters
    ----------
    datefolder : TYPE, optional
        DESCRIPTION. The default is datefolder.
    runfolder : TYPE, optional
        DESCRIPTION. The default is runfolder.

    Returns
    -------
    None.

    """
    today = datetime.today()
    shell = 'C:\Program Files (x86)\Mobatek\MobaXterm\MobaXterm.exe'
    kill = '-exitwhendone'
    tab = '-newtab'
    rsync = ('rsync --chmod=Du=rwx,Dgo=rwx,Fu=rw,Fog=rw -avh'+
             ' \'/bin/ssh -x -T -c arcfour -o Compression=no\''+
             ' /drives/e/Imaging/'+str(today.date())+
             ' jmt0288@talon3.hpc.unt.edu:/storage/scratch2/share/pi_an0047/autoupload/')
    subprocess.call([shell, kill, tab, rsync])
#%%
def TiffSave(runfolder):
    """
    Generates a TIFF stack from a folder of numpy arrays for quick analysis
    in ImageJ.

    Parameters
    ----------
    runfolder : pathstring
        The folder containing numpy arrays to be concatenated as a 
        Tiff stack.

    Returns
    -------
    None.

    """
    os.chdir(runfolder)
    filelist = glob.glob(runfolder + '\\*.npy')
    filelist = sorted(filelist)
    datacube = np.array([np.load(fname) for fname in filelist])
    tifffile.imsave('out.tiff', datacube)
#%%
def CheckRotators(A, B, C):
    """
    Verifies physical position of half wave plate rotation mounts and assigns
    initialized rotators to the correct variables for HalfWaveLoop().

    Parameters
    ----------
    A : object
        Instrumental K10CR1 object.
    B : object
        Instrumental K10CR1 object.
    C : object
        Instrumental K10CR1 object.

    Returns
    -------
    rotator_top : object
        Instrumental K10CR1 object.
    rotator_bottom : object
        Instrumental K10CR1 object.
    cp_post : object
        Instrumental K10CR1 object.

    """
    response = ''
    while response != 'y':
        response = input("Are the rotator locations unchanged? Enter " +
                         "'y' to continue, 'n' to manually set rotator_top " +
                         "and rotator_bottom\n" + 
                         '>>>')
        rotator_top = input("Enter name (A, B, or C) of post-sample half-wave"
                            + " rotator:\n" +
                            ">>>")
        if rotator_top == 'A':
            rotator_top = A
        elif rotator_top == 'B':
            rotator_top = B
        elif rotator_top == 'C':
            rotator_top = C
        else:
            pass
        rotator_bottom = input("Enter name (A, B, or C) of pre-sample " +
                               "half-wave rotator:\n" +
                               ">>>")
        if rotator_bottom == 'A':
            rotator_bottom = A
        elif rotator_bottom == 'B':
            rotator_bottom = B
        elif rotator_bottom == 'C':
            rotator_bottom = C
        else:
            pass
        cp_post = input("Enter name (A, B, or C) of post-sample " +
                               "quarter-wave rotator:\n" +
                               ">>>")
        if cp_post == 'A':
            cp_post = A
        elif cp_post == 'B':
            cp_post = B
        elif cp_post == 'C':
            cp_post = C
        else:
            pass
    return rotator_top, rotator_bottom, cp_post

#%%
    

def Power(): 
    """Reads power from Gentec TPM300 via VISA commands
    The while loop avoids outputting invalid token
    >>>returns float
    
    to-do: incorporate different power ranges (itteratively check all avaliable
    ranges and chose the best fit. Log this choice)"""
    
    
    while True:
        try:
            Pread = Pmeter.query("*READPOWER:")
            Power = float(Pread.split('e')[0].split('+')[1])
            return Power
        except:
             continue
         
        
def MoveWav(position):
    """Helper function for instrumental to avoid clutter and make code 
    more readable
    >>>returns null"""
    MaiTai.write(f"WAV {position}")
    
def ReadWav():
    """Helper function for instrumental to avoid clutter and make code 
    more readable
    >>>returns int"""
    w = int(MaiTai.query("WAV?").split('n')[0])
    return w
    
def Shutter(op):
    """Helper function for instrumental to avoid clutter and make code 
    more readable
    >>>returns string""" 
    if op == 1:
        MaiTai.write("SHUT 1")
        #tqdm.write("Shutter Opened")
    else:
        MaiTai.write("SHUT 0")
        #tqdm.write("Shutter Closed")
        
        
def MoveRot(position, Rot=C):
    """Helper function for instrumental to avoid clutter and make code 
    more readable
    >>>returns null"""
    Rot.move_to(position*u.degree)


def PowerRotLoop(pstart, pstop, pstep, pwait):
    """Main acquisition code to collect power as a function of 
    rotation stage angle. This can be run separately, but is embedded in 
    WavLoop for wavelength dependent calibration
    ************
    pstart = Initial angular position in degrees
    pstop = Final angular position in degrees
    pstep = Angular step size in degrees
    pwait = wait time between steps in seconds
    >>>>>returns P = 2D Array"""
    
    print("Homing")
    C.home(wait = True)
    time.sleep(5)
    print('Homing finished')
    Pwr = []
    Pos = []
    for i in np.arange(pstart,pstop + pstep,pstep):
        MoveRot(i)
        time.sleep(pwait)
        print(str(C.position) + '>>>>>> ' + str(Power()) +' mW')
        Pos.append(float(str(C.position).split(' ')[0]))
        Pwr.append(Power())
        P = np.asarray([Pos,Pwr])
    return P
    
def WavLoop(wavstart,wavstop,wavstep,wavwait,
            wpstart, wpstop, wpstep, wpwait, 
            filename):
    """Main acquisiton code used to collect wavelength dependent calibration.
    First, the laser wavelength is set to wavstart. Then, PowerRotLoop is 
    called to collect Power vs Angle for said wavelength. The laser
    wavelength is then set to the next step.
    
    *************
    Figure out the inputs yourself, it's pretty self exxplanitory.
    
    ********
    >>>>>returns W = a list of 2D arrays """
    print('Starting Wavelength Loop')
    if (950 >= wavstart >= 750 and 950 >= wavstop >= 750):
        MoveWav(wavstart)
        time.sleep(wavwait*2)
        W = []
        for w in np.arange(wavstart,wavstop+wavstep,wavstep):
            MoveWav(w)
            time.sleep(wavwait)
            wavelength = ReadWav()
            print (f'Starting Power Loop at {wavelength}nm')
            Pwr = PowerRotLoop(wpstart, wpstop, wpstep, wpwait)
            W.append([wavelength, Pwr])
        np.save(f'{filename}',W, allow_pickle = True)
        return W
    else:
        print('Wavelengths out of range')
        
#%% Save Function
def SaveAsFiles(filename):

    A = np.load(filename, allow_pickle=True) 
    B = [np.transpose(A[i,1]) for i in np.arange(0,A[:,0].size,1)]
    cwd = os.getcwd()
    today = datetime.today()
    path = 'PowerCalibration' + str(today.date())
    os.mkdir(path)
    os.chdir(path)
    [np.savetxt(f'{A[i,0]}.csv',B[i], delimiter=',') for i in np.arange(len(B))]
    os.chdir(cwd)
         
    
#%%    
def Sin2(angle,mag,xoffset, yoffset):
    return mag*np.sin(angle*2*np.pi/360 - xoffset)**2 + yoffset
def PowerFit(data):
    popt, pcov = curve_fit(Sin2, data[0], data[1])
    return popt, pcov

#%%
def InvSineSqr(y,y0,xc,w,A):
    return xc + (w/(np.pi))*np.arcsin(np.sqrt((y - y0)/A))

#%%Notes
    """Parameters for WavLoop shoiuld be ~ (750,950,2,60,0,44,2,5,'filename')
        for this to run in ~8hrs
        Further testing is needed to determine optimal parameters"""
        
        
        #%%%%%%%%%%%%%%%%%%%%%%%
def WavelengthRASHG(power, wavelength_start, wavelength_end, wavestep, 
                    wavelength_wait, res, exp, sample, sample_origin, Tiff):
    D = np.loadtxt('C:/Users/Mai Tai/Desktop/Python Code/PowerCalibDatabase.txt')
    Dlist = np.ndarray.tolist(D[:,0])
    
    cam.roi = (500,1700,600,1800)
    
    
    datefolder = InitializeDateFolder()
       
    
    MoveWav(wavelength_start)
    time.sleep(wavelength_wait*2)
    
    pbar =tqdm(np.arange(wavelength_start,wavelength_end+wavestep, wavestep),desc='Total',position=0)
    
    for w in pbar:
       
# =============================================================================
#             with h5py.File(datefolder+'\\'+sample+'.h5','a') as hdf: 
#                 wgrp = hdf.create_group(f'{w}')
# =============================================================================
        
            runfolder, sample = InitializeRunFolder(sample, sample_origin, 
                                                        f'{w}', datefolder)
            Shutter(0)
            MoveRot(InvSineSqr(power,*D[Dlist.index(w),1:5]))
            MoveWav(w)
            #print(f'Moving to {w}')
           
            time.sleep(wavelength_wait)
            #print(f'Starting at {w} nm')
            
            Shutter(1)
            
            HalfWaveLoop('parallel', datefolder, w, 
                        sample, sample_origin, cam,
                        rotator_top, rotator_bottom, runfolder,
                        start = 0, stop=180, step=res, zfill=5,
                        delay=0, exp_time=exp,
                        circ_pol=None)
            HalfWaveLoop('perpendicular', datefolder, w, 
                        sample, sample_origin, cam,
                        rotator_top, rotator_bottom, runfolder,
                        start = 0, stop=180, step=res, zfill=5,
                        delay=0, exp_time=exp,
                        circ_pol=None)
            ClusterSync()
        
 
    

 
 
WavelengthRASHG(20, 780, 920, 2, 10, 2, 1000, 'MoS2_hires_wav', 'Y+V', False)    
    

