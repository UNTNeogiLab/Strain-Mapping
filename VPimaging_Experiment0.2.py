# -*- coding: utf-8 -*-
"""
Spyder Editor

This script defines functions for initializing data structures
and instruments used for SHG strain imaging microscopy.

"""

from instrumental import instrument, u
from pyvcam import pvc
from pyvcam.camera import Camera
import numpy as np
from time import sleep
from datetime import datetime
import os
import tifffile
import glob
from tqdm import tqdm
import subprocess

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
        pass
    else:
        os.mkdir(datefolder)
        os.chdir(datefolder)
        
    return datefolder
#%%
def InitializeRunFolder(sample, sample_origin, datefolder, circ_pol=None):
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
    time = now.strftime("%H.%M")
    if circ_pol == None:
        runfolder = datefolder + '\\' + sample + '_' + time
    else:
        cp_folder = datefolder + '\\' + sample + '_' + circ_pol
        os.mkdir(cp_folder)
        os.chdir(cp_folder)
    os.mkdir(runfolder)
    os.chdir(runfolder)
    
    return runfolder, sample
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
#%%
def HalfWaveLoop(wavelength, sample, sample_origin, cam, rotator_top,
                 rotator_bottom, start=0, stop=180, step=1, zfill=4,
                 delay=180, exp_time=10000, circ_pol=None):
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
    tick = datetime.now()
    stop = start + stop
    sleep(delay)
    for i in tqdm(np.arange(start, stop, step)):
        position = i * step * u.degree
        position_top = position - rotator_top.offset
        position_bottom = position - rotator_bottom.offset
        strpos = str(2 * i)
        padded = strpos.zfill(zfill)
        name = 'halfwave' + padded
        rotator_top.move_to(position_top)
        rotator_bottom.move_to(position_bottom)
        rotator_top.wait_for_move()
        rotator_bottom.wait_for_move()
        sleep(1)
        frame = cam.get_frame(exp_time=exp_time)
        np.save(name, frame, allow_pickle=False)
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
def Start():
    """
    Basic CLI for polarization-resolved SHG microscopy. Prompts for user
    input and sets values for experiment parameters, initializes
    data directories and instruments, and runs HalfWaveLoop().

    Returns
    -------
    None.

    """
    response = input('Is this an ongoing VP measurement? Enter "y" for yes\n'
                     + '>>>')
    if response == 'y':
        vp = True   # if hi-rez circular polarization is implemented,
                    # use this to turn off the escape time pre-delay in
                    # HalfWaveLoop() and defeat the default value of 
                    # circ_pol in InitializeRunFolder
        pass
    else:
        sample = input('Enter sample name:\n'
                       + '>>>')
        sample_origin = input('Enter sample origin:\n'
                              + '>>>')
        wavelength = input('Enter wavelength:\n' + 
                           '>>>')
        resolution = input('Enter polarization resolution in degrees:\n' +
                           '>>>')
        exp_time = input('Enter exposure time in ms:\n' + 
                         '>>>')
        response = ''
        
    response = input('Save TIFF stack?')
    
    if response == 'n' or 'no':
        tiffsave = False
    else:
        tiffsave = True
        while response != 'y':
            response = input('Please verify that all settings are correct and'
                            +' that the\ncamera and rotators'
                             ' are ready for initialization.\n'+
                             'Enter y to continue.\n' + 
                             '>>>')
    cam, A, B, C = InitializeInstruments()
    rotator_top, rotator_bottom, cp_post = CheckRotators(A,B,C)
    datefolder = InitializeDateFolder()
    runfolder, sample = InitializeRunFolder(sample, sample_origin, datefolder)
    HalfWaveLoop(wavelength, sample, sample_origin, cam, rotator_top,
                 rotator_bottom, step=resolution, exp_time=exp_time)
    if tiffsave == True:
        TiffSave(runfolder)
    ClusterSync()
# =============================================================================
#     if vp == True:
#         circ_pol = input('Enter circular polarization value in degrees:\n' +
#                          '>>>')
#         InitializeRunFolder(sample, sample_origin, datefolder, circ_pol)
#         cp_post.move_to(circ_pol)
#         cp_post.wait_for_move()
#         sleep(1)
#         HalfWaveLoop(wavelength=wavelength, sample=sample,
#                      sample_origin=sample_origin, cam=cam,
#                      rotator_top=rotator_top, rotator_bottom=rotator_bottom,
#                      circ_pol=circ_pol)
#     else:
#         HalfWaveLoop(wavelength=wavelength, sample=sample,
#                      sample_origin=sample_origin, cam=cam,
#                      rotator_top=rotator_top, rotator_bottom=rotator_bottom)
#         
# =============================================================================
#%%
    Start()


