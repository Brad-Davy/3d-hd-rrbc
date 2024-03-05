"""Analysis file of 3d rotating rayleigh benard convection
fig = plt.figure(figsize=(7.5*cm, 9*cm))
Usage:
    analysis.py --dir=<directory> [--t=<transient> --fig=<Figure> --snap_t=<snap_t>]
    analysis.py -h | --help

Options:
    -h --help   			Display this help message
    --dir=<directory>        		Directory
    --t=<transient>   			Transient to be ignored [default: 500]
    --fig=<Figure>    			Produce Figures [default: 0]
    --snap_t=<snap_t>	                Snapshots to be ignored [default: 5]
"""

from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy import stats
from colours import*
import subprocess

# =============================================================================
# Extract the docopt arguments 
# =============================================================================

args = docopt(__doc__)
dir = str(args['--dir'])
nConvectiveTimeUnits = int(args['--t'])
fig_bool = int(args['--fig'])
snap_t = int(args['--snap_t'])

# =============================================================================
# Extract run information
# =============================================================================

for idx,lines in enumerate(dir.split('/')[1].split('_')):
    if idx == 1:
        Ra = float(lines.replace('-','.'))
    if idx == 3:
        a,b,c,d,e,f,g,h = lines
        Ek = float(a+'.'+c+d+'e-'+g+h) 
    if idx == 5:
        Pr = float(lines.replace('-','.'))
    if idx == 7:
        Nx = float(lines)  

Ny = Nx
Nz = Nx/2
Lx = Nx/Nz

# =============================================================================
# Deal with the analysis tasks to calculate boundary layers
# =============================================================================

analysis_file = os.listdir(dir)

# =============================================================================
# Fine the file names and store them as a variable 
# =============================================================================

for idx,lines in enumerate(os.listdir(dir)):
    if lines.find('analysis') != -1:
        analysis_file_name = os.listdir(dir)[idx]

for idx,lines in enumerate(os.listdir(dir)):
    if lines.find('snapshot') != -1:
        snapshot_file_name = os.listdir(dir)[idx]

# =============================================================================
# Load the data from the analysis file 
# =============================================================================

with h5py.File('{}{}/analysis.h5'.format(dir,analysis_file_name), mode = 'r') as file:
    z = np.copy(file['tasks']['z'])[-1,0,0,:]
    T_prof = np.copy(file['tasks']["T_prof"])

# =============================================================================
# Load the data from the snapshot file 
# =============================================================================

with h5py.File('{}{}/snapshots.h5'.format(dir,snapshot_file_name), mode = 'r') as file:
    numberOfFiles = np.shape(np.copy(file['tasks']['T']))[0]
    full_T = np.copy(file['tasks']['T'])[-snap_t:,:,:,:]

def calculateSTDOfTrms(data):
    topBoundaries = []
    bottomBoundaries = []
    shape = np.shape(data)
    zPoints = int(shape[1]/2)
    
    for lines in data:
        locationOne = np.argmax(lines[:zPoints])
        locationTwo = np.argmax(lines[zPoints:])        
        bottomBoundaries.append(0.5 + z[locationOne])
        topBoundaries.append(z[locationTwo])
    
    print(bottomBoundaries)
    avgBoundaryThickness = np.average(bottomBoundaries)
    stdBoundaryThickness = np.std(bottomBoundaries) 

    return avgBoundaryThickness, stdBoundaryThickness

# =============================================================================
# Construct a 3d array the same size as T in order to use numpy to subtract 
# T_prof, i.e. avoid for loops. - Not sure if this is still needed, will check
# on arc.
# =============================================================================

avg_T_prof = np.average(np.array(T_prof[:,0,0,:]), axis=0) 
T_rms = []

for idxt,time_step in enumerate(full_T):
    temp_array = []
    if idxt >= 0:
        for idx,lines in enumerate(np.rot90(time_step, k = 3, axes = (0,2))):
            temp_array.append(np.average(np.sqrt((lines - avg_T_prof[idx]*np.ones(np.shape(lines)))**2)))
        T_rms.append(temp_array)

avg, std = calculateSTDOfTrms(T_rms)
print('There are {} files.'.format(numberOfFiles))
print('Avg: {:.4f}, STD: {:.4f}, % diff: {:.2f}%'.format(avg, std, (std/avg)*100))

