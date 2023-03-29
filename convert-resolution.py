""" 
    A script which converts the resolution of a run in both x and y, leaving
    z untouched. Used to compare the DNS cases to the HD cases.

Usage:
    convert-resolution.py [--dir=<directory> --scale=<Scale>] 
    convert-resolution.py -h | --help

Options:
    -h --help                           Display this help message
    --dir=<directory>                   Directory [default: results/Ra_1-40e+08_Ek_1-00e-05_Pr_7-0_N_128_q_1-2_k_48_enhanced/]
    --scale=<Scale>                     The conversion ratio between the old and new resolution [default: 1]
"""

from docopt import docopt
import numpy as np
from dedalus import public as de
import os 
import h5py 
import matplotlib.pyplot as plt

# =============================================================================
# Extract the docopt arguments 
# =============================================================================

args = docopt(__doc__)
dataDirectory = str(args['--dir'])
scale = float(str(args['--scale']))

## Some variables ##

Gamma = 2

# =============================================================================
# Extract the snapshot filename 
# =============================================================================

for idx,lines in enumerate(os.listdir(dataDirectory)):
    if lines.find('snapshot') != -1:
        snapshot_file_name = os.listdir(dataDirectory)[idx]

myData = h5py.File('{}{}/last-time-step.h5'.format(dataDirectory, snapshot_file_name), mode = 'r')
sim_time = np.copy(myData['sim_time'])
u = np.copy(myData['u'])
v = np.copy(myData['v'])
w = np.copy(myData['w'])
p = np.copy(myData['p'])
T = np.copy(myData['T'])
Tz = np.copy(myData['Tz'])
uz = np.copy(myData['uz'])
vz = np.copy(myData['vz'])
wz = np.copy(myData['wz'])
myData.close()

oldShape = np.shape(u)
oldHorizontalResolution = oldShape[0]
verticalResolution = oldShape[2]

print('\n'+'-'*80)
print('The conversion scale is: {}, the new x and y resolution is: {}, please ensure this is an interger.'.format(scale, scale*oldHorizontalResolution))
print('-'*80 + '\n')


x_basis = de.Fourier('x', oldHorizontalResolution, interval = (0, Gamma), dealias = 3/2)
y_basis = de.Fourier('y', oldHorizontalResolution, interval = (0, Gamma), dealias = 3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)


def convert_resolution(data):
     
    newArray = []
    for n in range(verticalResolution):
        field = domain.new_field()
        field['g'] = np.take(data, indices = n, axis = 2)
        field.set_scales(scale)
        newArray.append(field['g'])
    return np.rot90(newArray, k = 1, axes = (0,2))

convertedT = convert_resolution(T)
print(np.average(convertedT))
print(np.average(T))


hf = h5py.File('{}{}/last-time-step-{}.h5'.format(dataDirectory,snapshot_file_name, scale), 'w')
hf.create_dataset('u', data = convert_resolution(u))
hf.create_dataset('v', data = convert_resolution(v))
hf.create_dataset('w', data = convert_resolution(w))
hf.create_dataset('T', data = convert_resolution(T))
hf.create_dataset('p', data = convert_resolution(p))
hf.create_dataset('uz', data = convert_resolution(uz))
hf.create_dataset('vz', data = convert_resolution(vz))
hf.create_dataset('wz', data = convert_resolution(wz))
hf.create_dataset('Tz', data = convert_resolution(Tz))
hf.create_dataset('sim_time', data = sim_time)

hf.close()





