"""Analysis file of 3d rotating rayleigh benard convection

Usage:
    plot-txt-file.py --dir=<directory> [--t=<transient> --snap_t=<snap_t>]
    plot-txt-file.py -h | --help

Options:
    -h --help   		  Display this help message
    --dir=<directory>             Directory
    --t=<transient>               Transient to be ignored [default: 2000]
    --snap_t=<snap_t>             Snapshot interval [default: 1000] 
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
transient = int(args['--t'])
snapshotInterval = int(args['--snap_t'])

# =============================================================================
# Create empty arrays to contain the run data over time 
# =============================================================================

ThermalBoundary = []
ViscousBoundary = []
Points_in_thermal_boundary = []
Points_in_viscous_Boundary = []
run = 1
Peclet = []
Nu_integral = []
Time = []
Nu_Error = []
Re = []
Nu_top = []
Nu_bottom = []
Nu_midplane = []
u_max = []
v_max = []
w_max = []
Buoyancy = []
Dissipation = []
Energy_Balance = []
D_viscosity = []
upper_thermal_boundary = []
memory = []

# =============================================================================
# Define plotting parameters
# =============================================================================

lineWidth = 2
figSize = (18,8)

# =============================================================================
# Extract run data from the file name 
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
# Find all files in the run directory
# =============================================================================

onlyFiles = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

# =============================================================================
# Find the log file and create a variable which stores its path
# =============================================================================

for lines in onlyFiles:

    if lines.find('.txt') != -1:
        logFile = dir + '/' + lines

log_file = open(logFile, 'r')
contents = log_file.read().split('\n')

# =============================================================================
# Check if there is a img directory and if there isnt then create one 
# =============================================================================

if os.path.isdir(dir+'/img') == True:
    pass
else:
    os.system('mkdir {}/img'.format(dir))
    
# =============================================================================
# Load the arrays up with the appropriate data for plotting 
# =============================================================================    

for idx,lines in enumerate(contents):
    if idx != 0 or idx != 1:
        try:
            dataFromFile = lines.split()
            Time.append(float(dataFromFile[0]))
            Re.append(float(dataFromFile[1]))
            Nu_top.append(float(dataFromFile[2]))
            Nu_bottom.append(float(dataFromFile[3]))
            Nu_midplane.append(float(dataFromFile[4]))
            Nu_integral.append(float(dataFromFile[5]))
            u_max.append(float(dataFromFile[6]))
            v_max.append(float(dataFromFile[7]))
            w_max.append(float(dataFromFile[8]))
            Buoyancy.append(float(dataFromFile[9]))
            Dissipation.append(float(dataFromFile[10]))
            Energy_Balance.append(float(dataFromFile[11]))
            Nu_Error.append(float(dataFromFile[12]))
            D_viscosity.append(float(dataFromFile[13]))
            memory.append(float(memoryUsage))
        except:
            pass
        
print('-'*60)
print('The estimated Nusselt number: {:.4f}'.format(np.average(Nu_integral[-50:])))
print('The estimated Peclet number: {:.4f}'.format(np.average(Re[-500:])*7))
print('The length of the .txt files is: {}'.format(len(u_max)))

# =============================================================================
# Determine where the snapshots were taken so this can be visualised on the plots   
# =============================================================================
    
Time = np.array(Time)
iterations = len(Time)*10
convectiveTimeSteps = Time * np.average(Re[:-100]) * 7 

print('The total time ran for in convective over turn time is: {:.2f}.'.format(convectiveTimeSteps[-1]))
print('-'*60)
# =============================================================================
# Create and save the plots 
# =============================================================================

print('The output is set to plot, creating figures in directory {}img.'.format(dir))
Nusselt_fig, ax1 = plt.subplots(figsize = figSize)
ax1.plot(convectiveTimeSteps, Nu_integral, label = '$Nu_I$', color = CB91_Blue, lw = lineWidth)
ax1.plot(convectiveTimeSteps, Nu_bottom, label = '$Nu_b$', color = CB91_Violet, lw = lineWidth)
ax1.plot(convectiveTimeSteps, Nu_top, label = '$Nu_t$', color = CB91_Amber, lw = lineWidth)
ax2 = ax1.twiny()
ax2.plot(np.linspace(0, iterations, len(Nu_integral)), Nu_integral, label = '$Nu_I$', color = 'gray', lw = lineWidth, linestyle = 'dashed')
ax1.set_xlabel('Convective Overturn Time')
plt.ylabel('Nusselt Number')

ax1.legend(frameon = False, ncol = 2)
plt.savefig('{}img/Nusselt.eps'.format(dir), dpi = 500)

with open('{}img/nusselt.txt'.format(dir), 'w') as kineticFile:
    kineticFile.write(str(Nu_integral))

with open('{}img/time.txt'.format(dir), 'w') as kineticFile:
    kineticFile.truncate(0)
        
with open('{}img/time.txt'.format(dir), 'a') as kineticFile:
    for lines in convectiveTimeSteps:
        kineticFile.write(str(lines)+'\n')

Velocity_fig = plt.figure(figsize = figSize)
plt.plot(convectiveTimeSteps, u_max, label = 'u', color = CB91_Blue, lw = lineWidth)
plt.plot(convectiveTimeSteps, v_max, label = 'v', color = CB91_Violet, lw = lineWidth)
plt.plot(convectiveTimeSteps, w_max, label = 'w', color = CB91_Amber, lw = lineWidth)
plt.legend(frameon = False, ncol = 2)
plt.xlabel('Convective Overturn Time')
plt.savefig('{}img/Max_Velocity.eps'.format(dir), dpi = 500)

Buoyancy_dissipation_fig = plt.figure(figsize = figSize)
plt.plot(convectiveTimeSteps, Buoyancy, label = 'Buoyancy', lw = lineWidth, color = CB91_Blue)
plt.plot(convectiveTimeSteps, Dissipation, label = 'Dissipation', lw = lineWidth, color = CB91_Amber)
plt.legend(frameon = False, ncol = 2)
plt.xlabel('Convective Overturn Time')
plt.savefig('{}img/Buoyancy_Dissipation.eps'.format(dir), dpi = 500)

Re_fig = plt.figure(figsize = figSize)
plt.plot(convectiveTimeSteps, Re, lw = lineWidth, color = CB91_Blue)
plt.xlabel('Convective Overturn Time')
plt.ylabel('Reynolds Number')
plt.savefig('{}img/Reynolds.eps'.format(dir), dpi = 500)

D_viscosity_fig = plt.figure(figsize = figSize)
plt.plot(convectiveTimeSteps, D_viscosity, lw = lineWidth, color = CB91_Blue)
plt.xlabel('Convective Overturn Time')
plt.ylabel('Viscous Dissipation')                  
plt.show()
plt.savefig('{}img/D_viscosity.eps'.format(dir), dpi = 500)
