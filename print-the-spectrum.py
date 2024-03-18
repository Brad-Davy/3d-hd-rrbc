""" 
   A script which plots the forces, vorticity and kinetic energy as a function of kh.

Usage:
    plot-the-spectrum.py [--dir=<directory>]
    plot-the-spectrum.py -h | --help

Options:
    -h --help            Display this help message
    --dir=<directory>    Directory [default: results/Ra_1-40e+08_Ek_1-00e-05_Pr_7-0_N_128_]
"""

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from colours import *

vorticity = False

##################################
## Defining plotting parameters ##
##################################

plt.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1
plt.rcParams["figure.autolayout"] = True
cm = 1/2.54
BuoyancyColour = CB91_Green
ViscosityColour = CB91_Pink
InertiaColour = 'maroon'
CoriolisColour = CB91_Violet
PressureColour = 'darkorange'
ACColour = CB91_Blue
spectrumlw = 1

###############
## Functions ##
###############

def convertStringArray(data : str) -> list:
    data = data.replace('[',' ')
    data = data.replace(']', ' ')
    array = []
    for i in data.split(' '):
        try:
           array.append(float(i))
        except:
           pass
    return array

def returnArrayFromTxTFile(file : str) -> list:
    with open(dir+'img/'+file, 'r') as file:
         return np.array(convertStringArray(file.read()))

def findLargest(spectrums : list) -> float:
    largestValues = []
    for lines in spectrums:
        largestValues.append(max(lines))
    return max(largestValues)

def printFiles(fileName, STDfileName):

    print('\n')
    print('+'*70)  
    print('+'*20 + str(fileName) + '+'*20)
    print('+'*70)
    print(list(returnArrayFromTxTFile(fileName)))
    print('\n')
    print('+'*70)
    print('+'*20 + str(STDfileName) + '+'*20)
    print('+'*70)
    print(list(returnArrayFromTxTFile(STDfileName)))
    print('\n')

###########################
## Read in the data file ##
###########################

args = docopt(__doc__)
dir = str(args['--dir'])

##############################
## Deal with kinetic energy ##
##############################

kineticEnergy = returnArrayFromTxTFile('kinetic.txt')
kineticEnergySTD = returnArrayFromTxTFile('kineticSTD.txt')
waveNumbers = range(len(kineticEnergy))
fig = plt.figure(figsize=(9*cm, 9*cm))
plt.plot(waveNumbers, kineticEnergy)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Kinetic Energy')
plt.xlabel('$k_h$')
plt.savefig(dir+'img/kineticEnergy.svg', dpi=500)
perpindicularLengthScale = np.argmax(kineticEnergy) + 1
print('The peak of the kinetic spectrum is: {}.'.format(perpindicularLengthScale))

######################
## Deal with forces ##
######################

ViscositySpectrum = returnArrayFromTxTFile('viscosity.txt')
CoriolisSpectrum = returnArrayFromTxTFile('coriolis.txt')
InertiaSpectrum = returnArrayFromTxTFile('inertia.txt')
BuoyancySpectrum = returnArrayFromTxTFile('buoyancy.txt')
PressureSpectrum = returnArrayFromTxTFile('pressure.txt')
ACoriolisSpectrum = returnArrayFromTxTFile('ageocor.txt')

fig = plt.figure(figsize=(9*cm, 9*cm))
largestForce = findLargest([ViscositySpectrum, CoriolisSpectrum, InertiaSpectrum, BuoyancySpectrum, PressureSpectrum, ACoriolisSpectrum])
plt.plot(range(len(ViscositySpectrum)), ViscositySpectrum / largestForce, label = '$F_v$', color = ViscosityColour, lw=spectrumlw)
plt.plot(range(len(ViscositySpectrum)), CoriolisSpectrum / largestForce, label = '$F_C$', color = CoriolisColour, lw=spectrumlw)
plt.plot(range(len(InertiaSpectrum)), InertiaSpectrum / largestForce, label = '$F_I$', color = InertiaColour, lw=spectrumlw)
plt.plot(range(len(BuoyancySpectrum)), BuoyancySpectrum / largestForce, label = '$F_B$', color = BuoyancyColour, lw=spectrumlw)
plt.plot(range(len(BuoyancySpectrum)), PressureSpectrum / largestForce, label = '$F_P$', color = PressureColour, lw=spectrumlw)
plt.plot(range(len(ViscositySpectrum)), ACoriolisSpectrum / largestForce, label = '$F_{AC}$', color = ACColour, lw=spectrumlw, linestyle = 'dashed')
plt.xscale("log")
plt.yscale("log")
plt.xlabel('$K_h$')
plt.ylabel('$|F|$')
plt.legend(ncol=2, frameon=False)
plt.savefig('{}/img/ForceSpectrum.svg'.format(dir), dpi=500)
plt.show()

#########################
## Deal with vorticity ##
#########################

vorticityViscositySpectrum = returnArrayFromTxTFile('voritcityViscosity.txt')
vorticityCoriolisSpectrum = returnArrayFromTxTFile('vorticityCoriolis.txt')
vorticityInertiaSpectrum = returnArrayFromTxTFile('vorticityInertia.txt')
vorticityBuoyancySpectrum = returnArrayFromTxTFile('vorticityBuoyancy.txt')

fig = plt.figure(figsize=(9*cm, 9*cm))
largestVorticity = findLargest([vorticityViscositySpectrum, vorticityCoriolisSpectrum, vorticityInertiaSpectrum, vorticityBuoyancySpectrum])
plt.plot(range(len(ViscositySpectrum)), vorticityViscositySpectrum / largestVorticity, label = '$\\omega_v$', color = ViscosityColour, lw=spectrumlw)
plt.plot(range(len(ViscositySpectrum)), vorticityCoriolisSpectrum / largestVorticity, label = '$\\omega_C$', color = CoriolisColour, lw=spectrumlw)
plt.plot(range(len(InertiaSpectrum)), vorticityInertiaSpectrum / largestVorticity, label = '$\\omega_I$', color = InertiaColour, lw=spectrumlw)
plt.plot(range(len(BuoyancySpectrum)), vorticityBuoyancySpectrum / largestVorticity, label = '$\\omega_B$', color = BuoyancyColour, lw=spectrumlw)
plt.xscale("log")
plt.yscale("log")
plt.xlabel('$K_h$')
plt.ylabel('$| \\omega |$')
plt.ylim(0.1*(min(vorticityCoriolisSpectrum) / largestVorticity), 5)
plt.legend(ncol=2, frameon=False)
plt.savefig('{}/img/vorticityForceSpectrum.eps'.format(dir), dpi=500)
plt.show()


#############################################
## Deal with local non-dimensional numbers ##
#############################################

Re = vorticityInertiaSpectrum / vorticityViscositySpectrum
Ek = vorticityViscositySpectrum / vorticityCoriolisSpectrum
Ro = vorticityInertiaSpectrum / vorticityCoriolisSpectrum
print('Re = {:.4e}, Ek = {:.4e}, Ro = {:.4e}'.format(Re[perpindicularLengthScale], Ek[perpindicularLengthScale], Ro[perpindicularLengthScale]))

###########################
## Print the text files ##
##########################

if vorticity == True:
    printFiles('voritcityViscosity.txt', 'voritcityViscositySTD.txt')
    printFiles('vorticityBuoyancy.txt', 'vorticityBuoyancySTD.txt')
    printFiles('vorticityCoriolis.txt', 'vorticityCoriolisSTD.txt') 
    printFiles('vorticityInertia.txt', 'vorticityInertiaSTD.txt')
else:
    printFiles('viscosity.txt', 'viscositySTD.txt')
    printFiles('buoyancy.txt', 'buoyancySTD.txt')
    printFiles('coriolis.txt', 'coriolisSTD.txt') 
    printFiles('inertia.txt', 'inertiaSTD.txt')
    printFiles('pressure.txt', 'pressureSTD.txt')
    printFiles('ageocor.txt.txt', 'ageocorSTD.txt')
