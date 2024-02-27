""" 
   A script which looks at how the forces in rotating rayleigh benard 
   convection changes as a function of the length scale.

Usage:
    forces_spectrums.py [--dir=<directory>] [--snap_t=<snapshot_transient>] [--mask=<mask>] [--t=<transient> --fig=<Figure> --calc_all=<Calculate_All> --g=<Gamma>] 
    forces_spectrums.py -h | --help

Options:
    -h --help                           Display this help message
    --dir=<directory>                   Directory [default: results/Ra_1-40e+08_Ek_1-00e-05_Pr_7-0_N_128_q_1-2_k_48_enhanced/]
    --snap_t=<transient>                Snapshot transient [default: 5]
    --mask=<mask>                       Number of viscous boundaries to ignore [default: 1] 
    --t=<transient>                     Transient to be ignored [default: 2000]
    --fig=<Figure>                      Produce Figures [default: False]
    --calc_all=<Calculate_All>          Calculate all spectra [default: 0]
    --g=<Gamma>				Apsect ration [default:2]
"""

from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy import stats
from numpy.fft import fftn
from dedalus import public as de
from dedalus.core.operators import Integrate 
from colours import *
import matplotlib
matplotlib.use('Agg')

# =============================================================================
# Extract the docopt arguments 
# =============================================================================

args = docopt(__doc__)
dir = str(args['--dir'])
transient = int(args['--t'])
fig_bool = bool(args['--fig'])
transient = int(args['--t'])
snap_t = int(args['--snap_t'])
mask = int(args['--mask'])
calculateAll = int(args['--calc_all'])
gamma = float(args['--g'])

# =============================================================================
# Set some plotting parameters for later 
# =============================================================================

plt.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1

BuoyancyColour = CB91_Green
ViscosityColour = CB91_Pink
InertiaColour = 'maroon'
CoriolisColour = CB91_Violet
PressureColour = 'darkorange'
ACColour = CB91_Blue
spectrumlw = 1

# =============================================================================
# Create the string which leads to the right directory
# =============================================================================

for idx,lines in enumerate(os.listdir(dir)):
    if lines.find('analysis') != -1:
        forces_file_name = os.listdir(dir)[idx]

for idx,lines in enumerate(dir.split('/')[1].split('_')):
    if idx == 1:
        Ra = float(lines.replace('-','.'))
    if idx == 3:
        a,b,c,d,e,f,g,h = lines
        Ek = float(a+'.'+c+d+'e-'+g+h)
    if idx == 5:
        Pr = float(lines.replace('-','.'))
    if idx == 7:
        N = int(lines)
        
# =============================================================================
# Check the calculate all argument
# =============================================================================
    
# calculateAll = calculateAll == 1
        
# =============================================================================
# Extract some profiles from the analysis file, these are used later
# =============================================================================

with h5py.File('{}{}/analysis.h5'.format(dir,forces_file_name), mode = 'r') as file:

    z = np.copy(file['tasks']["z"])[0,0,0,:]
    U_h = np.copy(file['tasks']['U_H_prof'])
    
# =============================================================================
# Check to see if there is an img directory, if not then create one.
# =============================================================================

if os.path.isdir(dir+'/img') == True:
    pass
else:
    os.system('mkdir {}/img'.format(dir))

# =============================================================================
# Extract the snapshot filename 
# =============================================================================

for idx,lines in enumerate(os.listdir(dir)):
    if lines.find('snapshot') != -1:
        snapshot_file_name = os.listdir(dir)[idx]

# =============================================================================
# Load the data from the snapshot data file
# =============================================================================

with h5py.File('{}{}/new_snapshots_file.h5'.format(dir,snapshot_file_name), mode = 'r') as file:

    # =========================================================================
    # Velocity data
    # =========================================================================
    
    u = np.copy(file['tasks']["u"])[-snap_t:,:,:,:]
    v = np.copy(file['tasks']["v"])[-snap_t:,:,:,:]
    w = np.copy(file['tasks']["w"])[-snap_t:,:,:,:]
    T = np.copy(file['tasks']["T"])[-snap_t:,:,:,:]

    # =========================================================================
    # Momentum X equationnp.sum(realFieldData, axis = 2)
    # =========================================================================
    
    x_pressure = np.copy(file['tasks']["x_pressure"])[-snap_t:,:,:,:]
    x_diffusion = np.copy(file['tasks']["x_diffusion"])[-snap_t:,:,:,:]
    x_coriolis = np.copy(file['tasks']["x_coriolis"])[-snap_t:,:,:,:]
    x_inertia = np.copy(file['tasks']["x_inertia"])[-snap_t:,:,:,:]

    # =========================================================================
    # Momentum Y equation
    # =========================================================================
    
    y_pressure = np.copy(file['tasks']["y_pressure"])[-snap_t:,:,:,:]
    y_diffusion = np.copy(file['tasks']["y_diffusion"])[-snap_t:,:,:,:]
    y_coriolis = np.copy(file['tasks']["y_coriolis"])[-snap_t:,:,:,:]
    y_inertia =	np.copy(file['tasks']["y_inertia"])[-snap_t:,:,:,:]

    # =========================================================================
    # Momentum Z equation
    # =========================================================================
    
    z_pressure = np.copy(file['tasks']["z_pressure"])[-snap_t:,:,:,:]
    z_diffusion = np.copy(file['tasks']["z_diffusion"])[-snap_t:,:,:,:]
    z_inertia = np.copy(file['tasks']["z_inertia"])[-snap_t:,:,:,:]
    z_buoyancy = np.copy(file['tasks']["z_bouyancy"])[-snap_t:,:,:,:]
    
    # =========================================================================
    # Vorticity X equation
    # =========================================================================
    
    vorticity_x_diffusion = np.copy(file['tasks']["vorticity_x_diffusion"])[-snap_t:,:,:,:]
    vorticity_x_coriolis = np.copy(file['tasks']["vorticity_x_coriolis"])[-snap_t:,:,:,:]
    vorticity_x_bouyancy = np.copy(file['tasks']["vorticity_x_bouyancy"])[-snap_t:,:,:,:]
    vorticity_x_inertia = np.copy(file['tasks']["vorticity_x_inertia"])[-snap_t:,:,:,:]

    # =========================================================================
    # Vorticity Y equation
    # =========================================================================
    
    vorticity_y_diffusion = np.copy(file['tasks']["vorticity_y_diffusion"])[-snap_t:,:,:,:]
    vorticity_y_coriolis = np.copy(file['tasks']["vorticity_y_coriolis"])[-snap_t:,:,:,:]
    vorticity_y_bouyancy = np.copy(file['tasks']["vorticity_y_bouyancy"])[-snap_t:,:,:,:]
    vorticity_y_inertia =	np.copy(file['tasks']["vorticity_y_inertia"])[-snap_t:,:,:,:]

    # =========================================================================
    # Vorticity Z equation
    # =========================================================================
    
    vorticity_z_diffusion = np.copy(file['tasks']["vorticity_z_diffusion"])[-snap_t:,:,:,:]
    vorticity_z_inertia = np.copy(file['tasks']["vorticity_z_inertia"])[-snap_t:,:,:,:]
    vorticity_z_coriolis = np.copy(file['tasks']["vorticity_z_coriolis"])[-snap_t:,:,:,:]
    

print('-'*60)
print('All data loaded in.')
print('-'*60)

# =============================================================================
# All my functions which I use through out the script
# =============================================================================

def findIntersection(arrOne, arrTwo):
    """ 
    

    Parameters
    ----------
    arrOne : Spectra one.
    arrTwo : Spectra two.

    Returns
    -------
    Returns the intersection between two lines by considering the difference 
    between the two and returning where this line crosses 0. If statement 
    catches the case where the lines dont intersect.

    """
    
    difference = arrOne - arrTwo
    minimumDifferenceIndex = np.argmin(difference)
    
    signChange = np.where(np.diff(np.sign(difference)) != 0)[0] + 1
    
    if len(signChange) > 0:
        m,b = np.polyfit([signChange[0] - 1, signChange[0]] , [difference[signChange[0] - 1], difference[signChange[0]]], 1)
        return -b/m, abs(b)

    return 1,1

def compareSpectrum(Field):
    """ 
    

    Parameters
    ----------
    Field : TA dedalus field object

    Returns
    -------
    None.
    
    Notes
    -----
    Integrates the field in real space and then computes a summation of the 
    spectrum to show that they are indeed the same.

    """
    
    integratedRealField = Field.integrate()['g'][0][0][0]
    spectrum = Field['c']
    runningSum = 0
    realZerothSpectrum = spectrum[0][0][:].real
 
    for idx,lines in enumerate(realZerothSpectrum):
         
        if idx == 0:
            runningSum += 2*spectrum[0][0][idx]

        elif idx % 2 == 0:
            runningSum -= 2*spectrum[0][0][idx] / (idx**2 - 1)  
        else:
           pass

    print(2*runningSum, integratedRealField)

def multiplyFields(field):
    """ 
    

    Parameters
    ----------
    field : A field of complex coefficients.

    Returns
    -------
    multipliedField: A field full of real numbers.

    """
    multipliedField = np.ones_like(field)
    
    for i in range(np.shape(field)[0]):
        for j in range(np.shape(field)[1]):
            for k in range(np.shape(field)[2]):
                multipliedField[i][j][k] = field[i][j][k].real**2 + field[i][j][k].imag**2

    return multipliedField

def computeRMS(Fx, Fy, Fz, N = 128):
    """ 
    

    Parameters
    ----------
    Fx : FORCE IN X.
    Fy : FORCE IN Y.
    Fz : FORCE IN Z.

    Returns
    -------
    Removes any mean profile from the flow and returns the RMS of this 
    new, meanless, field.

    """

    Nx = N
    Ny = N 
    Lx = Ly = 1
    Lz = 1
    Nz = int(N/Lx)

    x_basis = de.Fourier('x', Nx, interval = (0,Lx), dealias=3/2)
    y_basis = de.Fourier('y', Ny, interval = (0,Ly), dealias=3/2)
    z_basis = de.Chebyshev('z', Nz, interval = (-Lz/2,Lz/2), dealias =3/2) 
    domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)
    
    forceXField = domain.new_field(name='Fx')
    forceYField = domain.new_field(name='Fy')
    forceZField = domain.new_field(name='Fz')
    
    forceXField['g'] = Fx
    forceYField['g'] = Fy
    forceXField['g'] = Fz

    FFTofforceXField = forceXField['c']
    FFTofforceYField = forceYField['c']
    FFTofforceZField = forceZField['c']

    return multiplyFields(FFTofforceXField) + multiplyFields(FFTofforceYField) + multiplyFields(FFTofforceZField)

def removeMeanProfile(Force):
    """ 
    

    Parameters
    ----------
    Force : FORCE.

    Returns
    -------
    Removes any mean profile found in the flow (Force).
    """
    
    rotatedForce = np.rot90(Force, k = 1, axes = (2,0))
    meanProfileOfForce = [np.average(planes) for planes in rotatedForce]
    meanProfile3DArray = np.ones_like(rotatedForce)
    
    for idx,plane in enumerate(meanProfile3DArray):
        meanProfile3DArray[idx] = meanProfile3DArray[idx]*meanProfileOfForce[idx]

    return np.rot90(rotatedForce - meanProfile3DArray, k = -1, axes = (2,0)) 

def createSurfacePlot(FFTofForce):
    """ 
    

    Parameters
    ----------
    FFToFForce : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    forceSpectrum = FFTofForce['c'].real
    z_avg = np.sum(forceSpectrum, axis = 2) 
    plt.imshow(np.log10(z_avg))
    return forceSpectrum

def plotHorizontalWaveNumber(FFTofForce):
    """ 
    

    Parameters
    ----------
    FFToFForce : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    forceSpectrum = FFTofForce['c'].real
    Lx = Ly = 2
    sumOverZ = np.sum(forceSpectrum, axis = 2) 
    
    Nmax = max(np.shape(sumOverZ)[0], np.shape(sumOverZ)[1])
    horizontalAveragedSpectra = []
    for p in range(1, Nmax):
        
        temporarySum = 0
        
        for kx in range(np.shape(sumOverZ)[0]):
            for ky in range(np.shape(sumOverZ)[1]):
                
                if p - 1 < (kx**2 + ky**2)**0.5 <= p:
                    temporarySum += sumOverZ[kx][ky]
                    
        
        horizontalAveragedSpectra.append(temporarySum)
    
    plt.plot(range(1,  Nmax),horizontalAveragedSpectra)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
def computeKineticEnergy(u, v, w, N = 128):
    """ 
    

    Parameters
    ----------
    u : x component of velocity
    v : y component of velocity
    w : z component of velocity
    

    Returns
    -------
    Integrated kinetic spectrum.
    
    Notes
    -----
    Integrates the field in real space and then computes a summation of the 
    spectrum to show that they are indeed the same.

    """
    Nx = N
    Ny = N 
    Lx = Ly = 1
    Lz = 1
    Nz = int(N/Lx)

    x_basis = de.Fourier('x', Nx, interval = (0,Lx), dealias = 3/2)
    y_basis = de.Fourier('y', Ny, interval = (0,Ly), dealias = 3/2)
    z_basis = de.Chebyshev('z', Nz, interval = (-Lz/2,Lz/2), dealias = 3/2) 
    domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)
    
    kineticEnergy = domain.new_field(name='kinetic')
    kineticEnergy['g'] = (u**2 + v**2 + w**2) 
    
    
    return kineticEnergy.integrate()['g'][0][0][0]
    #return np.sum(kineticEnergy['g'])/(128**3)
        
def plotTotalWaveNumber(u,v,w):
    """ 
    

    Parameters
    ----------
    FFToFForce : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    
    Notes
    -----
    Creates a plot of how some paticular quantity changes as a function of 
    length scales, as such the spectra is summed over shells between
    k and k+1. Where k = (kx^2 + ky^2 + kz^2)^0.5. Note that it is expected that
    this should be a squared field such that imaginary parts are 0.

    """

    u = u.real**2 + u.imag**2
    v = v.real**2 + v.imag**2
    w = w.real**2 + w.imag**2
    
    kineticEnergy = u + v + w
    visited = np.zeros_like(kineticEnergy)
    
    Lx = Ly = 2
    
    horizontalAveragedSpectra = [kineticEnergy[0][0][0]]
    Nmax = int((np.shape(kineticEnergy)[0]**2 + np.shape(kineticEnergy)[1]**2 + np.shape(kineticEnergy)[2]**2)**0.5) + 1
    
    for p in range(1, Nmax):
        
        temporarySum = 0
        index = 0
        
        for kx in range(np.shape(kineticEnergy)[0]):
            for ky in range(np.shape(kineticEnergy)[1]):
                for kz in range(np.shape(kineticEnergy)[2]):
                
                    if p - 1 < (kx**2 + ky**2 + kz**2)**0.5 <= p:
                        temporarySum += kineticEnergy[kx][ky][kz]
                        visited[kx][ky][kz] = 1
                        index += 1
                
        horizontalAveragedSpectra.append(temporarySum)
    
    plt.plot(range(Nmax)[64:],horizontalAveragedSpectra[64:])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    print('Total Sum: {:.3e}'.format(np.sum(kineticEnergy)))
    return horizontalAveragedSpectra, visited
    
def computeSpectrum(FFTofForce):
    """ 
    

    Parameters
    ----------
    Force : Field object from dedalus.

    Returns
    -------
    The spectrum of the force averaged over two directions.

    """
    
    forceSpectrum = FFTofForce['c'].real
    Lx = Ly = 2
    sumOverZ = np.sum(forceSpectrum, axis = 2) 
    
    Nmax = max(np.shape(sumOverZ)[0], np.shape(sumOverZ)[1])
    horizontalAveragedSpectra = []
    for p in range(1, Nmax):
        
        temporarySum = 0
        
        for kx in range(np.shape(sumOverZ)[0]):
            for ky in range(np.shape(sumOverZ)[1]):
                
                if p - 1 < (kx**2 + ky**2)**0.5 <= p:
                    temporarySum += sumOverZ[kx][ky]
                    
        
        horizontalAveragedSpectra.append(temporarySum)
    
    return horizontalAveragedSpectra

def computeFromNegativeSpectrum(negativeForce):
    """ 
    

    Parameters
    ----------
    negativeForce : A force in coefficient space containing negative numbers

    Returns
    -------
    The dependence of field on a paticular wavenumber.

    """

    forceSpectrum = (negativeForce['c'].real**2 + negativeForce['c'].imag**2)**0.5
    Lx = Ly = 2
    sumOverZ = np.sum(forceSpectrum, axis = 2) 
    
    Nmax = max(np.shape(sumOverZ)[0], np.shape(sumOverZ)[1])
    horizontalAveragedSpectra = []
    for p in range(1, Nmax):
        
        temporarySum = 0
        index = 0
        
        for kx in range(np.shape(sumOverZ)[0]):
            for ky in range(np.shape(sumOverZ)[1]):
                
                if p - 1 < (kx**2 + ky**2)**0.5 <= p:
                    temporarySum += sumOverZ[kx][ky]
                    index += 1
                    
        
        horizontalAveragedSpectra.append(temporarySum)
    
    return horizontalAveragedSpectra

def computeHorizontalSpectraFromRealSpace(Fx, Fy, Fz, N = N, Gamma = gamma):
    
    Nx = Ny = N
    Nz = int(N/Gamma)
    Lx = Gamma
    Ly = Gamma
    Lz = 1

    print('The input shape of the data is: {}, {}, {}.'.format(np.shape(Fx), np.shape(Fy), np.shape(Fz)))    
    x_basis = de.Fourier('x', Nx, interval = (0, Lx))
    y_basis = de.Fourier('y', Ny, interval = (0, Ly))
    z_basis = de.Chebyshev('z', Nz, interval = (-1/2,1/2))
    domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)
    
    realField = domain.new_field(name='realField')
    coefficientField = domain.new_field(name='coefficientField')
    
    fft2Fx = fftn(Fx, axes = (0,1))
    fft2Fy = fftn(Fy, axes = (0,1))
    fft2Fz = fftn(Fz, axes = (0,1))
    
    coefficientField['g'] = (fft2Fx.real**2 + fft2Fx.imag**2 + fft2Fy.real**2 + fft2Fy.imag**2 + fft2Fz.real**2 + fft2Fz.imag**2) / (Nx*Ny)
    realField['g'] = Fx**2 + Fy**2 + Fz**2
    
    realIntegral = (1/(Lx*Ly*Lz))*realField.integrate()['g'][0,0,0]
    zIntegralCoefficient = coefficientField.integrate('z')['g'][:,:,0] / (Nx*Ny)
    zSumCoefficient = np.sum(coefficientField['g'], axis=2) / (Nx*Ny)
    sumOfSpectra = np.sum(zSumCoefficient) 
    halfNx = Nx // 2

    newfield = zIntegralCoefficient[:,:halfNx] + np.flip(zIntegralCoefficient[:,halfNx:],1)
    newfield = newfield[:halfNx,:] + np.flip(newfield[halfNx:,:],0)
    horizontalSpectra = [zSumCoefficient[0,0]]

    print('-'*60)
    print('Spectra {:.3e}'.format(np.sum(zIntegralCoefficient)))
    print('Real: {:.3e}'.format(realIntegral))
    print('Difference: {:.3f}'.format(np.sum(newfield)/realIntegral))
    print('-'*60)

    for k in range(1, halfNx):
         temporarySum = 0 
         for kx in range(Nx):
             for ky in range(Ny):
 
                 if k - 1 < np.sqrt(kx**2 + ky**2) <= k:
                     temporarySum += newfield[kx, ky]
 
         horizontalSpectra.append(temporarySum)
 
    return horizontalSpectra

def computeHorizontalSpectraViscousDissipation(u, v, w, Fx, Fy, Fz, N = N, Gamma = gamma):

    Nx = Ny = N
    Nz = int(N/Gamma)
    Lx = Gamma
    Ly = Gamma
    Lz = 1

    print('The input shape of the data is: {}, {}, {}.'.format(np.shape(Fx), np.shape(Fy), np.shape(Fz)))
    x_basis = de.Fourier('x', Nx, interval = (0, Lx))
    y_basis = de.Fourier('y', Ny, interval = (0, Ly))
    z_basis = de.Chebyshev('z', Nz, interval = (-1/2,1/2))
    domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

    realField = domain.new_field(name='realField')
    coefficientField = domain.new_field(name='coefficientField')

    ## Diffusion FFT ##
    fft2Fx = fftn(Fx, axes = (0,1))
    fft2Fy = fftn(Fy, axes = (0,1))
    fft2Fz = fftn(Fz, axes = (0,1))

    ## Velocity FFT ##
    fft2U = fftn(u, axes=(0,1))
    fft2V = fftn(v, axes=(0,1))
    fft2W = fftn(w, axes=(0,1))

    coefficientField['g'] = (fft2Fx.real*fft2U.real + fft2Fx.imag*fft2U.imag  + fft2Fy.real*fft2V.real + fft2Fy.imag*fft2V.imag + fft2Fz.real*fft2W.real + fft2Fz.imag*fft2W.imag) / (Nx*Ny)
    realField['g'] = -(Fx*u + Fy*v + Fz*w)

    realIntegral = (1/(Lx*Ly*Lz))*realField.integrate()['g'][0,0,0]
    zIntegralCoefficient = coefficientField.integrate('z')['g'][:,:,0] / (Nx*Ny)
    zSumCoefficient = np.sum(coefficientField['g'], axis=2) / (Nx*Ny)
    sumOfSpectra = np.sum(zSumCoefficient) / (Nx*Ny)
    halfNx = Nx // 2

    newfield = zIntegralCoefficient[:,:halfNx] + np.flip(zIntegralCoefficient[:,halfNx:],1)
    newfield = newfield[:halfNx,:] + np.flip(newfield[halfNx:,:],0)
    horizontalSpectra = [zSumCoefficient[0,0]]

    print('-'*60)
    print('Viscous Dissipation')
    print('Spectra {:.3e}'.format(np.sum(zIntegralCoefficient)))
    print('Real: {:.3e}'.format(realIntegral))
    print('Difference: {:.3f}'.format(np.sum(newfield)/realIntegral))
    print('-'*60)

    for k in range(1, halfNx):
         temporarySum = 0
         for kx in range(Nx):
             for ky in range(Ny):

                 if k - 1 < np.sqrt(kx**2 + ky**2) <= k:
                     temporarySum += newfield[kx, ky]

         horizontalSpectra.append(temporarySum)

    return horizontalSpectra    

def derivative(data,z):
    """ 
    

    Parameters
    ----------
    data : 1d data.
    z : z domain array.

    Returns
    -------
    Derivative with respect to z.

    """
    return np.gradient(data,z)

def determineRoot(derivative,z):
    """ 
     

    Parameters
    ----------
    derivative : The derivative of a profile.
    z : The z domain.

    Returns
    -------
    upper : Upper boundarie.
    lower : Lower boundarie.
    avg : Average of the two.
    avg_points : Average number of points in both.

    """

    upper,lower,avg = 0,0,0
    zero_crossings = np.where(np.diff(np.sign(derivative)))[0]

    # =========================================================================
    # Determine the lower crossing 
    # =========================================================================
    
    x,y = [z[zero_crossings[0]],z[zero_crossings[0]+1]], [derivative[zero_crossings[0]], derivative[zero_crossings[0]+1]]
    m,b = np.polyfit(y,x,1)
    lower = b
    
    # =========================================================================
    # Determine the upper crossing 
    # =========================================================================
    
    x,y = [z[zero_crossings[-1]],z[zero_crossings[-1]+1]], [derivative[zero_crossings[-1]],derivative[zero_crossings[-1]+1]]
    m,b = np.polyfit(y,x,1)
    upper = b
    avg = (lower + (1-upper))/2

    # =========================================================================
    # Calculate the average number of points in the boundaries.
    # =========================================================================
    
    avg_points = zero_crossings[0]*mask

    return upper, lower, avg, avg_points


def removeBoundaries(Mask, Force):
    """ 
    

    Parameters
    ----------
    Mask : An array of ones and zeroes which removes the boundaries.
    Force : The force from which we want to remove the boundaries.

    Returns
    -------
    None: preformed in place.

    """

    rotatedForce = np.rot90(Force, k=1, axes = (2,0))
    maskedForce = rotatedForce*Mask
    return np.rot90(maskedForce,	k=-1, axes = (2,0))

# =============================================================================
# Avg the horizontal velocity profile over time
# Create an array which contains only the last N points, then avg over this.
# =============================================================================

avg_u_h = np.average(np.array(U_h[-transient:,0,0,:]), axis=0) 
upper_viscous_boundary, lower_viscous_boundary, avg_viscous_boundary, avg_points = determineRoot(derivative(avg_u_h,z),z)

# =============================================================================
# Construct a matrix which is 1 everywhere other than in the viscous boundary 
# where it is 0. This can be used to remove the viscous boundaries from my 
# force plots.
# =============================================================================

Mask = np.rot90(np.ones(np.shape(x_diffusion[-1]),dtype=np.int32), k=1 , axes=(2,0))

for idx,slices in enumerate(Mask):
    if idx < avg_points or idx >= len(z) - avg_points:
        Mask[idx] = np.zeros(np.shape(slices))

# =============================================================================
# Arrays containing the time series for the spectra
# =============================================================================

ViscosityTimeSeries = []
CoriolisTimeSeries = []
BuoyancyTimeSeries = []
InertiaTimeSeries = []
PressureTimeSeries = []
ACoriolisTimeSeries = []
KineticTimeSeries = []
UTimeSeries = []
VTimeSeries = []
WTimeSeries = []

vorticityViscosityTimeSeries = []
vorticityCoriolisTimeSeries = []
vorticityBuoyancyTimeSeries = []
vorticityInertiaTimeSeries = []

viscousDissipationTimeSeries = []

if calculateAll:

    xVorticityViscosityTimeSeries = []
    xVorticityCoriolisTimeSeries = []
    xVorticityBuoyancyTimeSeries = []
    xVorticityInertiaTimeSeries = []
    
    yVorticityViscosityTimeSeries = []
    yVorticityCoriolisTimeSeries = []
    yVorticityBuoyancyTimeSeries = []
    yVorticityInertiaTimeSeries = []
    
    xyVorticityViscosityTimeSeries = []
    xyVorticityCoriolisTimeSeries = []
    xyVorticityBuoyancyTimeSeries = []
    xyVorticityInertiaTimeSeries = []

blankMatrix = np.zeros(np.shape(z_diffusion[-1]), dtype=np.int32)

for idx in range(1,snap_t+1):

    # =============================================================================
    # Load data
    # =============================================================================
       
    ViscosityTimeSeries.append(computeHorizontalSpectraFromRealSpace(removeBoundaries(Mask, x_diffusion[-idx]), removeBoundaries(Mask, y_diffusion[-idx]), removeBoundaries(Mask, z_diffusion[-idx])))
    CoriolisTimeSeries.append(computeHorizontalSpectraFromRealSpace(removeBoundaries(Mask, x_coriolis[-idx]), removeBoundaries(Mask, y_coriolis[-idx]), blankMatrix))
    InertiaTimeSeries.append(computeHorizontalSpectraFromRealSpace(removeBoundaries(Mask, x_inertia[-idx]), removeBoundaries(Mask, y_inertia[-idx]), removeBoundaries(Mask, z_inertia[-idx])))
    BuoyancyTimeSeries.append(computeHorizontalSpectraFromRealSpace(blankMatrix, blankMatrix, z_buoyancy[-idx]))
    PressureTimeSeries.append(computeHorizontalSpectraFromRealSpace(removeBoundaries(Mask, x_pressure[-idx]), removeBoundaries(Mask, y_pressure[-idx]), removeBoundaries(Mask, z_pressure[-idx])))
    ACoriolisTimeSeries.append(computeHorizontalSpectraFromRealSpace(removeBoundaries(Mask, x_pressure[-idx] + x_coriolis[-idx]), removeBoundaries(Mask, y_pressure[-idx] + y_coriolis[-idx]), removeBoundaries(Mask, z_pressure[-idx])))
    KineticTimeSeries.append(computeHorizontalSpectraFromRealSpace(removeBoundaries(Mask, u[-idx]) , removeBoundaries(Mask, v[-idx]), removeBoundaries(Mask, w[-idx])))
        
    vorticityViscosityTimeSeries.append(computeHorizontalSpectraFromRealSpace(removeBoundaries(Mask, vorticity_x_diffusion[-idx]), removeBoundaries(Mask, vorticity_y_diffusion[-idx]), removeBoundaries(Mask, vorticity_z_diffusion[-idx])))
    vorticityCoriolisTimeSeries.append(computeHorizontalSpectraFromRealSpace(removeBoundaries(Mask, vorticity_x_coriolis[-idx]), removeBoundaries(Mask, vorticity_y_coriolis[-idx]), removeBoundaries(Mask, vorticity_y_coriolis[-idx])))
    vorticityInertiaTimeSeries.append(computeHorizontalSpectraFromRealSpace(removeBoundaries(Mask, vorticity_x_inertia[-idx]), removeBoundaries(Mask, vorticity_y_inertia[-idx]), removeBoundaries(Mask, vorticity_z_inertia[-idx])))
    vorticityBuoyancyTimeSeries.append(computeHorizontalSpectraFromRealSpace(removeBoundaries(Mask, vorticity_x_bouyancy[-idx]), removeBoundaries(Mask, vorticity_y_bouyancy[-idx]), blankMatrix))
    
    viscousDissipationTimeSeries.append(computeHorizontalSpectraViscousDissipation(u = removeBoundaries(Mask, u[-idx]), v = removeBoundaries(Mask, v[-idx]),w = removeBoundaries(Mask, w[-idx]), Fx = removeBoundaries(Mask, x_diffusion[-idx]), Fy = removeBoundaries(Mask, y_diffusion[-idx]), Fz = removeBoundaries(Mask, z_diffusion[-idx])))

    UTimeSeries.append(computeHorizontalSpectraFromRealSpace(removeBoundaries(Mask, u[-idx]), blankMatrix, blankMatrix))
    VTimeSeries.append(computeHorizontalSpectraFromRealSpace(removeBoundaries(Mask, v[-idx]), blankMatrix, blankMatrix))
    WTimeSeries.append(computeHorizontalSpectraFromRealSpace(removeBoundaries(Mask, w[-idx]), blankMatrix, blankMatrix))

# =============================================================================
# Time avg the spectrums 
# =============================================================================

ViscositySpectrum = np.average(np.array(ViscosityTimeSeries), axis=0)
InertiaSpectrum = np.average(np.array(InertiaTimeSeries), axis=0)
BuoyancySpectrum = np.average(np.array(BuoyancyTimeSeries), axis=0)
CoriolisSpectrum = np.average(np.array(CoriolisTimeSeries), axis=0)
PressureSpectrum = np.average(np.array(PressureTimeSeries), axis=0)
ACoriolisSpectrum = np.average(np.array(ACoriolisTimeSeries), axis=0)
KineticSpectrum = np.average(np.array(KineticTimeSeries), axis=0)
USpectrum = np.average(np.array(UTimeSeries), axis=0)
VSpectrum = np.average(np.array(VTimeSeries), axis=0)
WSpectrum = np.average(np.array(WTimeSeries), axis=0)
vorticityViscositySpectrum = np.average(np.array(vorticityViscosityTimeSeries), axis=0)
vorticityInertiaSpectrum = np.average(np.array(vorticityInertiaTimeSeries), axis=0)
vorticityBuoyancySpectrum = np.average(np.array(vorticityBuoyancyTimeSeries), axis=0)
vorticityCoriolisSpectrum = np.average(np.array(vorticityCoriolisTimeSeries), axis=0)
viscousDissipationSpectrum = np.average(np.array(viscousDissipationTimeSeries), axis=0)

# ============================================================================
# Standard Deviation
# ============================================================================

KineticSpectrumSTD = np.std(np.array(KineticTimeSeries), axis=0)
vorticityViscositySpectrumSTD = np.std(np.array(vorticityViscosityTimeSeries), axis=0)
vorticityInertiaSpectrumSTD = np.std(np.array(vorticityInertiaTimeSeries), axis=0)
vorticityBuoyancySpectrumSTD = np.std(np.array(vorticityBuoyancyTimeSeries), axis=0)
vorticityCoriolisSpectrumSTD = np.std(np.array(vorticityCoriolisTimeSeries), axis=0)
viscousDissipationSpectrumSTD = np.std(np.array(viscousDissipationTimeSeries), axis=0)

if calculateAll:

    xVorticityViscositySpectrum = np.average(np.array(xVorticityViscosityTimeSeries), axis=0)
    xVorticityInertiaSpectrum = np.average(np.array(xVorticityInertiaTimeSeries), axis=0)
    xVorticityBuoyancySpectrum = np.average(np.array(xVorticityBuoyancyTimeSeries), axis=0)
    xVorticityCoriolisSpectrum = np.average(np.array(xVorticityCoriolisTimeSeries), axis=0)
    
    yVorticityViscositySpectrum = np.average(np.array(yVorticityViscosityTimeSeries), axis=0)
    yVorticityInertiaSpectrum = np.average(np.array(yVorticityInertiaTimeSeries), axis=0)
    yVorticityBuoyancySpectrum = np.average(np.array(yVorticityBuoyancyTimeSeries), axis=0)
    yVorticityCoriolisSpectrum = np.average(np.array(yVorticityCoriolisTimeSeries), axis=0)
    
    xyVorticityViscositySpectrum = np.average(np.array(xyVorticityViscosityTimeSeries), axis=0)
    xyVorticityInertiaSpectrum = np.average(np.array(xyVorticityInertiaTimeSeries), axis=0)
    xyVorticityBuoyancySpectrum = np.average(np.array(xyVorticityBuoyancyTimeSeries), axis=0)
    xyVorticityCoriolisSpectrum = np.average(np.array(xyVorticityCoriolisTimeSeries), axis=0)

print('-'*60)
print('All spectra calculated.')
print('-'*60)

# =============================================================================
# Plotting the results
# =============================================================================

upperXLim = N // 2
def findLargest(arrays, N):
    largestValues = []
    for array in arrays:
        largestValues.append(max(array[0:N]))
    return max(largestValues)
        

fig = plt.figure(figsize=(10,10))
largestForce = findLargest([ViscositySpectrum, CoriolisSpectrum, InertiaSpectrum, BuoyancySpectrum, PressureSpectrum, ACoriolisSpectrum], upperXLim)
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
plt.xlim(1, upperXLim)



with open('{}img/viscosity.txt'.format(dir), 'w') as kineticFile:
    print('Viscosity file created')
    kineticFile.write(str(ViscositySpectrum))

with open('{}img/inertia.txt'.format(dir), 'w') as kineticFile:
    print('Inertia file created')
    kineticFile.write(str(InertiaSpectrum))

with open('{}img/coriolis.txt'.format(dir), 'w') as kineticFile:
    print('Coriolis file created')
    kineticFile.write(str(CoriolisSpectrum))

with open('{}img/buoyancy.txt'.format(dir), 'w') as kineticFile:
    print('Buoyancy file created')
    kineticFile.write(str(BuoyancySpectrum))

with open('{}img/pressure.txt'.format(dir), 'w') as kineticFile:
    print('Pressure file created')
    kineticFile.write(str(PressureSpectrum))

with open('{}img/ageocor.txt'.format(dir), 'w') as kineticFile:
    print('Ageo coriolis file created')
    kineticFile.write(str(ACoriolisSpectrum))

with open('{}img/voritcityViscosity.txt'.format(dir), 'w') as kineticFile:
    print('Viscosity file created')
    kineticFile.write(str(vorticityViscositySpectrum))

with open('{}img/vorticityInertia.txt'.format(dir), 'w') as kineticFile:
    print('Inertia file created')
    kineticFile.write(str(vorticityInertiaSpectrum))

with open('{}img/vorticityCoriolis.txt'.format(dir), 'w') as kineticFile:
    print('Coriolis file created')
    kineticFile.write(str(vorticityCoriolisSpectrum))

with open('{}img/vorticityBuoyancy.txt'.format(dir), 'w') as kineticFile:
    print('Buoyancy file created')
    kineticFile.write(str(vorticityBuoyancySpectrum))

with open('{}img/viscousDissipation.txt'.format(dir), 'w') as kineticFile:
    print('Dissipation file created')
    kineticFile.write(str(viscousDissipationSpectrum))

with open('{}img/viscousDissipationSTD.txt'.format(dir), 'w') as kineticFile:
    print('Dissipation file created')
    kineticFile.write(str(viscousDissipationSpectrumSTD))

with open('{}img/uSpectrum.txt'.format(dir), 'w') as kineticFile:
    print('U file created')
    kineticFile.write(str(USpectrum))

with open('{}img/vSpectrum.txt'.format(dir), 'w') as kineticFile:
    print('V file created')
    kineticFile.write(str(VSpectrum))

with open('{}img/wSpectrum.txt'.format(dir), 'w') as kineticFile:
    print('W file created')
    kineticFile.write(str(WSpectrum))

with open('{}img/voritcityViscositySTD.txt'.format(dir), 'w') as kineticFile:
    print('Viscosity file created')
    kineticFile.write(str(vorticityViscositySpectrumSTD))

with open('{}img/vorticityInertiaSTD.txt'.format(dir), 'w') as kineticFile:
    print('Inertia file created')
    kineticFile.write(str(vorticityInertiaSpectrumSTD))

with open('{}img/vorticityCoriolisSTD.txt'.format(dir), 'w') as kineticFile:
    print('Coriolis file created')
    kineticFile.write(str(vorticityCoriolisSpectrumSTD))

with open('{}img/vorticityBuoyancySTD.txt'.format(dir), 'w') as kineticFile:
    print('Buoyancy file created')
    kineticFile.write(str(vorticityBuoyancySpectrumSTD))


legend_properties = {'weight':'bold'}
plt.legend(ncol=2, fontsize=14,prop=legend_properties,frameon=False)
plt.savefig('{}/img/ForceSpectrum.eps'.format(dir), dpi=500)
plt.show()

fig = plt.figure(figsize=(10,10))
largestVorticity = findLargest([vorticityViscositySpectrum, vorticityCoriolisSpectrum, vorticityInertiaSpectrum, vorticityBuoyancySpectrum], upperXLim)
plt.plot(range(len(ViscositySpectrum)), vorticityViscositySpectrum / largestVorticity, label = '$\\omega_v$', color = ViscosityColour, lw=spectrumlw)
plt.plot(range(len(ViscositySpectrum)), vorticityCoriolisSpectrum / largestVorticity, label = '$\\omega_C$', color = CoriolisColour, lw=spectrumlw)
plt.plot(range(len(InertiaSpectrum)), vorticityInertiaSpectrum / largestVorticity, label = '$\\omega_I$', color = InertiaColour, lw=spectrumlw)
plt.plot(range(len(BuoyancySpectrum)), vorticityBuoyancySpectrum / largestVorticity, label = '$\\omega_B$', color = BuoyancyColour, lw=spectrumlw)
plt.xscale("log")
plt.yscale("log")
plt.xlabel('$K_h$')
plt.ylabel('$| \\omega |$')
plt.xlim(1, upperXLim)
plt.ylim(0.1*(min(vorticityCoriolisSpectrum) / largestVorticity), 5)
plt.legend(ncol=2, fontsize=14,prop=legend_properties,frameon=False)
plt.savefig('{}/img/vorticityForceSpectrum.eps'.format(dir), dpi=500)
plt.show()


fig = plt.figure(figsize=(8,8))
plt.plot(range(len(ViscositySpectrum)), KineticSpectrum, lw = 1, color = ViscosityColour, label = 'K.e')

with open('{}img/kinetic.txt'.format(dir), 'w') as kineticFile:
    kineticFile.write(str(KineticSpectrum))
    print('Kinetic file created')

with open('{}img/kineticSTD.txt'.format(dir), 'w') as kineticFile:
    kineticFile.write(str(KineticSpectrumSTD))
    print('Kinetic file created')


plt.xscale("log")
plt.yscale("log")
plt.xlabel('$K_h$')
plt.ylabel('K.E')
plt.xlim(1, upperXLim)
plt.savefig('{}/img/KineticSpectrum.eps'.format(dir), dpi=500)
plt.show()

if calculateAll:
    
    fig = plt.figure(figsize=(10,10))
    plt.plot(range(len(yVorticityViscositySpectrum)), yVorticityViscositySpectrum, label = '$\\omega_v$', color = ViscosityColour, lw=spectrumlw)
    plt.plot(range(len(yVorticityViscositySpectrum)), yVorticityCoriolisSpectrum, label = '$\\omega_C$', color = CoriolisColour, lw=spectrumlw)
    plt.plot(range(len(yVorticityViscositySpectrum)), yVorticityInertiaSpectrum, label = '$\\omega_I$', color = InertiaColour, lw=spectrumlw)
    plt.plot(range(len(yVorticityViscositySpectrum)), yVorticityBuoyancySpectrum, label = '$\\omega_B$', color = BuoyancyColour, lw=spectrumlw)
    plt.xscale("log")
    plt.title('Y - Vorticity Equation')
    plt.yscale("log")
    plt.xlabel('$K_h$')
    plt.ylabel('Magnitude')
    plt.xlim(1, upperXLim) 
    plt.ylim(min(yVorticityViscositySpectrum)*0.05, 2*max(yVorticityCoriolisSpectrum))
    plt.legend(ncol=2, fontsize=14,prop=legend_properties,frameon=False)
    plt.savefig('{}/img/yVorticityForceSpectrum.eps'.format(dir), dpi=500)
    plt.show()
    
    fig = plt.figure(figsize=(10,10))
    plt.plot(range(len(xyVorticityViscositySpectrum)), xyVorticityViscositySpectrum, label = '$\\omega_v$', color = ViscosityColour, lw=spectrumlw)
    plt.plot(range(len(xyVorticityViscositySpectrum)), xyVorticityCoriolisSpectrum, label = '$\\omega_C$', color = CoriolisColour, lw=spectrumlw)
    plt.plot(range(len(xyVorticityViscositySpectrum)), xyVorticityInertiaSpectrum, label = '$\\omega_I$', color = InertiaColour, lw=spectrumlw)
    plt.plot(range(len(xyVorticityViscositySpectrum)), xyVorticityBuoyancySpectrum, label = '$\\omega_B$', color = BuoyancyColour, lw=spectrumlw)
    plt.xscale("log")
    plt.title('X - Y Averaged Vorticity Equation')
    plt.yscale("log")
    plt.xlabel('$K_h$')
    plt.ylabel('Magnitude')
    plt.xlim(1, upperXLim)
    plt.ylim(min(xyVorticityViscositySpectrum)*0.05, 2*max(xyVorticityCoriolisSpectrum))
    plt.savefig('{}/img/xyVorticityForceSpectrum.eps'.format(dir), dpi=500)
    plt.show()
    
    fig = plt.figure(figsize=(10,10))
    plt.plot(range(len(xVorticityViscositySpectrum)), xVorticityViscositySpectrum, label = '$\\omega_v$', color = ViscosityColour, lw=spectrumlw)
    plt.plot(range(len(xVorticityViscositySpectrum)), xVorticityCoriolisSpectrum, label = '$\\omega_C$', color = CoriolisColour, lw=spectrumlw)
    plt.plot(range(len(xVorticityViscositySpectrum)), xVorticityInertiaSpectrum, label = '$\\omega_I$', color = InertiaColour, lw=spectrumlw)
    plt.plot(range(len(xVorticityViscositySpectrum)), xVorticityBuoyancySpectrum, label = '$\\omega_B$', color = BuoyancyColour, lw=spectrumlw)
    plt.xscale("log")
    plt.title('X - Vorticity Equation')
    plt.yscale("log")
    plt.xlabel('$K_h$')
    plt.ylabel('Magnitude')
    plt.xlim(1, upperXLim)
    plt.ylim(min(xVorticityViscositySpectrum)*0.05, 2*max(xVorticityCoriolisSpectrum))
    plt.savefig('{}/img/xVorticityForceSpectrum.eps'.format(dir), dpi=500)
    plt.show()
