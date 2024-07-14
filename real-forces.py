"""Analysis of forces file of 3d rotating rayleigh benard convection

Usage:
    forces_spectrums.py [--dir=<directory>] [--snap_t=<snapshot_transient>] [--mask=<mask>] [--t=<transient> --fig=<Figure>]
    forces_spectrums.py -h | --help

Options:
    -h --help                           Display this help message
    --dir=<directory>                   Directory [default: results/Ra_1-40e+08_Ek_1-00e-05_Pr_7-0_N_128_q_1-2_k_48_enhanced/]
    --snap_t=<transient>                Snapshot transient [default: 2]
    --mask=<mask>                       Number of viscous boundaries to ignore [default: 0] 
    --t=<transient>                     Transient to be ignored [default: 2000]
    --fig=<Figure>                      Produce Figures [default: False]
"""

from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy import stats
from scipy.fft import fft
from dedalus import public as de
from dedalus.core.operators import Integrate 
from colours import *

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
Nh = 192
Nz = 192
Gamma = 1
printMidPoint = False

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
ACoriolisProfileACColour = CB91_Blue
spectrumlw = 2

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
# All my functions which I use through out the script
# =============================================================================

def plotASlice(Field):
    """ 
    Parameters
    ----------
    Field : Dedalus Field object

    Raises
    ------
    Must be working with 3D data.

    Returns
    -------
    None.
    
    Notes
    -----
    Creates several plots looking at the length scales in the given field.
    """

    realFieldData = Field['g']
    fieldName = Field.name
    midPoint = np.shape(realFieldData)[-1] // 2

    if len(np.shape(realFieldData)) != 3:
        raise Exception('Expecting 3d data.')
    
    slice = np.sum(realFieldData, axis = 2)
    sliceXAverage = np.average(slice, axis = 0)
    sliceYAverage = np.average(slice, axis = 1)
    horizontalDomain = np.linspace(0, 2, len(sliceYAverage)) 

    fig,ax = plt.subplots(2,2, figsize=(10,10))
    ax[0][0].imshow(slice, cmap = 'coolwarm')
    ax[0][0].set_title('Average {} Field'.format(fieldName))
    ax[1][0].plot(horizontalDomain, sliceXAverage, lw = spectrumlw, color = CB91_Blue)
    ax[1][0].set_title('X Average')   
    ax[0][1].plot(horizontalDomain, sliceYAverage, lw = spectrumlw, color = CB91_Violet)
    ax[0][1].set_title('Y Average')
    spectraX = (fft(sliceXAverage).real**2 + fft(sliceXAverage).imag**2)**0.5
    spectraY = (fft(sliceYAverage).real**2 + fft(sliceYAverage).imag**2)**0.5
    midPoint = len(spectraX) // 2
    ax[1][1].plot(spectraX[:midPoint], lw = spectrumlw, color = CB91_Blue)
    ax[1][1].plot(spectraY[:midPoint], lw = spectrumlw, color = CB91_Violet) 
    ax[1][1].set_title('Spectra')
    plt.savefig('{}/img/plotASlice.eps'.format(dir), dpi=500)
    plt.show()
    
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

def plotMidPlane(Field):
    """ 
    Parameters
    ----------
    Field : Dedalus Field object

    Raises
    ------
    Must be working with 3D data.

    Returns
    -------
    None.
    
    Notes
    -----
    Created a imshow plot of the mid section of the domain.
    """

    realFieldData = Field['g']
    fieldName = Field.name
    midPoint = np.shape(realFieldData)[-1] // 2

    if len(np.shape(realFieldData)) != 3:
        raise Exception('Expecting 3d data.')
    
    rotatedData = np.rot90(realFieldData, k=1, axes = (2,0))
    midPointSlice = rotatedData[midPoint]
    fig = plt.figure(figsize = (6,6))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(midPointSlice, cmap = 'bwr', interpolation = 'gaussian')
    plt.show()
    
def computeRMS(Fx, Fy, Fz):
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

    return np.sqrt(Fx**2 + Fy**2 + Fz**2)


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

def horizontalAverage(ForceRMS):
    """ 
    

    Parameters
    ----------
    ForceRMS : Root mean square of the force.
ffns
    -------yVorticityViscositySpectrum,
    The averaged of each force over x and y, leaving a 1d profile against z. 
    As done in (Guzman, 2021).

    """
    profile = []
    rotatedForce = np.rot90(ForceRMS, k=1, axes=(2,0))
 
    for lines in rotatedForce:
        profile.append(np.average(lines))

    return np.array(profile)


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

    RotatedForce = np.rot90(Force['g'], k=1, axes = (2,0))
    MaskedForce = RotatedForce*Mask
    Force['g'] = np.rot90(MaskedForce,	k=-1, axes = (2,0))

def integrate(field, p):
    return field.integrate()['g'][0,0,0] / p

def curlHelper(Fx, Fy, Fz):

    ## Create a domain ##
    Nx,Ny,Nz = np.shape(Fx)
    Lx = Ly = 2
    Lz = 1

    x_basis = de.Fourier('x', Nx, interval = (0,Lx), dealias=1)
    y_basis = de.Fourier('y', Ny, interval = (0,Ly), dealias=1)
    z_basis = de.Chebyshev('z', Nz, interval = (-Lz/2,Lz/2), dealias=1) 
    domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

    forceXField = domain.new_field(name='Fx')
    forceYField = domain.new_field(name='Fy')
    forceZField = domain.new_field(name='Fz')
    
    forceXField['g'] = Fx
    forceYField['g'] = Fy
    forceZField['g'] = Fz
    
    dFzdy = y_basis.Differentiate(forceZField).evaluate()
    dFydz = z_basis.Differentiate(forceYField).evaluate()
    dFzdx = x_basis.Differentiate(forceZField).evaluate()
    dFxdz = z_basis.Differentiate(forceXField).evaluate()
    dFydx = x_basis.Differentiate(forceYField).evaluate()
    dFxdy = y_basis.Differentiate(forceXField).evaluate()
    
    return dFzdy - dFydz, dFxdz - dFzdx, dFydx - dFxdy


def curl(Fx, Fy, Fz):

    ## Data comes in as numpy data set with [T,x,y,z] ##
    numberOfTimeSteps = np.shape(Fx)[0]
    OxArray = []
    OyArray = []
    OzArray = []
    for idx in range(numberOfTimeSteps):
        Ox, Oy, Oz = curlHelper(Fx[idx], Fy[idx], Fz[idx])
        OxArray.append(Ox['g'])
        OyArray.append(Oy['g'])
        OzArray.append(Oz['g'])
    return OxArray, OyArray, OzArray

def generateMask(nToRemove, blThickness, z):

    Mask = np.rot90(np.ones(np.shape(x_diffusion[-1]),dtype=np.int32), k=1 , axes=(2,0))
    layerToRemove = (blThickness * nToRemove) - 0.5
    percentageOfDomain = 1-(blThickness * nToRemove * 2)
    for idx,lines in enumerate(z):
        if lines > layerToRemove:
            avg_points = idx - 1
            
            break

    for idx,slices in enumerate(Mask):
        if idx < avg_points or idx >= len(z) - avg_points:
            Mask[idx] = np.zeros(np.shape(slices))

    return Mask, percentageOfDomain

# =============================================================================
# Load the data from the snapshot data file
# =============================================================================

with h5py.File('{}{}/snapshots.h5'.format(dir,snapshot_file_name), mode = 'r') as file:

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
    y_inertia = np.copy(file['tasks']["y_inertia"])[-snap_t:,:,:,:]

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
    try:
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
        vorticity_y_inertia =   np.copy(file['tasks']["vorticity_y_inertia"])[-snap_t:,:,:,:]

        # =========================================================================
        # Vorticity Z equation
        # =========================================================================

        vorticity_z_diffusion = np.copy(file['tasks']["vorticity_z_diffusion"])[-snap_t:,:,:,:]
        vorticity_z_inertia = np.copy(file['tasks']["vorticity_z_inertia"])[-snap_t:,:,:,:]
        vorticity_z_coriolis = np.copy(file['tasks']["vorticity_z_coriolis"])[-snap_t:,:,:,:]
    
    except:
        vorticity_x_diffusion, vorticity_y_diffusion, vorticity_z_diffusion = curl(x_diffusion,y_diffusion,z_diffusion)
        vorticity_x_coriolis, vorticity_y_coriolis, vorticity_z_coriolis = curl(x_coriolis, y_coriolis, np.zeros(np.shape(x_coriolis)))
        vorticity_x_inertia, vorticity_y_inertia, vorticity_z_inertia = curl(x_inertia, y_inertia, z_inertia)
        vorticity_x_bouyancy, vorticity_y_bouyancy, vorticity_z_bouyancy = curl(np.zeros(np.shape(x_coriolis)), np.zeros(np.shape(x_coriolis)), z_buoyancy)

# =============================================================================
# Create a dedalus domain which is the same as the domain of the problem so
# we can use in-built dedalus functions to preform integration and compute
# spectra.
# =============================================================================

Nx = Ny = Nh
Lx = Ly = Gamma
Lz = 1

x_basis = de.Fourier('x', Nx, interval = (0,Lx), dealias=1)
y_basis = de.Fourier('y', Ny, interval = (0,Ly), dealias=1)
z_basis = de.Chebyshev('z', Nz, interval = (-Lz/2,Lz/2), dealias =1)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

# =============================================================================
# Create fields in dedalus so we can use dedalus tools for computations, 
# retain spectral accuracy e.c.t.
# =============================================================================

# =============================================================================
# Momentum Equation
# =============================================================================

Viscosity = domain.new_field(name='viscosity')
Coriolis = domain.new_field(name='coriolis')
ACoriolis = domain.new_field(name='Acoriolis')
Inertia = domain.new_field(name='inertia')
Buoyancy = domain.new_field(name='buoyancy')
Pressure = domain.new_field(name='pressure')

vViscosity = domain.new_field(name='vviscosity')
vCoriolis = domain.new_field(name='vcoriolis')
vInertia = domain.new_field(name='vinertia')
vBuoyancy = domain.new_field(name='vbuoyancy')

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

Mask, percentageOfDomain = generateMask(nToRemove = mask, blThickness = avg_viscous_boundary, z = z)

# =============================================================================
# Arrays containing the time series for the horizontal average
# =============================================================================

horizontalAvgViscosityTimeSeries = []
horizontalAvgCoriolisTimeSeries = []
horizontalAvgBuoyancyTimeSeries = []
horizontalAvgInertiaTimeSeries = []
horizontalAvgPressureTimeSeries = []
horizontalAvgACoriolisTimeSeries = []

horizontalAvgvViscosityTimeSeries = []
horizontalAvgvCoriolisTimeSeries = []
horizontalAvgvBuoyancyTimeSeries = []
horizontalAvgvInertiaTimeSeries = []

integratedPressure = []
integratedCoriolis = []
integratedViscosity = []
integratedInertia = []
integratedBuoyancy = []
integratedACoriolis = []

integratedvCoriolis = []
integratedvViscosity = []
integratedvInertia = []
integratedvBuoyancy = []

blankMatrix = np.zeros(np.shape(z_diffusion[-1]), dtype=np.int32)

for idx in range(1,snap_t+1):

    # =============================================================================
    # Load data
    # =============================================================================
    
    Viscosity['g'] = computeRMS(x_diffusion[-idx], y_diffusion[-idx], z_diffusion[-idx])
    Coriolis['g'] = computeRMS(x_coriolis[-idx], y_coriolis[-idx], blankMatrix)
    Inertia['g'] = computeRMS(x_inertia[-idx], y_inertia[-idx], z_inertia[-idx])
    Buoyancy['g'] = computeRMS(blankMatrix, blankMatrix, removeMeanProfile(z_buoyancy[-idx]))
    Pressure['g'] = computeRMS(removeMeanProfile(x_pressure[-idx]), removeMeanProfile(y_pressure[-idx]), removeMeanProfile(z_pressure[-idx]))
    ACoriolis['g'] = computeRMS(x_pressure[-idx] + x_coriolis[-idx], y_pressure[-idx] + y_coriolis[-idx], z_pressure[-idx] - blankMatrix)

    vViscosity['g'] = computeRMS(vorticity_x_diffusion[-idx], vorticity_y_diffusion[-idx], vorticity_z_diffusion[-idx])
    vCoriolis['g'] = computeRMS(vorticity_x_coriolis[-idx], vorticity_y_coriolis[-idx], vorticity_z_coriolis[-idx])
    vInertia['g'] = computeRMS(vorticity_x_inertia[-idx], vorticity_y_inertia[-idx], vorticity_z_inertia[-idx])
    vBuoyancy['g'] = computeRMS(vorticity_x_bouyancy[-idx], vorticity_y_bouyancy[-idx], blankMatrix)

    # =========================================================================
    # Remove the boundaries
    # =========================================================================
    
    removeBoundaries(Mask, Viscosity)
    removeBoundaries(Mask, Pressure)
    removeBoundaries(Mask, Coriolis)
    removeBoundaries(Mask, Inertia)
    removeBoundaries(Mask, ACoriolis)
    removeBoundaries(Mask, Buoyancy)

    removeBoundaries(Mask, vViscosity)    
    removeBoundaries(Mask, vCoriolis)
    removeBoundaries(Mask, vInertia)
    removeBoundaries(Mask, vBuoyancy)

    integratedPressure.append(integrate(Pressure, p = percentageOfDomain))
    integratedViscosity.append(integrate(Viscosity, p = percentageOfDomain))
    integratedInertia.append(integrate(Inertia, p = percentageOfDomain))
    integratedBuoyancy.append(integrate(Buoyancy, p = percentageOfDomain))
    integratedCoriolis.append(integrate(Coriolis, p = percentageOfDomain))
    integratedACoriolis.append(integrate(ACoriolis, p = percentageOfDomain))

    integratedvViscosity.append(integrate(vViscosity, p = percentageOfDomain))
    integratedvInertia.append(integrate(vInertia, p = percentageOfDomain))
    integratedvBuoyancy.append(integrate(vBuoyancy, p = percentageOfDomain))
    integratedvCoriolis.append(integrate(vCoriolis, p = percentageOfDomain))



    # =========================================================================
    # Compute the horizontal average
    # =========================================================================
    
    horizontalAvgViscosityTimeSeries.append(abs(horizontalAverage(Viscosity['g'])))
    horizontalAvgCoriolisTimeSeries.append(abs(horizontalAverage(Coriolis['g'])))
    horizontalAvgInertiaTimeSeries.append(abs(horizontalAverage(Inertia['g'])))
    horizontalAvgBuoyancyTimeSeries.append(abs(horizontalAverage(Buoyancy['g'])))
    horizontalAvgPressureTimeSeries.append(abs(horizontalAverage(Pressure['g'])))
    horizontalAvgACoriolisTimeSeries.append(abs(horizontalAverage(ACoriolis['g'])))

    horizontalAvgvViscosityTimeSeries.append(abs(horizontalAverage(vViscosity['g'])))
    horizontalAvgvCoriolisTimeSeries.append(abs(horizontalAverage(vCoriolis['g'])))
    horizontalAvgvInertiaTimeSeries.append(abs(horizontalAverage(vInertia['g'])))
    horizontalAvgvBuoyancyTimeSeries.append(abs(horizontalAverage(vBuoyancy['g'])))


# =============================================================================
# Time avg the profiles
# =============================================================================

ViscosityProfile = np.average(np.array(horizontalAvgViscosityTimeSeries), axis=0)
InertiaProfile = np.average(np.array(horizontalAvgInertiaTimeSeries), axis=0)
BuoyancyProfile = np.average(np.array(horizontalAvgBuoyancyTimeSeries), axis=0)
CoriolisProfile = np.average(np.array(horizontalAvgCoriolisTimeSeries), axis=0)
PressureProfile = np.average(np.array(horizontalAvgPressureTimeSeries), axis=0)
ACoriolisProfile = np.average(np.array(horizontalAvgACoriolisTimeSeries), axis=0)

vViscosityProfile = np.average(np.array(horizontalAvgvViscosityTimeSeries), axis=0)
vInertiaProfile = np.average(np.array(horizontalAvgvInertiaTimeSeries), axis=0)
vBuoyancyProfile = np.average(np.array(horizontalAvgvBuoyancyTimeSeries), axis=0)
vCoriolisProfile = np.average(np.array(horizontalAvgvCoriolisTimeSeries), axis=0)

mid_point = int(Nz/2)

if printMidPoint == True:
    print('-'*120)
    print('Mid Points ')
    print('-'*120)
    print('Viscosity: {:.3e}, Coriolis: {:.3e}, Inertia: {:.3e}, Buoyancy: {:.3e}, Pressure: {:.3e}, Ageo Coriolis: {:.3e}.'.format(ViscosityProfile[mid_point], CoriolisProfile[mid_point], InertiaProfile[mid_point], BuoyancyProfile[mid_point], PressureProfile[mid_point], ACoriolisProfile[mid_point]))
    print('-'*120)
    print('Vorticity: Viscosity: {:.3e}, Coriolis: {:.3e}, Inertia: {:.3e}, Buoyancy: {:.3e}.'.format(vViscosityProfile[mid_point], vCoriolisProfile[mid_point], vInertiaProfile[mid_point], vBuoyancyProfile[mid_point]))
    print('-'*120)

print('-'*120)
print('Forces -> Viscosity: {:.3e}, Coriolis: {:.3e}, Inertia: {:.3e}, Buoyancy: {:.3e}, Pressure: {:.3e}, Ageo Coriolis: {:.3e}'.format(np.average(integratedViscosity), np.average(integratedCoriolis), np.average(integratedInertia), np.average(integratedBuoyancy), np.average(integratedPressure), np.average(integratedACoriolis)))
print('Vorticity -> Viscosity: {:.3e}, Coriolis: {:.3e}, Inertia: {:.3e}, Buoyancy: {:.3e}'.format(np.average(integratedvViscosity), np.average(integratedvCoriolis), np.average(integratedvInertia), np.average(integratedvBuoyancy)))
