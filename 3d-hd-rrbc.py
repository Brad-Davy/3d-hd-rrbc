""" Dedalus simulation of 3d Rayleigh benard rotating convection.

Usage:
    3d-hd-rrbc.py [--ra=<rayleigh>] [--ek=<ekman>] [--Nh=<resolution>] [--Nz=<resolution>] [--max_dt=<Maximum_dt>]  [--init_dt=<Initial_dt>] [--pr=<prandtl>] [--mesh=<mesh>] [--q=<q>] [--k_0=<k_0>] [--e_z=<enhanced_z>] [--seed=<Seed>] [--gamma=<Gamma>]
    3d-hd-rrbc.py -h | --help
Options:
    -h --help               Display this help message
    --ra=<rayliegh>         Rayleigh number [default: 3.3e5]
    --ek=<ekman>            Ekman number [default: 1e-3]
    --Nh=<resolution>       Horizontal resolution [default: 256]
    --Nz=<vertical>         Vertical resolution [default: 128]
    --max_dt=<Maximum_dt>   Maximum Time Step [default: 1e-3]
    --pr=<prandtl>          Prandtl number [default: 7]
    --mesh=<mesh>           Parallel mesh [default: None]
    --init_dt=<Initial_dt>  Initial Time Step [default: 1e-3]
    --q=<q>                 Hyperdiffusion parameter [default: 1.05]
    --k_0=<k_0>             Hyperdiffusion parameter [default: 8]
    --e_z=<enhanced_z>      Enhanced number of nodes in z  [default: 0]
    --seed=<Seed>           Seed for initial conditions [default: None]
    --gamma=<Gamma>         Aspect ratio [default: 2]
"""

from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time
from dedalus import public as de
from dedalus.extras import flow_tools
from docopt import docopt
import logging
import os
import h5py
logger = logging.getLogger(__name__)

# =============================================================================
# Setting up Docopt 
# =============================================================================

args = docopt(__doc__)
comm = MPI.COMM_WORLD

# =============================================================================
# Pull the number of spectral modes from the input 
# =============================================================================

Nh = int(args['--Nh'])
Nz = int(args['--Nz'])
gamma = float(args['--gamma'])
enhancedZ = int(args['--e_z'])
Nx = Ny = Nh

# =============================================================================
# Set the aspect ratio 
# =============================================================================

Lx = Ly = gamma
Lz = 1

# =============================================================================
# Pull the parameters from the input file
# =============================================================================

Rayleigh = float(args['--ra'])
Ekman = float(args['--ek'])
Prandtl = float(args['--pr'])
max_dt = float(args['--max_dt'])
init_dt = float(args['--init_dt'])
q = float(args['--q'])
k_0 = int(args['--k_0'])
seed = args['--seed']

# =============================================================================
# Format the mesh input
# =============================================================================

if args['--mesh']!="None":
    mesh = (int(args['--mesh'].split(',')[0]),int(args['--mesh'].split(',')[1]))
else:
    mesh=None

# =============================================================================
# Simulation interation values 
# =============================================================================

sim_end_time = 20
max_iterations = 30000
sim_wall_time = 24*60*60*24

# =============================================================================
# Set the file names 
# =============================================================================

if enhancedZ == 0:
    file_tag = "Ra_{:.2e}_Ek_{:.2e}_Pr_{}_N_{}_q_{}_k_{}_".format(Rayleigh,Ekman,Prandtl,N,q,k_0)

elif enhancedZ == 1:
    file_tag = "Ra_{:.2e}_Ek_{:.2e}_Pr_{}_N_{}_q_{}_k_{}_enhanced".format(Rayleigh,Ekman,Prandtl,N,q,k_0)

file_tag = file_tag.replace(".","-")


# =============================================================================
# Make the directory to save the output files
# =============================================================================

if comm.rank==0:
   os.system('mkdir results/{}'.format(file_tag))

# =============================================================================
# Set up the geometry of the run
# =============================================================================
print(Nx,Ny,Nz)
start_init_time = time.time()
x_basis = de.Fourier('x', Nx, interval = (0,Lx), dealias=3/2, hyp = q, cutoff = k_0)
y_basis = de.Fourier('y', Ny, interval = (0,Ly), dealias=3/2, hyp = q, cutoff = k_0)
z_basis = de.Chebyshev('z', Nz, interval = (-Lz/2,Lz/2), dealias = 3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, comm=comm, mesh=mesh)

# =============================================================================
# Parameters
# =============================================================================

problem = de.IVP(domain, variables = ['p','T','u','v','w','Tz','uz','vz','wz'])
problem.meta['p','T','u','v','w']['z']['dirichlet']=True
problem.parameters['Ra'] = Rayleigh
problem.parameters['Ek'] = Ekman
problem.parameters['Pr'] = Prandtl
problem.parameters['Lx'] = Lx
problem.parameters['Ly'] = Ly
problem.parameters['Lz'] = Lz

# =============================================================================
# Substitutions to make vorticity equation easier to code
# =============================================================================
problem.substitutions['w_1'] = " (dy(w) - vz)"
problem.substitutions['w_2'] = "-(dx(w) - uz)"
problem.substitutions['w_3'] = "dx(v) - dy(u)"
problem.substitutions['w_1_z'] = "dz(w_1)"
problem.substitutions['w_2_z'] = "dz(w_2)"
problem.substitutions['w_3_z'] = "dz(w_3)"

# =============================================================================
# Governing Equations
# =============================================================================

problem.add_equation("dx(u) + dy(v) + wz = 0")
problem.add_equation("dt(T) - (dx(dx(T)) + dy(dy(T)) + dz(Tz)) = w -(u*dx(T) + v*dy(T) + w*Tz)")
problem.add_equation("dt(u) + dx(p) - Pr*(hdx(hdx(u)) + hdy(hdy(u)) + dz(uz)) - (Pr/Ek)*v  = -(u*dx(u) + v*dy(u) + w*uz)") ## Note that a 2 has been added to the coriolis term to match (Schmitz et al, 2010) take this out of further runs 
problem.add_equation("dt(v) + dy(p) - Pr*(hdx(hdx(v)) + hdy(hdy(v)) + dz(vz)) + (Pr/Ek)*u  = -(u*dx(v) + v*dy(v) + w*vz)") ## Note that a 2 has been added to the coriolis term to match (Schmitz et al, 2010) take this out of further runs
problem.add_equation("dt(w) + dz(p) - Pr*(hdx(hdx(w)) + hdy(hdy(w)) + dz(wz)) - Ra*Pr*T = -(u*dx(w) + v*dy(w) +w*wz)")

problem.add_equation("Tz - dz(T) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")

# =============================================================================
# Boundary conditions 
# =============================================================================

problem.add_bc("left(T) = 0")
problem.add_bc("left(u)= 0")
problem.add_bc("left(v)= 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(T) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("right(v) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("integ_z(p) = 0", condition="(nx == 0)")

# =============================================================================
# Build the solver
# =============================================================================

solver = problem.build_solver("RK443")
logger.info('Solver built')

# =============================================================================
# Initial Conditions 
# =============================================================================

z = domain.grid(2)
T = solver.state['T']
Tz = solver.state['Tz']
u = solver.state['u']
v = solver.state['v']
w = solver.state['w']
uz = solver.state['uz']
vz = solver.state['vz']
wz = solver.state['wz']


if seed == 'None':

    # =============================================================================
    # Random perturbations, initialized globally for same results in parallel 
    # =============================================================================

    gshape = domain.dist.grid_layout.global_shape(scales=1) # This is the shape of the domain
    slices = domain.dist.grid_layout.slices(scales=1)
    rand = np.random.RandomState(seed=23)
    noise = rand.standard_normal(gshape)[slices]
    pert = 1e-4 * noise
    T['g'] =  pert
    T.differentiate('z',out=Tz)


else:
    with h5py.File('{}'.format(seed), mode = 'r') as file:

        seedUField = np.copy(file['u'])
        seedVField = np.copy(file['v'])
        seedWField = np.copy(file['w'])
        seedTemperatureField = np.copy(file['T'])
        seedUzField = np.copy(file['uz'])
        seedVzField = np.copy(file['vz'])
        seedWzField = np.copy(file['wz'])
        seedTzField = np.copy(file['Tz'])
        sim_start_time = 0 #np.copy(file['sim_time'])[-1]

    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)

    if gshape[0] != np.shape(seedTemperatureField)[0]:
        raise Exception('Make sure the seed field and new field match in dimensions.')

    else:
        T['g'] = seedTemperatureField[slices]
        u['g'] = seedUField[slices]
        v['g'] = seedVField[slices]
        w['g'] = seedWField[slices]
        uz['g'] = seedUzField[slices]
        vz['g'] = seedVzField[slices]
        wz['g'] = seedWzField[slices]
        Tz['g'] = seedTzField[slices]
        solver.sim_time = 0

# =============================================================================
# Setting the simulation duration
# =============================================================================

solver.stop_sim_time = sim_end_time
solver.stop_wall_time = sim_wall_time
solver.stop_iteration = max_iterations

# =============================================================================
# Snapshots of the entire domain
# =============================================================================

snap = solver.evaluator.add_file_handler('results/{}/{}snapshots'.format(file_tag,file_tag), iter = 3000, max_writes = 20)
snap.add_system(solver.state)

# =============================================================================
# Compute each terms seperatley, i.e. F_x, F_y, F_z and compute rms as 
# (Guzman, 2021) do.
# =============================================================================

# =============================================================================
# Momentum x-equation
# =============================================================================

snap.add_task("-dx(p)", name = 'x_pressure')
snap.add_task("Pr*(hdx(hdx(u)) + hdy(hdy(u)) + dz(uz))", name = 'x_diffusion')
snap.add_task("(Pr/Ek)*v", name = 'x_coriolis')
snap.add_task("-(u*dx(u) + v*dy(u) + w*uz)", name = 'x_inertia')

# =============================================================================
# Momentum y-equation
# =============================================================================

snap.add_task("-dy(p)", name = 'y_pressure')
snap.add_task("Pr*(hdx(hdx(v)) + hdy(hdy(v)) + dz(vz))", name = 'y_diffusion')
snap.add_task("-(Pr/Ek)*u", name = 'y_coriolis')
snap.add_task("-(u*dx(v) + v*dy(v) + w*vz)", name = 'y_inertia')

# =============================================================================
# Momentum z-equation
# =============================================================================

snap.add_task("-dz(p)", name = 'z_pressure')
snap.add_task("Pr*(hdx(hdx(w)) + hdy(hdy(w)) + dz(wz))", name = 'z_diffusion')
snap.add_task("-(u*dx(w) + v*dy(w) + w*wz)", name = 'z_inertia')
snap.add_task("Ra*Pr*T", name = 'z_bouyancy')


# =============================================================================
# Dealing with the curl of the equations to exclude pressure and terms balancing
# with pressure, i.e. the vorticity equation.
# =============================================================================


# =============================================================================
# Vorticity x-equation
# =============================================================================

snap.add_task("Pr*(hdx(hdx(w_1)) + hdy(hdy(w_1)) + dz(w_1_z))", name = 'vorticity_x_diffusion')
snap.add_task("-(Pr/Ek)*uz", name = 'vorticity_x_coriolis')
snap.add_task("Ra*Pr*dy(T)", name = 'vorticity_x_bouyancy')
snap.add_task("-(u*dx(w_1) + v*dy(w_1) + w*w_1_z)", name = 'vorticity_x_inertia')

# =============================================================================
# Vorticity y-equation
# =============================================================================

snap.add_task("Pr*(hdx(hdx(w_2)) + hdy(hdy(w_2)) + dz(w_2_z))", name = 'vorticity_y_diffusion')
snap.add_task("-(Pr/Ek)*vz", name = 'vorticity_y_coriolis')
snap.add_task("-Ra*Pr*dx(T)", name = 'vorticity_y_bouyancy')
snap.add_task("-(u*dx(w_2) + v*dy(w_2) + w*w_2_z)", name = 'vorticity_y_inertia')

# =============================================================================
# Vorticity z-equation
# =============================================================================

snap.add_task("Pr*(hdx(hdx(w_3)) + hdy(hdy(w_3)) + dz(w_3_z))", name = 'vorticity_z_diffusion')
snap.add_task("-(u*dx(w_3) + v*dy(w_3) + w*w_3_z)", name = 'vorticity_z_inertia')
snap.add_task("(Pr/Ek)*wz", name = 'vorticity_z_coriolis')

# =============================================================================
# Dedalus analysis files containing integral properties of the system
# =============================================================================

analysis = solver.evaluator.add_file_handler('results/{}/{}analysis'.format(file_tag,file_tag),iter=5, max_writes=np.inf)
analysis.add_task("sqrt((1/(Lz*Lx*Ly))*integ(u*u + v*v + w*w))", name = "Pe")
analysis.add_task("(1/Pr)*sqrt((1/(Lx*Ly*Lz))*integ(u*u + v*v + w*w))", name = "Re")
analysis.add_task("interp((1/(Ly))*integ( (1/Lx)*integ( -dz(T) + w*T ,'x'),'y') ,z=0.0)", name = 'nu_mid_plane')
analysis.add_task("(1/(Lz*Lx*Ly))*integ( -dz(T) + w*T)", name = 'nu_integral')
analysis.add_task("interp( (1/(Ly))*integ( (1/Lx)*integ(-dz(T) ,'x'),'y') , z=-0.5)", name = 'nu_bot_wall')
analysis.add_task("interp( (1/(Ly))*integ( (1/Lx)*integ(-dz(T) ,'x'),'y') , z=0.5)", name = 'nu_top_wall')
analysis.add_task("(1/Lx)*integ((1/Ly)*integ( T,'y'),'x')", name = "T_prof")
analysis.add_task("(1/Lx)*integ((1/Ly)*integ( sqrt(u*u),'y'),'x')", name = "u_prof")
analysis.add_task("(1/Lx)*integ((1/Ly)*integ( sqrt(v*v),'y'),'x')", name = "v_prof")
analysis.add_task("(1/Lx)*integ((1/Ly)*integ( sqrt(w*w),'y'),'x')", name = "w_prof")
analysis.add_task("sqrt((1/Lx)*integ((1/Ly)*integ(u*u + v*v + w*w,'y'),'x'))", name = "Pe_prof")
analysis.add_task("(1/Pr)*(1/Lx)*integ((1/Ly)*integ( sqrt(u*u + v*v + w*w),'y'),'x')", name = "Re_prof")
analysis.add_task("(1/Lx)*integ((1/Ly)*integ( sqrt(u*u + v*v),'y'),'x')", name = "U_H_prof")
analysis.add_task("-(Pr/(Lz*Lx*Ly))*integ(u*(hdx(hdx(u)) + hdy(hdy(u)) + dz(uz)) + v*(hdx(hdx(v)) + hdy(hdy(v)) + dz(vz)) + w*(hdx(hdx(w)) + hdy(hdy(w))+dz(wz)))", name = "dissip")
analysis.add_task("(Ra*Pr/(Lz*Lz*Ly))*integ(T*w)",name = "buoyancy")
analysis.add_task("z", name = "z")
analysis.add_task("(1/(Lx*Ly))*integ(integ( Tz,'x'),'y')", name = "conduction_prof")
analysis.add_task("(1/(Lx*Ly))*integ(integ( w*T,'x'),'y')", name = "advection_prof")
analysis.add_task("(1/(Lx*Ly*Lz))*integ(dx(T)*dx(T) + dy(T)*dy(T) + Tz*Tz)", name = "thermal_dissipation")

# =============================================================================
# Set CFL
# =============================================================================

CFL = flow_tools.CFL(solver, initial_dt = init_dt, cadence=5, safety=0.2, max_change=1.2, min_change=0.1, max_dt = max_dt)
CFL.add_velocities(('u', 'v', 'w'))

# =============================================================================
# Set up the flow properties
# =============================================================================

flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + v*v + w*w) / Pr", name='Re')
flow.add_property("sqrt((1/(Lz*Lx*Ly))*integ(u*u + v*v + w*w))", name = "Pe")
flow.add_property("(1/Pr)*sqrt((1/(Lz*Lx*Ly))*integ(u*u + v*v + w*w))", name = "Re")
flow.add_property("interp((1/(Ly))*integ( (1/Lx)*integ( -dz(T) + w*T ,'x'),'y') ,z=0.0)", name = 'nu_mid_plane')
flow.add_property("(1/Lz)*integ((1/(Ly))*integ( (1/Lx)*integ( -dz(T) + w*T ,'x'),'y') ,'z')", name = 'nu_integral')
flow.add_property("interp((1/(Ly))*integ((1/Lx)*integ(-dz(T) ,'x'),'y') , z = -0.5)", name = 'nu_bot_wall')
flow.add_property("interp((1/(Ly))*integ((1/Lx)*integ(-dz(T) ,'x'),'y') , z = 0.5)", name = 'nu_top_wall')
flow.add_property("interp(interp(interp(T,x=1),y=1),z=-0.5)", name = "T_bot")
flow.add_property("interp(interp(interp(T,x=1),y=1),z=0.5)", name = "T_top")
flow.add_property("interp(interp(interp(T,x=1),y=1),z=0)", name = "T_mid")
flow.add_property("-(Pr/(Lz*Ly*Lx))*integ(u*(hdx(hdx(u)) + hdy(hdy(u)) + dz(uz)) + v*(hdx(hdx(v)) + hdy(hdy(v)) + dz(vz)) + w*(hdx(hdx(w)) + hdy(hdy(w)) + dz(wz)))", name = "dissip")
flow.add_property("(Ra*Pr/(Lz*Lz*Ly))*integ(T*w)",name = "buoyancy")
flow.add_property("u", name = 'u')
flow.add_property('v', name = 'v')
flow.add_property('w', name = 'w')
flow.add_property("p", name = 'pressure')
flow.add_property('T', name = 'temperature')
flow.add_property("(1/(Lx*Ly*Lz))*integ(dx(T)*dx(T) + dy(T)*dy(T) + Tz*Tz)", name = "thermal_dissipation") ## Two lots of T so need to devide by T twice ##

end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))

# =============================================================================
# Open the log file and write the titles too 
# =============================================================================

if seed == 'None':
    log_file = open('results/{}/{}log.txt'.format(file_tag,file_tag),'w')
    log_file.write("3D-rrbc, Ra:{:.2e}, Ek:{:.2e}, Nz:{}, Ny:{}, Nx:{}, Pr:{}, \n".format(Rayleigh,Ekman,Nz,Ny,Nx,Prandtl))
    log_file.write('time\tRe\tNu-top\tNu-bottom\tNu-midplane\tNu-integral\tumax\tvmax\twmax\tbuoyancy\tdissip\tenergy-balance\tnu-error\tD_visc\tthermal_dissipation\t\n')
    log_file.close()

# =============================================================================
# Main loop 
# =============================================================================

try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info(" ")
            logger.info("-"*80)
            logger.info("Ra:{:.4e}, Ek:{:.4e}, Pr:{:.4e}, N:{}, q:{}, k_0:{}, enhanced:{}".format(Rayleigh, Ekman, Prandtl, N, q, k_0, enhancedZ == True))
            logger.info("-"*80)
            logger.info('Iteration: %i, Time: %e, dt: %e, Convective time: %f' %(solver.iteration, solver.sim_time, dt,solver.sim_time*np.sqrt(Rayleigh*Prandtl)))
            logger.info("Re:{:.4e}, Pe:{:.4f}, Max T:{:.4f}, Max P:{:.4e}".format(flow.max('Re'), flow.max('Pe'), flow.max('temperature'), flow.max('pressure')))
            nu_bot = flow.max('nu_bot_wall')
            nu_top = flow.max('nu_top_wall')
            nu_mid = flow.max('nu_mid_plane')
            nu_int = flow.max('nu_integral')
            reynolds = flow.max('Re')
            if np.isnan(reynolds):
                logger.error("Reynolds is NaN, simulation likely diverged. Check CFL.")
                raise
            sim_time=solver.sim_time
            t_top = flow.max('T_top')
            t_bot = flow.max('T_bot')
            t_mid = flow.max('T_mid')
            uu = flow.max('u')
            vv = flow.max('v')
            ww = flow.max('w')
            viscousDissipation = flow.max('dissip')
            T_dissip = flow.max('thermal_dissipation')
            buoyancy = flow.max('buoyancy')
            balance = np.abs(viscousDissipation - buoyancy)/np.abs(viscousDissipation)
            nu_balance = np.amax( [np.abs(nu_bot-nu_top)/np.abs(nu_top), np.abs(nu_bot-nu_int)/np.abs(nu_int), np.abs(nu_top-nu_int)/np.abs(nu_int)] )
            logger.info('Nusselt midplane={:.4f}, nu_top={:.4f}, nu_bottom={:.4f}, nu_integral={:.4f}, Nu_error:{:.4e}, Thermal Dissipation = {:.4f}'.format(flow.max('nu_mid_plane'),flow.max('nu_top_wall'),flow.max('nu_bot_wall'),flow.max('nu_integral'),nu_balance, flow.max('thermal_dissipation')))
            logger.info('u:{:.4e}, v:{:.4e}, w:{:.4e}'.format(uu,vv,ww))
            logger.info('buoyancy:{:5.4e}, dissipation:{:5.4e}, balance:{:.3f}%'.format(buoyancy, viscousDissipation, 100*balance))
            logger.info('Thermal Balance = {:.3f}%.'.format(100*((nu_int - flow.max('thermal_dissipation'))/flow.max('thermal_dissipation'))))
            if comm.rank == 0:
                output_file = open('results/{}/{}log.txt'.format(file_tag,file_tag),'a')
                output_file.write("{:5.4e}\t{:5.4e}\t{:5.4e}\t{:5.4e}\t{:5.4e}\t{:5.4e}\t{:5.4e}\t{:5.4e}\t{:5.4e}\t{:5.4e}\t{:5.4e}\t{:5.4e}\t{:5.4e}\t{:5.4e}\t\n".format(sim_time, reynolds, nu_top,nu_bot, nu_mid, nu_int, uu, vv, ww, buoyancy, viscousDissipation, balance, nu_balance, T_dissip))
                output_file.close()
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))
