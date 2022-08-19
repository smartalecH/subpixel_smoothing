import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product
from matplotlib import pyplot as plt
import nlopt

# ---------------------------------------- #
# basic parameters
# ---------------------------------------- #

resolution = 30
design_region_resolution = int(2*resolution)
dx = 10.0
dy = 0.22
filename = "grating"
filter_radius = 0.09

beta = np.inf

# ---------------------------------------- #
# initial parameterization
# ---------------------------------------- #

design_region_size = mp.Vector3(dx,dy)
Nx = int(design_region_resolution*design_region_size.x) + 1

init_period = 5*filter_radius
init = np.sin(np.arange(Nx)/design_region_resolution*2*np.pi/init_period)
init = (init + 1) / 2
# ---------------------------------------- #
# main routine
# ---------------------------------------- #
dpml = 1.0
waveguide_length = 2.0
silicon = mp.Medium(epsilon=12)
sx = 2*dpml + waveguide_length + dx
sy = 2*dpml + 2.5
cell_size = mp.Vector3(sx,sy,0)

boundary_layers = [mp.PML(thickness=dpml)]

## ensure reproducible results
np.random.seed(314159)

## random design region
p = np.random.rand(Nx)

## random epsilon perturbation for design region
deps = 1e-5
dp = deps*np.random.rand(Nx)

w = 1.0
waveguide_geometry = [mp.Block(material=silicon,
                               center=mp.Vector3(),
                               size=mp.Vector3(mp.inf,dy,mp.inf))]

fcen = 1/1.55
df = 0.23*fcen
sources = [mp.GaussianBeamSource(src=mp.GaussianSource(fcen,fwidth=df),
                              center=mp.Vector3(0,1),
                              size=mp.Vector3(sx,0),
                              beam_x0=mp.Vector3(0,10),
                              beam_E0=mp.Vector3(0,0,1),
                              beam_w0=3,
                              beam_kdir=mp.Vector3(y=-1))]

matgrid = mp.MaterialGrid(mp.Vector3(Nx),
                            mp.air,
                            silicon,
                            weights=init,
                            do_averaging=True,
                            beta=beta)

matgrid_region = mpa.DesignRegion(matgrid,
                                    volume=mp.Volume(center=mp.Vector3(),
                                                    size=mp.Vector3(design_region_size.x,design_region_size.y,0)))

matgrid_geometry = [mp.Block(center=matgrid_region.center,
                                size=matgrid_region.size,
                                material=matgrid)]

geometry = waveguide_geometry + matgrid_geometry

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=boundary_layers,
                    sources=sources,
                    geometry=geometry)

frequencies = [fcen]

sim.run(until=10)
sim.plot2D()
plt.show()
quit()
obj_list = [mpa.EigenmodeCoefficient(sim,
                                    mp.Volume(center=mp.Vector3(0.5*sx-dpml-0.1),
                                    size=mp.Vector3(0,sy-2*dpml,0)),
                                    1,
                                    eig_parity=eig_parity)]

def J(mode_mon):
    return npa.power(npa.abs(mode_mon),2)

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=J,
    objective_arguments=obj_list,
    design_regions=[matgrid_region],
    frequencies=frequencies)

def mapping(x,r):
    filtered_field = mpa.conic_filter(x,
                                      r,
                                      design_region_size.x,
                                      design_region_size.y,
                                      design_region_resolution)

    return filtered_field.flatten()

def f(x, grad):
    opt.design_regions[0].design_parameters.do_averaging = True
    opt.update_design([mapping(x,filter_radius)],beta=beta)
    
    f0, dJ_du = opt()
    bp_adjsol_grad = tensor_jacobian_product(mapping,0)(x,filter_radius,dJ_du)
    if grad.size > 0:
        grad[:] = np.squeeze(bp_adjsol_grad)
    #evaluation_history.append(np.real(f0))
    #sensitivity[0] = dJ_du
    '''plt.figure()
    plt.imshow(np.rot90(bp_adjsol_grad.reshape(Nx,Ny)))
    if mp.am_master():
        plt.show()
    quit()'''
    return float(np.real(f0))

# initial guess
x = np.arange(Nx) - int(Nx/2)
y = np.arange(Ny) - int(Ny/2)
X, Y = np.meshgrid(x,y)
Z = np.zeros((Nx,Ny))
mask = (np.abs(Y)  <= 0.5*design_region_resolution) & (np.abs(X) <= int(1e10))
p = np.zeros((Nx,Ny))
p[mask.T] = 1
p = p.flatten()

algorithm = nlopt.LD_MMA
n = Nx * Ny
maxeval = 10
solver = nlopt.opt(algorithm, n)
solver.set_lower_bounds(0)
solver.set_upper_bounds(1)
solver.set_max_objective(f)
solver.set_maxeval(maxeval)
#x0 = np.ones((Nx*Ny,))
x = solver.optimize(p)

opt.update_design([mapping(x,filter_radius)],beta=beta)

opt.plot2D()
plt.show()