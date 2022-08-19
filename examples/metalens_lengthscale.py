import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import nlopt
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from scipy import special, signal

mp.verbosity(0)
Si = mp.Medium(index=2.4)
SiO2 = mp.Medium(index=1.0)
design_region_width = 20
design_region_height = 2
pml_size = 1.0
resolution = 30
Sx = design_region_width
Sy = 2*pml_size + design_region_height + 5
cell_size = mp.Vector3(Sx,Sy)

nf = 3
frequencies = np.array([1/0.45 ,1/0.55, 1/0.65])

minimum_length = 0.1 # minimum length scale (microns)
eta_i = 0.5 # blueprint (or intermediate) design field thresholding point (between 0 and 1)
eta_e = 0.55 # erosion design field thresholding point (between 0 and 1)
eta_d = 1-eta_e # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length,eta_e)
design_region_resolution = int(2*resolution)

pml_layers = [mp.PML(pml_size, direction=mp.Y)]

fcen = 1/0.55
width = 0.2
fwidth = width * fcen
source_center  = [0,-(design_region_height/2 + 1.5),0]
source_size    = mp.Vector3(Sx,0,0)
src1 = mp.GaussianSource(frequency=1/0.45,fwidth=fwidth)
src2 = mp.GaussianSource(frequency=fcen,fwidth=fwidth)
src3 = mp.GaussianSource(frequency=1/0.65,fwidth=fwidth)
source = [mp.Source(src1, component = mp.Ez, size = source_size, center=source_center),
            mp.Source(src2, component = mp.Ez, size = source_size, center=source_center),
            mp.Source(src3, component = mp.Ez, size = source_size, center=source_center)]

Nx = int(design_region_resolution*design_region_width)+1
Ny = int(design_region_resolution*design_region_height)+1
design_variables = mp.MaterialGrid(mp.Vector3(Nx,Ny),SiO2,Si,grid_type='U_MEAN')
design_region = mpa.DesignRegion(design_variables,volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(design_region_width, design_region_height, 0)))

def mapping(x,eta,beta):
    # filter
    filtered_field = mpa.conic_filter(x,filter_radius,design_region_width,design_region_height,design_region_resolution)
    # projection
    projected_field = mpa.tanh_projection(filtered_field,beta,eta)
    # interpolate to actual materials
    return projected_field.flatten()

geometry = [
    mp.Block(center=design_region.center, size=design_region.size, material=design_variables),
    mp.Block(center=design_region.center, size=design_region.size, material=design_variables, e1=mp.Vector3(x=-1))] # design region
kpoint = mp.Vector3()
sim = mp.Simulation(cell_size=cell_size,
                    boundary_layers=pml_layers,
                    k_point=kpoint,
                    geometry=geometry,
                    sources=source,
                    default_material=SiO2,
                    symmetries=[mp.Mirror(direction=mp.X)],
                    resolution=resolution)

far_x = [mp.Vector3(0,30,0)]
NearRegions = [mp.Near2FarRegion(center=mp.Vector3(0,design_region_height/2+1.5), size=mp.Vector3(design_region_width,0), weight=+1)]
FarFields = mpa.Near2FarFields(sim, NearRegions ,far_x)
ob_list = [FarFields]
def J1(alpha):
    return -npa.abs(alpha[0,:,2])**2

opt = mpa.OptimizationProblem(
    simulation = sim,
    objective_functions = [J1],
    objective_arguments = ob_list,
    design_regions = [design_region],
    frequencies=frequencies,
    maximum_run_time = 1000
)
geom_history = []
evaluation_history = []
cur_iter = [0]
def f(x, grad):
    t = x[0] # "dummy" parameter
    v = x[1:] # design parameters
    if grad.size > 0:
        grad[0] = 1
        grad[1:] = 0
    print("cur t",t)
    return t

def c(result,x,gradient,eta,beta):
    print("Current iteration: {}; current eta: {}, current beta: {}".format(cur_iter[0],eta,beta))

    t = x[0] # dummy parameter
    v = x[1:] # design parameters

    f0, dJ_du = opt([mapping(v,eta,beta)])
    # Backprop the gradients through our mapping function
    my_grad = np.zeros(dJ_du.shape)
    for k in range(opt.nf):
        my_grad[:,k] = tensor_jacobian_product(mapping,0)(v,eta,beta,dJ_du[:,k])

    # Assign gradients
    if gradient.size > 0:
        gradient[:,0] = -1 # gradient w.r.t. "t"
        gradient[:,1:] = my_grad.T # gradient w.r.t. each frequency objective

    result[:] = np.real(f0) - t
    print("cur f0,",f0)

    # store results
    evaluation_history.append(np.real(f0))
    cur_iter[0] = cur_iter[0] + 1

def geom(x,gradient,eta,beta):
    threshf = lambda v: mpa.tanh_projection(v,beta,eta)
    filterf = lambda v: mpa.conic_filter(v,filter_radius,design_region_width,design_region_height,design_region_resolution)
    v = x[1:]
    g_s = mpa.constraint_solid(v,1e7,eta_e,filterf,threshf,resolution) # constraint
    g_s_grad = grad(mpa.constraint_solid,0)(v,1e7,eta_e,filterf,threshf,resolution) # gradient
    gradient[0]=0
    gradient[1:] = g_s_grad
    geom_history.append(g_s)
    return g_s-(1-beta/256)*(1e-5)

algorithm = nlopt.LD_MMA
n = Nx * Ny # number of parameters
# Initial guess
x = np.ones((n,)) * 0.5
# lower and upper bounds
lb = np.zeros((Nx*Ny,))
ub = np.ones((Nx*Ny,))

# insert dummy parameter bounds and variable
x = np.insert(x,0,0) # our initial guess for the worst error
lb = np.insert(lb,0,-np.inf)
ub = np.insert(ub,0,0)

cur_beta = 8
beta_scale = 2
num_betas = 6
update_factor = 5
for iters in range(num_betas):
    solver = nlopt.opt(algorithm, n+1)
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_min_objective(f)
    solver.set_maxeval(update_factor)
    solver.add_inequality_mconstraint(lambda r,x,g: c(r,x,g,eta_i,cur_beta), np.array([1e-3]*nf))
    if cur_beta > 30:
        solver.add_inequality_constraint(lambda x,g: geom(x,g,eta_i,cur_beta), 1e-3)
    x[:] = solver.optimize(x)
    cur_beta = cur_beta*beta_scale

savev = x[1:]
np.save('/lens.npy', savev)
