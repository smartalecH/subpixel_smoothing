import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product
from enum import Enum
from matplotlib import pyplot as plt

# ---------------------------------------- #
# basic parameters
# ---------------------------------------- #

resolution = 30
design_region_resolution = int(2*resolution)
beta = 1e6
dx = 2.0
dy = 2.0
filename = "u_sweep"

u_bank = np.linspace(0,1,15)
idx = 37
idx2 = 78
smoothing_radius = 0.05
# ---------------------------------------- #
# main routine
# ---------------------------------------- #

silicon = mp.Medium(epsilon=12)
sxy = 5.0
cell_size = mp.Vector3(sxy,sxy,0)

dpml = 1.0
boundary_layers = [mp.PML(thickness=dpml)]

eig_parity = mp.EVEN_Y + mp.ODD_Z

design_region_size = mp.Vector3(dx,dy)
Nx = int(design_region_resolution*design_region_size.x) + 1

## ensure reproducible results
np.random.seed(314159)

## random design region
#p = np.random.rand(Nx)
p = np.zeros((Nx,))
p[idx-1:idx2] = 1

## random epsilon perturbation for design region
deps = 1e-5

w = 1.0
waveguide_geometry = [mp.Block(material=silicon,
                               center=mp.Vector3(),
                               size=mp.Vector3(mp.inf,w,mp.inf))]

fcen = 1/1.55
df = 0.23*fcen
sources = [mp.EigenModeSource(src=mp.GaussianSource(fcen,fwidth=df),
                              center=mp.Vector3(-0.5*sxy+dpml+0.1,0),
                              size=mp.Vector3(0,sxy-2*dpml),
                              eig_band=1,
                              eig_parity=eig_parity)]

matgrid = mp.MaterialGrid(mp.Vector3(Nx),
                            mp.air,
                            silicon,
                            weights=np.ones((Nx,)),
                            do_averaging=True)

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

obj_list = [mpa.EigenmodeCoefficient(sim,
                                    mp.Volume(center=mp.Vector3(0.5*sxy-dpml-0.1),
                                    size=mp.Vector3(0,sxy-2*dpml,0)),
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

grid = np.linspace(-dx/2,dx/2,Nx)
def mapping(x,r):
    kernel = np.where(np.abs(grid**2) <= r**2,
        (1 - np.sqrt(abs(grid**2)) / r), 0)
    kernel = kernel / np.sum(kernel.flatten())
    kernel = np.pad(kernel,(Nx,Nx),mode='edge')
    x = np.pad(x,(Nx,Nx),mode='edge')
    return np.convolve(x,kernel,mode="same")[Nx:2*Nx]

f_list = []
new_method = []
old_method = []
'''#opt.update_design([mapping(p,smoothing_radius)],beta=1e6)
opt.update_design([mpa.tanh_projection(mapping(p,smoothing_radius),1e6,0.5)],beta=0)
opt.plot2D()
plt.show()
quit()'''
for ui, u in enumerate(u_bank):
    '''perturb parameter'''
    p[idx] = u

    '''first simulate the new method with smoothing'''
    opt.design_regions[0].design_parameters.do_averaging = True
    f, _ = opt([mapping(p,smoothing_radius)],beta=1e6,need_gradient=False)
    new_method.append(f)

    '''next simulate the old method without smoothing'''
    opt.design_regions[0].design_parameters.do_averaging = False
    f, _ = opt([mpa.tanh_projection(mapping(p,smoothing_radius),1e6,0.5)],beta=0,need_gradient=False)
    old_method.append(f)

plt.figure()
plt.plot(u_bank,new_method,"o-",label="hybrid level-set")
plt.plot(u_bank,old_method,"o-",label="traditional density method")
plt.legend()
plt.grid(True)
plt.xlabel('$u_k$')
plt.ylabel("$f(u)$")
if mp.am_master():
    plt.savefig(filename+".png")
    np.savez(filename+".npz",u_bank=u_bank,new_method=new_method,old_method=old_method)
    plt.show()
