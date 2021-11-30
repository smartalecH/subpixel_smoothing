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
filename = "beta_mag_sweep"
filter_radius = 0.2

beta_list = np.logspace(0,5,20).tolist()

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
Ny = int(design_region_resolution*design_region_size.y) + 1

## ensure reproducible results
np.random.seed(314159)

## random design region
p = np.random.rand(Nx*Ny)

## random epsilon perturbation for design region
deps = 1e-5
dp = deps*np.random.rand(Nx*Ny)

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

matgrid = mp.MaterialGrid(mp.Vector3(Nx,Ny),
                            mp.air,
                            silicon,
                            weights=p,
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

def mapping(x,r):
    filtered_field = mpa.conic_filter(x,
                                      r,
                                      design_region_size.x,
                                      design_region_size.y,
                                      design_region_resolution)

    return filtered_field.flatten()

def mapping_old(x,r,beta):
    filtered_field = mpa.conic_filter(x,
                                      r,
                                      design_region_size.x,
                                      design_region_size.y,
                                      design_region_resolution)

    y = filtered_field.flatten()
    if beta == 0:
        return y
    else:
        return mpa.tanh_projection(y,beta,0.5).flatten()

l2_new_list = [] #l2 norm of the gradient with hybrid levelset and subpixel smoothing
l2_old_list = [] #l2 norm of the gradient without
for bi, b in enumerate(beta_list):
    ''' test the new hybrid method and compute the l2 norm of the gradient'''
    opt.design_regions[0].design_parameters.do_averaging = True
    opt.update_design([mapping(p,filter_radius)],beta=1e6)
    opt.plot2D(True)
    plt.show()
    quit()
    f, adjsol_grad = opt()
    bp_adjsol_grad = tensor_jacobian_product(mapping,0)(p,filter_radius,adjsol_grad)
    l2_new_list.append(np.linalg.norm(bp_adjsol_grad))
    
    ''' test the old method and compute the l2 norm of the gradient'''
    opt.design_regions[0].design_parameters.do_averaging = False
    opt.update_design([mapping_old(p,filter_radius,b)],beta=0)
    f, adjsol_grad = opt()
    bp_adjsol_grad = tensor_jacobian_product(mapping_old,0)(p,filter_radius,b,adjsol_grad)

    l2_old_list.append(np.linalg.norm(bp_adjsol_grad))
    
print(l2_new_list)
print(l2_old_list)

plt.figure()
plt.loglog(np.array(beta_list),l2_new_list,"o-",label="Hybrid method")
plt.loglog(np.array(beta_list),l2_old_list,"o-",label="Traditional density method")
plt.grid(True)
plt.legend()
plt.xlabel('β')
plt.ylabel("L2 norm of gradient")
if mp.am_master():
    plt.savefig(filename+".png")
    np.savez(filename+".npz",beta_list=beta_list,l2_new_list=l2_new_list,l2_old_list=l2_old_list)
    plt.show()
