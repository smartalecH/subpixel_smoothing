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
filename = "radius_sweep"

#filter_radius = np.logspace(2,0.5*np.log10(design_region_resolution*dx),11) / design_region_resolution
filter_radius = np.linspace(2,0.5*design_region_resolution*dx,11) / design_region_resolution
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
                            weights=np.ones((Nx,Ny)),
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

radius_list = []
for ri, r in enumerate(filter_radius):
    opt.update_design([mapping(p,r)])

    f, adjsol_grad = opt()
    bp_adjsol_grad = tensor_jacobian_product(mapping,0)(p,r,adjsol_grad)
    if bp_adjsol_grad.ndim < 2:
        bp_adjsol_grad = np.expand_dims(bp_adjsol_grad,axis=1)
    bp_adjsol_grad_m = (dp[None,:]@bp_adjsol_grad).flatten()

    opt.update_design([mapping(p+dp/2,r)])
    f_pp, _ = opt(need_gradient=False)
    opt.update_design([mapping(p-dp/2,r)])
    f_pm, _ = opt(need_gradient=False)
    fd_grad = f_pp-f_pm
    rel_error = np.abs(bp_adjsol_grad_m-fd_grad) / np.abs(fd_grad)
    print(rel_error)
    radius_list.append(rel_error)

plt.figure()
plt.plot(np.array(filter_radius),radius_list,"o-")
plt.grid(True)
plt.xlabel('Smoothing radius (meep units)')
plt.ylabel("Relative error")
plt.title("Meep resolution: {}, MG resolution: {}".format(resolution,design_region_resolution))
if mp.am_master():
    plt.savefig(filename+".png")
    np.savez(filename+".npz",filter_radius=filter_radius,radius_list=radius_list,design_region_resolution=design_region_resolution)
    plt.show()
