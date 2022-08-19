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
dx = 2.0
dy = 2.0
filename = "beta_sweep_new_{}".format(resolution)
filter_radius = 0.05

beta_list = np.logspace(0,3,20).tolist()

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
            mp.Volume(center=mp.Vector3(+0.5*sxy-dpml-0.1),
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

def pad(x,pad):
    z0 = npa.zeros((pad[0]))
    z1 = npa.zeros((pad[1]))
    return npa.concatenate((z0,x,z1))

def convolve(x,h):
    X = npa.fft.fft(x)
    H = npa.fft.fft(h)
    Y = X*H
    y = npa.fft.fftshift(npa.real(npa.fft.ifft(Y)))
    return y

def mapping(x,r):
    kernel = npa.where(npa.abs(grid**2) <= r**2,
        (1 - npa.sqrt(npa.abs(grid**2)) / r), 0)
    kernel = kernel / npa.sum(kernel.flatten())
    kernel = pad(kernel,(Nx,Nx))
    x      = pad(x,(Nx,Nx))
    y      = convolve(x,kernel)
    return y[Nx:2*Nx]

def mapping_old(x,r,beta):
    kernel = npa.where(npa.abs(grid**2) <= r**2,
        (1 - npa.sqrt(npa.abs(grid**2)) / r), 0)
    kernel = kernel / npa.sum(kernel.flatten())
    kernel = pad(kernel,(Nx,Nx))
    x      = pad(x,(Nx,Nx))
    y      = convolve(x,kernel)[Nx:2*Nx]
    return mpa.tanh_projection(y,beta,0.5)

if __name__ == '__main__':
    smoothing = [False,True]
    no_smoothing_list = []
    smoothing_list = []
    for si, s in enumerate(smoothing):
        for bi, b in enumerate(beta_list):
            opt.design_regions[0].design_parameters.do_averaging = s
            beta = b

            opt.update_design([mapping(p,filter_radius)],beta=beta)

            f, adjsol_grad = opt()
            bp_adjsol_grad = tensor_jacobian_product(mapping,0)(p,filter_radius,adjsol_grad)
            if bp_adjsol_grad.ndim < 2:
                bp_adjsol_grad = np.expand_dims(bp_adjsol_grad,axis=1)
            bp_adjsol_grad_m = (dp[None,:]@bp_adjsol_grad).flatten()

            opt.update_design([mapping(p+dp/2,filter_radius)],beta=beta)
            f_pp, _ = opt(need_gradient=False)
            opt.update_design([mapping(p-dp/2,filter_radius)],beta=beta)
            f_pm, _ = opt(need_gradient=False)
            fd_grad = f_pp-f_pm
            rel_error = np.abs(bp_adjsol_grad_m-fd_grad) / np.abs(fd_grad)
            print(rel_error)
            if s:
                smoothing_list.append(rel_error)
            else:
                no_smoothing_list.append(rel_error)

    plt.figure()
    plt.loglog(beta_list,smoothing_list,"o-",label="With smoothing")
    plt.loglog(beta_list,no_smoothing_list,"o-",label="Without smoothing")
    plt.legend()
    plt.grid(True)
    plt.xlabel('β')
    plt.ylabel("Relative error")
    if mp.am_master():
        plt.savefig(filename+".png")
        np.savez("beta_sweep.npz",beta_list=beta_list,smoothing_list=smoothing_list,no_smoothing_list=no_smoothing_list)
        plt.show()
