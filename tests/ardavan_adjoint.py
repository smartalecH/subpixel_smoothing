import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product
import matplotlib.pyplot as plt

silicon = mp.Medium(epsilon=12.)

sxy = 5.
cell_size = mp.Vector3(sxy,sxy,0)

dpml = 1.
pml = [mp.PML(thickness=dpml)]

eig_parity = mp.EVEN_Y + mp.ODD_Z

design_region_size = mp.Vector3(1.5,1.5)
design_region_resolution = 256.
Nx = int(design_region_resolution*design_region_size.x)
Ny = int(design_region_resolution*design_region_size.y)

fcen = 1/1.55
df = 0.23*fcen
source = [mp.Source(src=mp.GaussianSource(fcen,fwidth=df,is_integrated=True),
                    center=mp.Vector3(-0.5*sxy+dpml,0),
                    size=mp.Vector3(0,sxy),
                    component=mp.Ez)]

x = np.linspace(-0.5*design_region_size.x,0.5*design_region_size.x,Nx)
y = np.linspace(-0.5*design_region_size.y,0.5*design_region_size.y,Ny)
xv, yv = np.meshgrid(x,y)

rad = 0.538948295
wdt = 0.194838432
weights = np.where(np.logical_and(np.sqrt(np.square(xv) + np.square(yv)) > rad,
                                  np.sqrt(np.square(xv) + np.square(yv)) < rad+wdt),
                   1.,
                   0.)

filter_radius = 20/design_region_resolution

def mapping(x):
    filtered_weights = mpa.conic_filter(x,
                                        filter_radius,
                                        design_region_size.x,
                                        design_region_size.y,
                                        design_region_resolution)

    return filtered_weights.flatten()

matgrid = mp.MaterialGrid(mp.Vector3(Nx,Ny),
                          mp.air,
                          silicon,
                          weights=np.ones((Nx,Ny)),
                          do_averaging=True,
                          beta=mp.inf)

matgrid_region = mpa.DesignRegion(matgrid,
                                  volume=mp.Volume(center=mp.Vector3(),
                                                   size=mp.Vector3(design_region_size.x,
                                                                   design_region_size.y,
                                                                   0)))

geometry = [mp.Block(center=matgrid_region.center,
                     size=matgrid_region.size,
                     material=matgrid)]

def J(mode_mon):
    return npa.power(npa.abs(mode_mon[:,4,10]),2)

# ensure reproducible results                                                                                                                                
rng = np.random.RandomState(9861548)

# random epsilon perturbation for design region                                                                                                              
deps = 1e-5
dp = deps*rng.rand(Nx*Ny)

p = 0.1 + 0.5*weights.flatten()
mapped_p = mapping(p)
mapped_p_plus_dp = mapping(p+dp)

for res in [25]:#[25., 50., 100., 200.]:
    sim = mp.Simulation(
        resolution=res,
        cell_size=cell_size,
        boundary_layers=pml,
        k_point=mp.Vector3(),
        sources=source,
        geometry=geometry
    )

    obj_list = [mpa.FourierFields(sim,
                                  mp.Volume(center=mp.Vector3(1.25),
                                            size=mp.Vector3(0.25,1.)),
                                  mp.Ez)]

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=J,
        objective_arguments=obj_list,
        design_regions=[matgrid_region],
        frequencies=[fcen],
        finite_difference_step=1e-8
    )
    opt.plot2D()
    plt.show()
    quit()

    f_unperturbed, adjsol_grad = opt([mapped_p])
    bp_adjsol_grad = tensor_jacobian_product(mapping,0)(p,adjsol_grad)
    
    f_perturbed, _ = opt([mapped_p_plus_dp], need_gradient=False)

    if bp_adjsol_grad.ndim < 2:
        bp_adjsol_grad = np.expand_dims(bp_adjsol_grad,axis=1)
    adj_scale = (dp[None,:]@bp_adjsol_grad).flatten()
    fd_grad = f_perturbed[0]-f_unperturbed[0]
    print(f"obj:, {f_perturbed[0]}, {f_unperturbed[0]}, {f_perturbed}")
    rel_err = abs((fd_grad - adj_scale[0])/fd_grad)
    print(f"dir_deriv:, {res}, {fd_grad}, {adj_scale[0]}, {rel_err}")

    sim.reset_meep()