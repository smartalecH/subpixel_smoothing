import meep as mp
try:
    import meep.adjoint as mpa
except:
    import adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product
from enum import Enum
from matplotlib import pyplot as plt

resolution = 30
si = mp.Medium(epsilon=12)
if 1:
    silicon = mp.Medium(epsilon=12)
else:
    # saphire rotated
    silicon = mp.Medium(epsilon_diag=(10.225,10.225,9.95),epsilon_offdiag=(-0.825,-0.55*np.sqrt(3/2),0.55*np.sqrt(3/2)))

sxy = 5.0
cell_size = mp.Vector3(sxy,sxy,0)

dpml = 1.0
boundary_layers = [mp.PML(thickness=dpml)]

eig_parity = mp.EVEN_Y + mp.ODD_Z

dx = 2.0
dy = 2.0
design_region_size = mp.Vector3(dx,dy)
design_region_resolution = int(2*resolution)
Nx = int(design_region_resolution*design_region_size.x) + 1
Ny = int(design_region_resolution*design_region_size.y) + 1

## ensure reproducible results
np.random.seed(314159)

## random design region
p = np.random.rand(Nx*Ny)
x = np.arange(Nx) - int(Nx/2)
y = np.arange(Ny) - int(Ny/2)
X, Y = np.meshgrid(x,y)
Z = np.zeros((Nx,Ny))
mask = (np.abs(Y)  <= 10) & (np.abs(X) <= int(1e10))
'''plt.imshow(mask)
plt.show()
quit()'''
#p = np.zeros((Nx,Ny))
#p[mask.T] = 1
#p = p.flatten()
#p = p*0

## random epsilon perturbation for design region
deps = 1e-5
dp = deps*np.random.rand(Nx*Ny)

w = 1.0
waveguide_geometry = [mp.Block(material=si,
                               center=mp.Vector3(),
                               size=mp.Vector3(mp.inf,w,mp.inf))]

fcen = 1/1.55
df = 0.23*fcen
mode = 1
sources = [mp.EigenModeSource(src=mp.GaussianSource(fcen,fwidth=df),
                              center=mp.Vector3(-0.5*sxy+dpml+0.1,0),
                              size=mp.Vector3(0,sxy-2*dpml),
                              eig_band=mode,
                              eig_parity=eig_parity)]

matgrid = mp.MaterialGrid(mp.Vector3(Nx,Ny),
                            mp.air,
                            silicon,
                            #grid_type='U_MEAN',
                            damping=0,
                            weights=np.ones((Nx*Ny,)),
                            do_averaging=True)

matgrid_region = mpa.DesignRegion(matgrid,
                                    volume=mp.Volume(center=mp.Vector3(),
                                                    size=mp.Vector3(design_region_size.x,design_region_size.y,0)))

matgrid_geometry =  [mp.Block(center=matgrid_region.center,
                                size=matgrid_region.size,
                                material=matgrid)]
'''matgrid_geometry += [mp.Block(center=matgrid_region.center,
                                size=matgrid_region.size,
                                material=matgrid,e2=mp.Vector3(y=-1))]'''

geometry = waveguide_geometry + matgrid_geometry

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=boundary_layers,
                    sources=sources,
                    geometry=geometry)

frequencies = [fcen]
#frequencies = 1/np.array([1.5,1.52,1.53,1.54,1.55])
#frequencies = 1/np.linspace(1.5,1.6,10)
obj_list = [mpa.EigenmodeCoefficient(sim,
                                    mp.Volume(center=mp.Vector3(0.5*sxy-dpml-0.1),
                                    size=mp.Vector3(0,sxy-2*dpml,0)),
                                    mode,
                                    eig_parity=eig_parity)]

def J(mode_mon):
    return npa.power(npa.abs(mode_mon),2)

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=J,
    objective_arguments=obj_list,
    design_regions=[matgrid_region],
    frequencies=frequencies)

def mapping(x,filter_radius):
    filtered_field = mpa.conic_filter(x,
                                      filter_radius,
                                      design_region_size.x,
                                      design_region_size.y,
                                      design_region_resolution)

    return filtered_field.flatten()
filter_radius = 0.25
beta = 1e5
opt.update_design([mapping(p,filter_radius)],beta=beta)

'''opt.plot2D(False)
plt.show()
quit()'''

'''sim.init_sim()
e2 = np.zeros([int(sxy*resolution), int(sxy*resolution)], dtype=np.complex128)
e1 = np.zeros([int(sxy*resolution), int(sxy*resolution)], dtype=np.complex128)
x = np.linspace(-sxy/2, sxy/2, int(sxy*resolution))
y = np.linspace(-sxy/2, sxy/2, int(sxy*resolution))
for i in range(len(x)):
    for j in range(len(y)):
        v3 = mp.py_v3_to_vec(sim.dimensions, mp.Vector3(x[i], y[j], 0), sim.is_cylindrical)
        e2[i, j] = sim.fields.get_inveps(mp.Ex, mp.Y, v3)
        e1[i, j] = sim.fields.get_inveps(mp.Ey, mp.X, v3)

plt.figure()
plt.imshow(np.rot90(np.real(e1)))
plt.colorbar()
if mp.am_master():
    plt.show()
quit()'''

f, adjsol_grad = opt()
#bp_adjsol_grad = tensor_jacobian_product(mapping,0)(p,filter_radius,adjsol_grad)
if np.asarray(frequencies).size > 1:
    bp_adjsol_grad = np.zeros(adjsol_grad.shape)
    for i in range(len(frequencies)):
        bp_adjsol_grad[:,i] = tensor_jacobian_product(mapping,0)(p,filter_radius,adjsol_grad[:,i])
else:
    bp_adjsol_grad = tensor_jacobian_product(mapping,0)(p,filter_radius,adjsol_grad)
if bp_adjsol_grad.ndim < 2:
    bp_adjsol_grad = np.expand_dims(bp_adjsol_grad,axis=1)

'''if mp.am_master():
    plt.figure()
    plt.imshow(np.rot90(bp_adjsol_grad[:,-1].reshape(Nx,Ny)))
    plt.colorbar()

    #plt.figure()
    #plt.semilogy(bp_adjsol_grad[:,-1].reshape(Nx,Ny)[int(Nx/2),:])
    plt.show()
quit()'''

#print(bp_adjsol_grad[:,-1].reshape(Nx,Ny))
#quit()

bp_adjsol_grad_m = (dp[None,:]@bp_adjsol_grad).flatten()



opt.update_design([mapping(p+dp/2,filter_radius)])
f_pp, _ = opt(need_gradient=False)
opt.update_design([mapping(p-dp/2,filter_radius)])
f_pm, _ = opt(need_gradient=False)
fd_grad = f_pp-f_pm
rel_error = np.abs(bp_adjsol_grad_m-fd_grad) / np.abs(fd_grad)

print("Directional derivative -- adjoint solver: {}, FD: {}|| rel_error = {}".format(bp_adjsol_grad_m,fd_grad,rel_error))
