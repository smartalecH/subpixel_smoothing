import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
from matplotlib import pyplot as plt
import nlopt
from utils import fit_initial_params
mp.quiet()

# ---------------------------------------- #
# basic parameters
# ---------------------------------------- #

resolution = 30
design_region_resolution = int(2*resolution)
dx = 2.0
dy = 2.0
filename = "bend"
min_length = 0.09

eta = 0.5
eta_e = 0.75

dpml = 1.0

maxeval = 10

w = 0.5
sep = 0.5

data = []
results = []
beta_history = []
# ---------------------------------------- #
# derived quantities
# ---------------------------------------- #
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.44)


eta_d = 1-eta_e
filter_radius = mpa.get_conic_radius_from_eta_e(min_length,eta_e)
c = (design_region_resolution*filter_radius)**4

sx = dx + 2 + 2*dpml
sy = dy + 2 + 2*dpml

# ---------------------------------------- #
# main routine
# ---------------------------------------- #

cell_size = mp.Vector3(sx,sy,0)
boundary_layers = [mp.PML(thickness=dpml)]

eig_parity = mp.NO_PARITY

design_region_size = mp.Vector3(dx,dy)
Nx = int(design_region_resolution*design_region_size.x) + 1
Ny = int(design_region_resolution*design_region_size.y) + 1

waveguide_geometry = [mp.Block(material=Si, center=mp.Vector3(-sx/2),
                               size=mp.Vector3(sx,w,mp.inf)),
                      mp.Block(material=Si, center=mp.Vector3(0,sx/2),
                               size=mp.Vector3(w,sx,mp.inf))        
                            ]

fcen = 1/1.55
df = 0.23*fcen
sources = [mp.EigenModeSource(src=mp.GaussianSource(fcen,fwidth=df),
                              center=mp.Vector3(-dy/2-.75,0),
                              size=mp.Vector3(0,sy-2*dpml),
                              eig_band=1)]

matgrid = mp.MaterialGrid(mp.Vector3(Nx,Ny),
                            SiO2,
                            Si,
                            weights=np.ones((Nx,Ny)),
                            grid_type="U_MEAN",
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
                    default_material=SiO2,
                    geometry=geometry)

frequencies = np.array([fcen])

mon_size = 2.0
obj_list = [mpa.EigenmodeCoefficient(sim,
                                    mp.Volume(center=mp.Vector3(-dx/2-0.5),
                                    size=mp.Vector3(0,mon_size,0)),
                                    1),
            mpa.EigenmodeCoefficient(sim,
                                    mp.Volume(center=mp.Vector3(0,dy/2+0.5),
                                    size=mp.Vector3(mon_size,0,0)),
                                    1)]

def J(input,top):
    return 1-npa.abs(top/input)**2

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=J,
    objective_arguments=obj_list,
    design_regions=[matgrid_region],
    frequencies=frequencies)

# ---------------------------------------------- #
#
# ---------------------------------------------- #

x = np.linspace(-dx/2,dy/2,Nx)
y = np.linspace(-dx/2,dy/2,Ny)
Y, X = np.meshgrid(x,y)
Z = np.zeros((Nx,Ny))
Si_mask = (((np.abs(Y) <= w/2) & (X  == -dx/2))
+ ((np.abs(X) <= w/2) & (Y  == dy/2))
)

SiO2_mask = ((X  == -dx/2)+(X  == dx/2)+(Y  == -dy/2)+(Y  == dy/2))

SiO2_mask = SiO2_mask ^ Si_mask

def mapping(x):
    x = x.reshape(Nx,Ny)
    x = npa.where(Si_mask,1,npa.where(SiO2_mask,0,x))
    x = mpa.conic_filter(x,
                        filter_radius,
                        design_region_size.x,
                        design_region_size.y,
                        design_region_resolution)
    x = npa.clip(x,0,1)
    x = (npa.flipud(npa.fliplr(x.T)) + x) / 2
    return x.flatten()

def f(x, grad, beta):
    data.append(x.copy())
    y = mapping(x)
    print("x: min: {}  max: {}".format(np.min(x),np.max(x)))
    print("y: min: {}  max: {}".format(np.min(y),np.max(y)))

    opt.update_design([y],beta=beta)
    
    f0, dJ_du = opt()
    bp_adjsol_grad = tensor_jacobian_product(mapping,0)(x,dJ_du)
    if grad.size > 0:
        grad[:] = np.squeeze(bp_adjsol_grad)
    
    if mp.am_master():
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(grad.reshape(Nx,Ny))
        plt.subplot(1,2,2)
        plt.imshow(x.reshape(Nx,Ny))
        plt.show()
    opt.plot2D(False,eps_parameters={'resolution':100})
    plt.show()

    print("f: {}".format(float(np.real(f0))))
    results.append(float(np.real(f0)))
    beta_history.append(beta)
    return float(np.real(f0))

def constraints(result,x,gradient):
    tol = 1e-5
    beta_mod = 8
    def mapping_mod(a):
        return mapping(a).reshape(Nx,Ny)
    c_solid = lambda a: mpa.constraint_solid(a, c, eta_e,
        mapping_mod, 
        lambda b: mpa.tanh_projection(b,beta_mod,eta), 
        design_region_resolution) - tol
    c_void = lambda a: mpa.constraint_void(a, c, eta_d, 
        mapping_mod,
        lambda b: mpa.tanh_projection(b,beta_mod,eta),
        design_region_resolution) - tol
    solid = float(np.real(c_solid(x)))
    void = float(np.real(c_void(x)))
    result[0] = solid
    result[1] = void
    print("solid: {}, void: {}   |   ".format(solid,void), result)
    if gradient.size > 0:
        gradient[0,:] = grad(c_solid)(x)
        gradient[1,:] = grad(c_void)(x)

x = mapping(np.ones((Nx,Ny))*0.5)
opt.update_design([x])
'''opt.prepare_forward_run()
opt.plot2D(True)
plt.show()
quit()
opt.forward_run()
opt.prepare_adjoint_run()
opt.sim.reset_meep()
opt.sim.change_sources(opt.adjoint_sources[0])
print(opt.adjoint_sources[0])
opt.sim.run(until=20)
opt.plot2D(fields=mp.Ez)
plt.show()
quit()'''
if __name__ == '__main__':
    algorithm = nlopt.LD_MMA
    n = Nx * Ny
    beta = [8, 32, np.inf]
    for iters in range(len(beta)):
        solver = nlopt.opt(algorithm, n)
        solver.set_lower_bounds(0)
        solver.set_upper_bounds(1)
        solver.set_min_objective(lambda a,g: f(a,g,beta[iters]))
        solver.set_maxeval(maxeval)
        if iters == len(beta)-1:
            solver.add_inequality_mconstraint(constraints,[0]*2)

        x[:] = solver.optimize(x)
    data.append(x.copy())

    opt.update_design([mapping(x)],beta=beta[-1])
    f0, _ = opt(need_gradient=False)
    results.append(float(np.real(f0)))

    if mp.am_really_master():
        np.savez(filename+"_data.npz",data=data,results=results,beta_history=beta_history)

    opt.plot2D(False,eps_parameters={'resolution':100})
    plt.show()