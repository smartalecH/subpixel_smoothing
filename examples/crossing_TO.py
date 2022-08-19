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
dx = 3.0
dy = 3.0
filename = "cross_TO_new"
min_length = 0.09

eta = 0.5
eta_e = 0.75

dpml = 1.0

data = []
results = []
beta_history = []
# ---------------------------------------- #
# derived quantities
# ---------------------------------------- #

eta_d = 1-eta_e
filter_radius = mpa.get_conic_radius_from_eta_e(min_length,eta_e)
c = (design_region_resolution*filter_radius)**4

sxy = dx + 1 + 2*dpml

# ---------------------------------------- #
# main routine
# ---------------------------------------- #


silicon = mp.Medium(epsilon=12)
cell_size = mp.Vector3(sxy,sxy,0)
boundary_layers = [mp.PML(thickness=dpml)]

eig_parity = mp.EVEN_Y + mp.ODD_Z

design_region_size = mp.Vector3(dx,dy)
Nx = int(design_region_resolution*design_region_size.x)
Ny = int(design_region_resolution*design_region_size.y)

w = 0.5
waveguide_geometry = [mp.Block(material=silicon,
                               size=mp.Vector3(mp.inf,w,mp.inf)),
                      mp.Block(material=silicon,
                               size=mp.Vector3(w,mp.inf,mp.inf))         
                            ]

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
                    geometry=geometry)

NF = 10
frequencies = 1/np.linspace(1.5,1.6,NF)

obj_list = [mpa.EigenmodeCoefficient(sim,
                                    mp.Volume(center=mp.Vector3(-0.5*sxy+dpml+0.2),
                                    size=mp.Vector3(0,sxy-2*dpml,0)),
                                    1,
                                    eig_parity=eig_parity),
            mpa.EigenmodeCoefficient(sim,
                                    mp.Volume(center=mp.Vector3(0.5*sxy-dpml-0.2),
                                    size=mp.Vector3(0,sxy-2*dpml,0)),
                                    1,
                                    eig_parity=eig_parity)]

def J(input,output):
    return 1-npa.power(npa.abs(output/input),2)

opt = mpa.OptimizationProblem(
    simulation=sim,
    maximum_run_time=500,
    objective_functions=J,
    objective_arguments=obj_list,
    design_regions=[matgrid_region],
    frequencies=frequencies)


# ---------------------------------------------- #
#
# ---------------------------------------------- #

x = np.linspace(-dx/2,dy/2,Nx)
y = np.linspace(-dx/2,dy/2,Ny)
X, Y = np.meshgrid(x,y)
Z = np.zeros((Nx,Ny))
Si_mask = (((np.abs(Y)  <= w/2) & (X  == -dx/2))
+ ((np.abs(Y)  <= w/2) & (X  == dx/2))
+ ((np.abs(X)  <= w/2) & (Y  == -dx/2))
+ ((np.abs(X)  <= w/2) & (Y  == dx/2))
)

SiO2_mask = (((np.abs(Y)  >= w/2) & (X  == -dx/2))
+ ((np.abs(Y)  >= w/2) & (X  == dx/2))
+ ((np.abs(X)  >= w/2) & (Y  == -dx/2))
+ ((np.abs(X)  >= w/2) & (Y  == dx/2))
)

def mapping(x):
    x = x.reshape(Nx,Ny)
    x = npa.where(Si_mask,1,npa.where(SiO2_mask,0,x))
    x = (x + npa.rot90(x) + npa.rot90(npa.rot90(x)) + npa.rot90(npa.rot90(npa.rot90(x)))) / 4
    x = (npa.fliplr(x) + x)*0.5
    x = (npa.flipud(x) + x)*0.5
    x = mpa.conic_filter(x,
                        filter_radius,
                        design_region_size.x,
                        design_region_size.y,
                        design_region_resolution)
    x = (x + npa.rot90(x) + npa.rot90(npa.rot90(x)) + npa.rot90(npa.rot90(npa.rot90(x)))) / 4
    x = npa.clip(x,0,1)
    return x.flatten()

def f(x, grad):
    t = x[0]  # "dummy" parameter
    v = x[1:]  # design parameters
    if grad.size > 0:
        grad[0] = 1
        grad[1:] = 0
    return t

def cm(nlopt_result, x, gradient, beta):
    t = x[0]
    v = x[1:]
    data.append(x.copy())

    opt.design_regions[0].design_parameters.do_averaging = True
    opt.update_design([mapping(v)],beta=beta)
    opt.plot2D()
    if mp.am_master():
        plt.savefig("current_crossing_TO_geom.png")
    plt.close("all")
    #plt.show()
    f0, dJ_du = opt()

    bp_adjsol_grad = np.zeros(dJ_du.shape)
    for i in range(len(frequencies)):
        bp_adjsol_grad[:,i] = tensor_jacobian_product(mapping,0)(v,dJ_du[:,i])
        # Assign gradients
    
    if gradient.size > 0:
        gradient[:, 0] = -1  # gradient w.r.t. "t"
        gradient[:, 1:] = bp_adjsol_grad.T  # gradient w.r.t. each frequency objective
    
    nlopt_result[:] = np.real(f0) - t

    print("t: {} | f: {}".format(t,np.real(f0)))
    results.append(np.real(f0))
    '''
    plt.figure()
    for k in range(10):
        plt.subplot(2,5,k+1)
        #plt.imshow(np.rot90(gradient[k,1:].reshape(Nx,Ny)))
        plt.imshow(np.rot90(dJ_du[:,k].reshape(Nx,Ny)))
    if mp.am_master():
        plt.show()'''

def constraints(result,x,gradient):
    x = x[1:]
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
        gradient[:,0] = 0
        gradient[0,1:] = grad(c_solid)(x)
        gradient[1,1:] = grad(c_void)(x)

# initial guess
x = np.linspace(-dx/2,dy/2,Nx)
y = np.linspace(-dx/2,dy/2,Ny)
X, Y = np.meshgrid(x,y)
Z = np.zeros((Nx,Ny))
mask = (np.abs(Y)  <= w/2) + (np.abs(X)  <= w/2)
p = np.zeros((Nx,Ny))
p[mask.T] = 1

x = mapping(np.ones((Nx,Ny))*0.5)


if __name__ == '__main__':

    maxeval = 30
    algorithm = nlopt.LD_MMA
    n = Nx * Ny
    beta = [8, 32, np.inf]
    x = np.array([1] + x.tolist())
    for iters in range(len(beta)):
        #f0, dummy = opt([mapping(x[1:])],need_gradient=False,beta=beta[iters])
        x[0] = 0#np.max(f0)
        solver = nlopt.opt(algorithm, n+1)
        solver.set_lower_bounds(0)
        solver.set_upper_bounds(1)
        solver.set_min_objective(f)
        solver.add_inequality_mconstraint(lambda a,b,c: cm(a,b,c,beta[iters]),[0]*NF)
        if iters == len(beta)-1:
            solver.add_inequality_mconstraint(constraints,[0]*2)
            maxeval = 30
        solver.set_maxeval(maxeval)
        x[:] = solver.optimize(x)
    data.append(x.copy())
    v = np.array(x[1:])

    opt.update_design([mapping(v)],beta=beta[-1])
    f0, _ = opt(need_gradient=False)
    results.append(np.real(f0))

    if mp.am_really_master():
        np.savez(filename+"_data.npz",data=data,results=results,beta_history=beta_history)

    opt.plot2D(False,eps_parameters={'resolution':100})
    plt.show()