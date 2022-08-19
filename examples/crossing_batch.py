'''crossing_batch.py
design a crossing from various starting conditions
'''

import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
from matplotlib import pyplot as plt
import nlopt
import argparse
mp.verbosity(0)

import os, psutil

def get_memory():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

# ---------------------------------------- #
# CL parameters
# ---------------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument('-r','--resolution', help='Simulation resolution', default=20, type=float)
parser.add_argument('-i','--init_condition', help='Simulation resolution', default=0, type=int)
parser.add_argument('-m','--maxeval', help='Simulation resolution', default=30, type=int)
args = parser.parse_args()
# ---------------------------------------- #
# basic parameters
# ---------------------------------------- #

resolution = args.resolution
design_region_resolution = int(2*resolution)

dx = 3.0
dy = 3.0
filename = "crossing_batch_{}_{}_{}".format(args.resolution,args.init_condition,args.maxeval)
min_length = 0.09

eta = 0.5
eta_e = 0.75

dpml = 1.0

maxeval = args.maxeval

data = []
results = []
beta_history = []
print("start: ",get_memory())
# ---------------------------------------- #
# derived quantities
# ---------------------------------------- #

eta_d = 1-eta_e
filter_radius = mpa.get_conic_radius_from_eta_e(min_length,eta_e)
c = (design_region_resolution*filter_radius)**4

sx = dx + 2 + 2*dpml
sy = dy + 2 + 2*dpml

# ---------------------------------------- #
# main routine
# ---------------------------------------- #


silicon = mp.Medium(epsilon=12)
cell_size = mp.Vector3(sx,sy,0)
boundary_layers = [mp.PML(thickness=dpml)]

eig_parity = mp.EVEN_Y + mp.ODD_Z

design_region_size = mp.Vector3(dx,dy)
Nx = int(design_region_resolution*design_region_size.x) + 1
Ny = int(design_region_resolution*design_region_size.y) + 1
print("size ",Nx*Ny)

w = 0.5
sep = 0.5
wgy = sep/2 + w/2
waveguide_geometry = [mp.Block(material=silicon,
                               size=mp.Vector3(mp.inf,w,mp.inf)),
                      mp.Block(material=silicon,
                               size=mp.Vector3(w,mp.inf,mp.inf)) ]
fcen = 1/1.55
df = 0.3*fcen
frequencies = 1/np.linspace(1.5,1.6,10)
sources = [mp.EigenModeSource(src=mp.GaussianSource(fcen,fwidth=df),
                              center=mp.Vector3(-0.5*sx+dpml+0.1,0),
                              size=mp.Vector3(0,sy-2*dpml),
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

obj_list = [mpa.EigenmodeCoefficient(sim,
                                    mp.Volume(center=mp.Vector3(-0.5*sx+dpml+0.2),
                                    size=mp.Vector3(0,sy-2*dpml,0)),
                                    1,
                                    eig_parity=eig_parity),
            mpa.EigenmodeCoefficient(sim,
                                    mp.Volume(center=mp.Vector3(0.5*sx-dpml-0.2),
                                    size=mp.Vector3(0,sy-2*dpml,0)),
                                    1,
                                    eig_parity=eig_parity)]

def J(input,output):
    return npa.mean(1-npa.power(npa.abs(output/input),2))

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=J,
    objective_arguments=obj_list,
    design_regions=[matgrid_region],
    frequencies=frequencies)


'''opt.plot2D()
plt.show()
quit()'''

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
print("mask: ",get_memory())
def mapping(x):
    x = x.reshape(Nx,Ny)
    x = npa.where(Si_mask,1,npa.where(SiO2_mask,0,x))
    x = mpa.conic_filter(x,
                        filter_radius,
                        design_region_size.x,
                        design_region_size.y,
                        design_region_resolution)
    x = (x + npa.rot90(x) + npa.rot90(npa.rot90(x)) + npa.rot90(npa.rot90(npa.rot90(x)))) / 4
    x = (x + npa.flipud(x)) / 2
    x = (x + npa.fliplr(x)) / 2
    x = npa.clip(x,0,1)
    return x.flatten()

def f(x, grad, beta):
    data.append(x.copy())

    #opt.design_regions[0].design_parameters.do_averaging = True
    opt.update_design([mapping(x)],beta=beta)
    
    f0, dJ_du = opt()
    print(dJ_du.shape)
    dJ_du_temp = np.sum(dJ_du,axis=1)
    bp_adjsol_grad = tensor_jacobian_product(mapping,0)(x,dJ_du_temp)
    if grad.size > 0:
        grad[:] = np.squeeze(bp_adjsol_grad)
    
    opt.plot2D(False,eps_parameters={'resolution':100})
    if mp.am_master():
        plt.savefig(filename+"_geom.png")
    if mp.am_master():
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(grad.reshape(Nx,Ny))
        plt.subplot(1,2,2)
        plt.imshow(x.reshape(Nx,Ny))
    plt.savefig(filename+"_grad.png")
    plt.close("all")

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


print("c: ",get_memory())

# initial guess
x = np.linspace(-dx/2,dy/2,Nx)
y = np.linspace(-dx/2,dy/2,Ny)
X, Y = np.meshgrid(x,y)
Z = np.zeros((Nx,Ny))
mask = (np.abs(Y)  <= w/2) + (np.abs(X)  <= w/2)
p = np.zeros((Nx,Ny))
p[mask.T] = 1
print("init: ",get_memory())
# -------------------------------------- #
# initial conditions
# -------------------------------------- #

def draw_circles(radius,phasex=0,phasey=0):
    r = int(radius*design_region_resolution)
    p = 2*r
    px = int(phasex*design_region_resolution)
    py = int(phasey*design_region_resolution)
    x = np.ones((Nx,Ny))
    for ix in range(Nx):
        for iy in range(Ny):
            if ((((ix-px) % (p))-p/2)**2 + (((iy-py) % (p))-p/2)**2 <= r**2):
                x[ix,iy] = 0
    return x

# traditional TO
if args.init_condition == 0:
    x = mapping(np.ones((Nx,Ny))*0.5)
# crossing
if args.init_condition == 1:
    x = mapping(p)
# small circles
elif args.init_condition == 2:
    y = draw_circles(0.6*0.25,0.6*0.14,0.0)
    print(y)
    x = mapping(y)
    x= mapping(mpa.tanh_projection(x,np.inf,0.5))
# large circles
elif args.init_condition == 3:
    y = draw_circles(0.25,0.14,0)
    x = mapping(y)
    x= mapping(mpa.tanh_projection(x,np.inf,0.5))
# xlarge circles
elif args.init_condition == 4:
    y = draw_circles(1.5*0.25,1.5*0.14,0)
    x = mapping(y)
    x= mapping(mpa.tanh_projection(x,np.inf,0.5))

print("end: ",get_memory())
opt.update_design([x],beta=np.inf)
opt.plot2D(eps_parameters={'resolution':100})
plt.show()
quit()
if __name__ == '__main__':
    algorithm = nlopt.LD_MMA
    n = Nx * Ny
    scale = 2
    if args.init_condition == 0:
        beta = [8, 32, np.inf]
    else:
        scale = 4
        beta = [np.inf]
    
    for iters in range(len(beta)):
        solver = nlopt.opt(algorithm, n)
        solver.set_lower_bounds(0)
        solver.set_upper_bounds(1)
        solver.set_min_objective(lambda a,g: f(a,g,beta[iters]))
        solver.set_maxeval(maxeval)
        if iters == len(beta)-1:
            solver.set_maxeval(scale*maxeval)
        #if iters == len(beta)-1:
        #    solver.add_inequality_mconstraint(constraints,[0]*2)

        x[:] = solver.optimize(x)
    data.append(x.copy())

    opt.update_design([mapping(x)],beta=beta[-1])
    f0, _ = opt(need_gradient=False)
    results.append(float(np.real(f0)))

    if mp.am_really_master():
        np.savez(filename+"_data.npz",data=data,results=results,beta_history=beta_history)

    opt.plot2D(False,eps_parameters={'resolution':100})
    plt.savefig(filename+"_finalfig.png",dpi=200)
    #plt.show()