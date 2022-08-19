import meep as mp
import meep.adjoint as mpa
import numpy as np
import jax
from autograd import numpy as npa
from autograd import tensor_jacobian_product
import nlopt
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from scipy import special, signal
import pickle
import argparse

class Data():
    '''An opaque structure to store important
    parameters and results from function 
    to function, as needed.
    You can always error-check with hasattr(a, 'field')'''
    pass

def mapping(x,args,d):
    x = npa.squeeze(x.reshape(d.Nx,d.Ny))
    x = npa.where(d.SiO2_mask,0,
        npa.where(d.Si_mask,1,x))
    radius = npa.max(npa.array([d.solid_radius,d.void_radius]))#*d.design_region_resolution
    x = mpa.conic_filter(x,radius,args.w,args.h,d.design_region_resolution)
    x = x.flatten()
    return x

# determine a good starting condition
def map_J(x, gradient, args, d):
    x0 = np.ones(d.Nx*d.Ny,)*x
    out = mapping(x0,10**args.alpha1,2**args.beta0,args,d)
    c = out.reshape(d.Nx,d.Ny)[int(d.Nx/2),int(d.Ny/2)]
    return np.abs(c-0.5)**2

'''def f(x, gradient):
    t = x[0] # "dummy" parameter
    v = x[1:] # design parameters
    if gradient.size > 0:
        gradient[0] = 1
        gradient[1:] = 0
    return t'''

def f(x,gradient,opt,beta,args,d,callback):
    print("Current iteration: {}; current beta: {}".format(d.cur_iter,beta))
    d.beta_history.append([beta])
    
    d.param_history.append(x.copy())

    opt.update_design([mapping(x,args,d)],beta=beta)
    opt.plot2D(eps_parameters={'resolution':100})
    if mp.am_really_master():
        plt.savefig('{}params.png'.format(d.filename_prefix),dpi=300)
        plt.close()

    f0, dJ_du = opt([mapping(x,args,d)],beta=beta)

    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping,0)(x,args,d,dJ_du)
    
    d.evaluation_history.append(np.real(f0))

    callback(x,args,d,opt)

    d.cur_iter = d.cur_iter + 1

    return float(np.real(f0))

def c(result,x,gradient,opt,args,d,alpha,beta,callback):
    print("Current iteration: {}; current alpha: {}; current beta: {}".format(d.cur_iter,alpha,beta))
    d.beta_history.append([beta])
    d.alpha_history.append([alpha])

    t = x[0] # dummy parameter
    v = x[1:] # design parameters
    if mp.am_really_master():
        plt.figure()
        plt.imshow(np.rot90(v.reshape(d.Nx,d.Ny)))
        plt.savefig('{}params.png'.format(d.filename_prefix),dpi=300)
    margs = [args,d]

    d.param_history.append(v.copy())
    v_temp = mp.merge_subgroup_data(v)
    if d.num_groups > 1:
        print("Are arrays equal: ",np.array_equal(v_temp[:,0],v_temp[:,1]),np.array_equal(v_temp[:,2],v_temp[:,1]))

    temp = mapping(v,*margs)

    print("min: {}, max: {}".format(np.min(temp),np.max(temp)))
    
    f0, dJ_du = opt([mapping(v,*margs)],beta=beta)
    
    # Backprop the gradients through our mapping function

    # Assign gradients
    if gradient.size > 0:
        gradient[:,0] = -1 # gradient w.r.t. "t"
        for k in range(opt.frequencies.size):
            gradient[k,1:] = tensor_jacobian_product(mapping,0)(v,*margs,dJ_du[:,k]) # gradient w.r.t. each frequency objective
    
    print(f0)
    print("t: ",t)
    result[:] = np.real(f0) - t
    
    # store results
    d.evaluation_history.append(np.real(f0))

    callback(v,args,d,opt)
        
    d.cur_iter = d.cur_iter + 1
    print("Computing next step...")

def _centered(arr, newshape):
    '''Helper function that reformats the padded array of the fft filter operation.
    Borrowed from scipy:
    https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py#L263-L270
    '''
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def glc(result,x,gradient,beta,args,d,opt):
    t = x[0] # dummy parameter
    v = x[1:] # design parameters
    
    radius = npa.max(npa.array([d.solid_radius,d.void_radius]))*d.design_region_resolution
    filter_f = lambda a: mpa.conic_filter(a,2*radius)
    threshold_f = lambda a: mpa.tanh_projection(a,beta,0.5)
    eta_e = 0.75
    eta_d = 0.25
    c0 = (2*radius)**4
    M1 = lambda a: mpa.constraint_solid(a.reshape(d.Nx,d.Ny),c0,eta_e,filter_f,threshold_f,1)
    M2 = lambda a: mpa.constraint_void(a.reshape(d.Nx,d.Ny),c0,eta_d,filter_f,threshold_f,1)

    a1 = args.a1

    g1 = mp.merge_subgroup_data(grad(M1)(v))[...,0]
    g2 = mp.merge_subgroup_data(grad(M2)(v))[...,0]
    
    result[0] = mp.merge_subgroup_data(M1(v))[...,0] - a1*t
    result[1] = mp.merge_subgroup_data(M2(v))[...,0] - a1*t
    gradient[0,1:] = g1.flatten()
    gradient[1,1:] = g2.flatten()
    gradient[:,0] = -a1

    if mp.am_really_master():
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(np.rot90(g1.reshape(d.Nx,d.Ny)))
        plt.subplot(1,2,2)
        plt.imshow(np.rot90(g2.reshape(d.Nx,d.Ny)))
        plt.savefig("{}_geoc.png".format(d.filename_prefix))
    print(result)

def run_opt(args,d,opt,save_data):

    d.evaluation_history = []
    d.cur_iter = 1
    d.results_history = []
    d.beta_history = []
    d.param_history = []
    d.alpha_history = []

    algorithm = nlopt.LD_MMA
    n = d.Nx * d.Ny # number of parameters

    # Initial guess
    if args.init > 1:
        np.random.seed(args.init)
        x = np.random.rand(n)
    else:
        x = np.ones((n,)) * args.init
    x[d.Si_mask.flatten()] = 1 # set the edges of waveguides to silicon
    x[d.SiO2_mask.flatten()] = 0 # set the other edges to SiO2

    # lower and upper bounds
    lb = np.zeros((d.Nx*d.Ny,))
    lb[d.Si_mask.flatten()] = 1
    ub = np.ones((d.Nx*d.Ny,))
    ub[d.SiO2_mask.flatten()] = 0

    # insert dummy parameter bounds and variable
    '''x = np.insert(x,0,1.0) # our initial guess for the worst error
    lb = np.insert(lb,0,0) # we can't get less than 0 error!
    ub = np.insert(ub,0,np.inf) # we can't get more than 1 error!'''
    
    load_args = False
    if args.load is not "":
        filename = args.load#"testbench.pickle"
        ld_idx = args.ld_idx
        my_filter_group = d.my_filter_group
        my_group = d.my_group
        variation=args.variation
        update_factor=args.update_factor
        a1=args.a1
        args, d = pickle.load( open( filename, "rb" ) )
        d.my_filter_group=my_filter_group
        d.my_group=my_group
        beta = [32]
        args.a1=a1
        args.update_factor=update_factor
        args.nalpha = 1
        args.variation=variation
        x[1:] = d.param_history[-1]
        load_args = True
    
    alpha = 1e-3
    beta = [8, 16, 32]
    for iters in range(len(beta)):
        solver = nlopt.opt(algorithm, n)
        solver.set_lower_bounds(lb)
        solver.set_upper_bounds(ub)
        solver.set_min_objective(lambda a,g: f(a,g,opt,beta[iters],args,d,save_data))
        #solver.set_param("inner_maxeval", 3)
        solver.set_param("dual_ftol_rel", 1e-8)
        #solver.set_xtol_rel(args.xtol)
        solver.set_maxeval(args.update_factor)
        
        # add quality measure constraints on last run
        runall = True#True if iters == (args.nalpha-1) else False
        num_c = 3 if runall else 1
        #solver.add_inequality_mconstraint(lambda r,x,g: M(r,x,g,args,d,alpha[iters],beta[iters],runall),[1e-5]*num_c)
        
        # add area constraints on specified run
        #if (iters >= args.nalpha+1):
        if load_args or (iters >= args.nalpha-1):
            print("adding constraints")       
            # add the geometric length constraints
            #solver.add_inequality_mconstraint(lambda r,x,g: glc(r,x,g,beta[iters],args,d,opt),[1e-8]*2)
        # Main objective function constraints
        #solver.add_inequality_mconstraint(lambda r,x,g: c(r,x,g,opt,args,d,alpha,beta[iters],save_data), np.array([1e-8]*opt.nf*d.num_groups))

        x[:] = solver.optimize(x)

    print("-----------------------------")
    print("Finalizing...")
    print("-----------------------------")

    result = np.zeros((opt.nf*d.num_groups,))
    gradient = np.zeros((opt.nf*d.num_groups,x.size))
    c(result,x,gradient,opt,args,d,alpha,beta[-1],save_data)

    print("-----------------------------")
    print("Finished")
    print("-----------------------------")

def get_args():
    parser = argparse.ArgumentParser()

    # topology optimization parameters
    parser.add_argument('--r_s', type=float, default=0.09, help='Minimum linewidth')
    parser.add_argument('--r_v', type=float, default=0.09, help='Minimum spacing')
    parser.add_argument('--a_s', type=float, default=0.08, help='Minimum area')
    parser.add_argument('--a_v', type=float, default=0.08, help='Minimum enclosed area')
    parser.add_argument('--area_c', type=int, default=0, help='Enforce area constraints?')
    parser.add_argument('--p', type=float, default=1.0, help='P norm of mapping function')
    parser.add_argument('--alpha0', type=float, default=-5, help='Smallest alpha')
    parser.add_argument('--alpha1', type=float, default=2, help='Largest alpha')
    parser.add_argument('--beta0', type=float, default=3, help='Smallest beta (power of 2)')
    parser.add_argument('--beta1', type=float, default=5, help='Largest beta (power of 2)')
    parser.add_argument('--nalpha', type=float, default=3, help='Number of alphas/betas')
    parser.add_argument('--update_factor', type=int, default=10, help='Number of iterations per alpha')
    parser.add_argument('--xtol', type=float, default=1e-6, help='Optimizer x tolerance')
    parser.add_argument('--a1', type=float, default=1e-5, help='')
    parser.add_argument('--variation', type=float, default=0.04, help='')
    parser.add_argument('--ld_idx', type=int, default=-1, help='')
    
    parser.add_argument('--fprefix', type=str, default="", help='Filename prefix')

    parser.add_argument('--load', type=str, default="", help='Load in existing data structure')

    parser.add_argument('--random', action='store_true', default=False, help='3D sim? (default: False)')
    parser.add_argument('--init', type=float, default=0.5, help='')


    return parser