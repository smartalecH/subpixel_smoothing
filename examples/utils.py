import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import grad
from matplotlib import pyplot as plt
import nlopt


def fit_initial_params(x_init,x_desired,filter_func,projection_func,maxeval=20):
    
    def err(x):
        x_p = projection_func(filter_func(x))
        return npa.sum((x_p-x_desired)**2).flatten()
    
    def f(x, gradient):
        err_val = float(np.real(err(x)))
        print(err_val)
        grad_value = grad(err)(x)
        if mp.am_master():
            plt.figure()
            plt.imshow(grad_value.reshape(121,121))
        if gradient.size > 0:
            gradient[:] = grad_value
        return err_val
    
    
    solver = nlopt.opt(nlopt.LD_MMA, x_init.size)
    solver.set_lower_bounds(0)
    solver.set_upper_bounds(1)
    solver.set_min_objective(f)
    solver.set_maxeval(maxeval)
    x = solver.optimize(x_init)

    return x