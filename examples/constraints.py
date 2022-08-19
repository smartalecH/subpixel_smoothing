import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import tensor_jacobian_product, grad
from autograd import numpy as npa
from matplotlib import pyplot as plt
import nlopt
from utils import fit_initial_params
import demux as dmx
opt = dmx.opt
import matplotlib.colors as colors

#data = np.load(dmx.filename+"_data.npz")
data = np.load("demux_data/demux_30.0_3_50_data.npz")
results = data["results"]
beta_history = data["beta_history"]
x_history = data["data"]
xf = x_history[-1,:]

def bilinear_interpolation_grad():
    return

def indicator_solid(x, c, filter_f, resolution):
    filtered_field = filter_f(x)
    gradient_filtered_field = npa.gradient(filtered_field)
    grad_mag = (gradient_filtered_field[0] *
                resolution)**2 + (gradient_filtered_field[1] * resolution)**2
    if grad_mag.ndim != 2:
        raise ValueError(
            "The gradient fields must be 2 dimensional. Check input array and filter functions."
        )
    I_s = npa.exp(-c * grad_mag)
    return I_s

def constraint_solid(x, c, eta_e, filter_f, resolution):

    filtered_field = filter_f(x)
    I_s = indicator_solid(x.reshape(filtered_field.shape), c, filter_f,
                          resolution).flatten()
    return npa.mean(I_s * npa.minimum(filtered_field.flatten() - eta_e, 0)**2)


def indicator_void(x, c, filter_f, resolution):

    filtered_field = filter_f(x).reshape(x.shape)
    gradient_filtered_field = npa.gradient(filtered_field)
    grad_mag = (gradient_filtered_field[0] *
                resolution)**2 + (gradient_filtered_field[1] * resolution)**2
    if grad_mag.ndim != 2:
        raise ValueError(
            "The gradient fields must be 2 dimensional. Check input array and filter functions."
        )
    return npa.exp(-c * grad_mag)


def constraint_void(x, c, eta_d, filter_f, resolution):

    filtered_field = filter_f(x)
    I_v = indicator_void(x.reshape(filtered_field.shape), c, filter_f,
                          resolution).flatten()
    return npa.mean(I_v * npa.minimum(eta_d - filtered_field.flatten(), 0)**2)

opt.update_design([dmx.mapping(xf)],beta=np.inf)

filter_f = lambda x: mpa.conic_filter(x,
                        dmx.filter_radius,
                        dmx.design_region_size.x,
                        dmx.design_region_size.y,
                        dmx.design_region_resolution)

cf = lambda x: constraint_void(x, dmx.design_region_resolution**4, dmx.eta_d,filter_f,dmx.design_region_resolution)
cv = cf(xf)
cv_grad = grad(cf)(xf)
print(cv)
print(np.sum(cv_grad))
if 1:
    if mp.am_master():
        plt.figure()
        plt.imshow(dmx.mapping(xf).reshape((dmx.Nx,dmx.Ny)),interpolation='bilinear')
        
        plt.figure()
        plt.imshow(cv_grad.reshape((dmx.Nx,dmx.Ny)))
        plt.show()
quit()