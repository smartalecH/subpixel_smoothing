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

opt.update_design([dmx.mapping(xf)],beta=np.inf,eta=dmx.eta_d)

print(cv)
print(np.sum(cv_grad))
if 1:
    if mp.am_master():
        plt.figure()
        plt.imshow(dmx.mapping(xf).reshape((dmx.Nx,dmx.Ny)),interpolation='bilinear')
quit()