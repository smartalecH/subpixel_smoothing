'''check_demux.py
visualize the results for the simple c-o band demultiplexer
'''

import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
from matplotlib import pyplot as plt
import nlopt
from utils import fit_initial_params
import demux as dmx
opt = dmx.opt
import matplotlib.colors as colors

#data = np.load(dmx.filename+"_data.npz")
data = np.load("demux_data/demux_30.0_2_50_data.npz")
results = data["results"]
beta_history = data["beta_history"]
x_history = data["data"]
xf = x_history[-1,:]


opt.update_design([dmx.mapping(xf)],beta=np.inf)

# plot response
if 0:
    N = 100
    frequencies = 1/np.concatenate((np.linspace(1.26,1.36,N),np.linspace(1.5,1.6,N)))
    opt.frequencies=frequencies
    opt(need_gradient=False)
    input, top_e, bot_e = opt.get_objective_arguments()
    power_top = np.abs(top_e/input)**2
    power_bot = np.abs(bot_e/input)**2
    if mp.am_master():
        plt.figure()
        plt.plot(1/frequencies,power_top)
        plt.plot(1/frequencies,power_bot)
        plt.show()
    quit()

# plot fields
if 1:
    srcs = opt.sim.sources
    opt.sim.sources = [srcs[0]]
    opt.sim.sources[0].src=mp.ContinuousSource(opt.frequencies[0])
    opt.sim.run(until=200)
    #opt.plot2D()

    opt.plot2D(fields=mp.Ez,field_parameters={'norm':colors.CenteredNorm()},eps_parameters={'resolution':100})
    plt.show()


