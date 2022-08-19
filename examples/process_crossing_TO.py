import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
from matplotlib import pyplot as plt
import nlopt
from utils import fit_initial_params
mp.quiet()
import crossing_TO
import matplotlib.gridspec as gridspec
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import colors

crossing_data = np.load("cross_TO_data.npz")
x = crossing_data['data']
results = crossing_data['results']

iteration = np.arange(1,x.shape[0]+1)
beta = [8, 32, np.inf]
beta_history = []
for iters in range(len(beta)):
    for i in range(crossing_TO.maxeval):
        beta_history.append(beta[iters])
# ----------------------------------------------- #
#               Evolution plot                    #
# ----------------------------------------------- #
if 1:
    fig = plt.figure(figsize=(5.25,4.0), constrained_layout=True)
    gs0 = gridspec.GridSpec(3, 1, figure=fig, hspace=0.1, wspace=0, height_ratios=[0.25,0.25,0.5])

    # view the evolution of various designs using TO
    iters = [0,3,10,16,19]
    gs00 = gridspec.GridSpecFromSubplotSpec(1, len(iters), subplot_spec=gs0[0], hspace=0, wspace=0)
    for i,ki in enumerate(iters):
        ax = fig.add_subplot(gs00[i])
        crossing_TO.opt.update_design([crossing_TO.mapping(x[ki,:])],beta=beta_history[ki])
        crossing_TO.opt.plot2D(False,output_plane=mp.Volume(size=mp.Vector3(2,2)),
            eps_parameters={'resolution':100})
        scalebar = ScaleBar(1, "um", length_fraction=0.5,location='lower left',box_alpha=0,width_fraction=0.04)
        ax.add_artist(scalebar)
        plt.axis("off")
    
    # view the evolution of various designs using shape optimization
    iters = [21,24,26,29,-1]
    gs00 = gridspec.GridSpecFromSubplotSpec(1, len(iters), subplot_spec=gs0[1], hspace=0, wspace=0)
    for i,ki in enumerate(iters):
        ax = fig.add_subplot(gs00[i])
        crossing_TO.opt.update_design([crossing_TO.mapping(x[ki,:])],beta=beta_history[ki])
        crossing_TO.opt.plot2D(False,output_plane=mp.Volume(size=mp.Vector3(2,2)),
            eps_parameters={'resolution':100})
        scalebar = ScaleBar(1, "um", length_fraction=0.5,location='lower left',box_alpha=0,width_fraction=0.04)
        ax.add_artist(scalebar)
        plt.axis("off")
    
    # add the iteration plot

    crossing_data = np.load("cross_TO_no_smoothing_data.npz")
    x = crossing_data['data']
    results = crossing_data['results']

    crossing_data_smoothing = np.load("cross_TO_smoothing_data.npz")
    x_smoothing = crossing_data_smoothing['data']
    results_smoothing = crossing_data_smoothing['results']

    gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[0.6,0.4], subplot_spec=gs0[2], hspace=0, wspace=0)
    ax = fig.add_subplot(gs00[0])
    #plt.plot(10*np.log10(1-results),'-o')
    plt.plot((1-results)*100,'-o')
    #plt.plot(10*np.log10(1-results_smoothing),'-o')
    plt.plot((1-results_smoothing)*100,'-o')
    plt.ylabel("Transmission (%)")
    plt.xlabel("Iteration")
    ax.tick_params(which='both',direction='in')
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.yaxis.set_major_locator(plt.MaxNLocator(8))
    #plt.yticks([-1.5,-1,-0.5,0])
    #plt.grid(True)
    #plt.title("(b)")

    # add the steady state plot
    ax = fig.add_subplot(gs00[1])
    crossing_TO.opt.update_design([crossing_TO.mapping(x[-1,:])],beta=beta_history[-1])
    crossing_TO.opt.sim.sources[0].src=mp.ContinuousSource(wavelength=1.55)
    crossing_TO.opt.sim.run(until=200)
    crossing_TO.opt.plot2D(False, fields=mp.Ez,
    output_plane=mp.Volume(size=mp.Vector3(3.5,3.5)),
    plot_boundaries_flag=False,plot_sources_flag=False,
        eps_parameters={'resolution':100},
        field_parameters={'alpha':0.80})
    scalebar = ScaleBar(1, "um", length_fraction=0.5,location='lower left',box_alpha=0,width_fraction=0.04)
    ax.add_artist(scalebar)
    plt.axis("off")
    #plt.title("(c)")

    if mp.am_master():
        plt.savefig("crossing_TO_smoothing.svg",dpi=300)
        plt.show()