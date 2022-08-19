import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
from matplotlib import pyplot as plt
import nlopt
from utils import fit_initial_params
mp.quiet()
import crossing
import matplotlib.gridspec as gridspec
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import colors

crossing_data = np.load("cross_new_data.npz",allow_pickle=True)
x = crossing_data['data']
results = crossing_data['results']
iteration = np.arange(1,x.shape[0]+1)

# ----------------------------------------------- #
#               Overview plot                     #
# ----------------------------------------------- #
if 0:
    fig = plt.figure(figsize=(5.25,2.2), constrained_layout=True)

    ax = plt.subplot(1,4,1)   
    plt.imshow(x[0,:].reshape(crossing.Nx,crossing.Ny),cmap='binary')
    plt.title("(a)")
    plt.axis("off")

    ax = plt.subplot(1,4,2)   
    plt.imshow(crossing.mapping(x[0,:]).reshape(crossing.Nx,crossing.Ny),cmap='binary')
    plt.title("(b)")
    plt.axis("off")
    
    ax = plt.subplot(1,4,3)
    crossing.opt.update_design([crossing.mapping(x[0,:])],beta=crossing.beta)
    crossing.opt.plot2D(False,output_plane=mp.Volume(size=mp.Vector3(2,2)),
            eps_parameters={'resolution':100})
    plt.axis("off")
    plt.title("(c)")
    scalebar = ScaleBar(1, "um", length_fraction=0.5,location='lower left',box_alpha=0,width_fraction=0.04)
    ax.add_artist(scalebar)
    
    f0, dJ_du = crossing.opt()
    bp_adjsol_grad = tensor_jacobian_product(crossing.mapping,0)(x[0,:],dJ_du)
    bp_adjsol_grad = bp_adjsol_grad.flatten()
    bp_adjsol_grad = bp_adjsol_grad-np.min(bp_adjsol_grad)
    bp_adjsol_grad = 2*bp_adjsol_grad/np.max(bp_adjsol_grad) - 1

    ax = plt.subplot(1,4,4)
    plt.imshow(bp_adjsol_grad.reshape(crossing.Nx,crossing.Ny),cmap='RdBu')
    plt.colorbar(aspect=7,location="bottom")
    plt.title("(d)")
    plt.axis("off")

    if mp.am_master():
        plt.savefig("shape_optimization.svg",dpi=300)
        plt.show()
# ----------------------------------------------- #
#               Evolution plot                    #
# ----------------------------------------------- #
if 1:
    fig = plt.figure(figsize=(5.25,3.0), constrained_layout=True)
    gs0 = gridspec.GridSpec(2, 1, figure=fig, hspace=0.0, wspace=0, height_ratios=[0.5,0.6])

    # view the evolution of various designs
    iters = [0,7,12,20,-1]
    gs00 = gridspec.GridSpecFromSubplotSpec(1, len(iters), subplot_spec=gs0[0], hspace=0, wspace=0)
    for i,ki in enumerate(iters):
        ax = fig.add_subplot(gs00[i])
        v = x[ki][1:] if i < (len(iters)-1) else x[ki]
        print(ki,v.shape)
        options = {}
        crossing.opt.update_design([crossing.mapping(v)])
        if i == (len(iters)-1):
            crossing.opt.update_design([crossing.mapping(x[-1])])
            crossing.opt.sim.sources[0].src=mp.ContinuousSource(wavelength=1.55)
            crossing.opt.sim.run(until=10)
            options = {'fields':mp.Ez,
            'field_parameters':{'alpha':0.80}}
            
        crossing.opt.plot2D(False,**options,output_plane=mp.Volume(size=mp.Vector3(3,3)),
            eps_parameters={'resolution':100})
        scalebar = ScaleBar(1, "um", length_fraction=0.5,location='lower left',box_alpha=0,width_fraction=0.04)
        ax.add_artist(scalebar)
        plt.axis("off")
        if i == 2:
            plt.title("(a)")

    # add the iteration plot
    gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[0.5,0.5], subplot_spec=gs0[1], hspace=0, wspace=0)
    ax = fig.add_subplot(gs00[0])
    result_end = results[-1,...].copy()
    results = results[0:-1,...]
    plt.fill_between(np.arange(1,results.shape[0]+1),
        np.min(100*(1-results),axis=1),
        np.max(100*(1-results),axis=1),
        alpha = 0.3
        )
    plt.plot(np.arange(1,results.shape[0]+1),np.mean(100*(1-results),axis=1))
    plt.ylabel("Transmission (%)")
    plt.xlabel("Iteration")
    plt.yticks([75,80,85,90,95])
    plt.xticks([0,5,10,15,20,25,30])
    plt.xlim([0,30])
    plt.title("(b)")
    plt.tick_params(which='both',direction='in')

    # add the steady state plot
    '''ax = fig.add_subplot(gs00[1])
    crossing.opt.update_design([crossing.mapping(x[-1])])
    crossing.opt.sim.sources[0].src=mp.ContinuousSource(wavelength=1.55)
    crossing.opt.sim.run(until=10)
    crossing.opt.plot2D(False, fields=mp.Ez,
    output_plane=mp.Volume(size=mp.Vector3(3.5,3.5)),
    plot_boundaries_flag=False,plot_sources_flag=False,
        eps_parameters={'resolution':100},
        field_parameters={'alpha':0.80})
    scalebar = ScaleBar(1, "um", length_fraction=0.5,location='lower left',box_alpha=0,width_fraction=0.04)
    ax.add_artist(scalebar)
    plt.axis("off")
    plt.title("(c)")'''

    ax = fig.add_subplot(gs00[1])
    plt.title("(c)")
    plt.plot(1/crossing.opt.frequencies,100*(1-result_end),'o-',markersize=5)
    plt.tick_params(which='both',direction='in')
    plt.ylabel("Transmission (%)")
    plt.yticks([87,88,89,90])
    plt.xlim([1.5,1.6])
    plt.xticks([1.5,1.52,1.54,1.56,1.58,1.6])
    plt.xlabel("Wavelength ($\mu$m)")
    #plt.yticks([70,75,80,85,90,95,100])

    if mp.am_master():
        plt.savefig("crossing_new_data.svg",dpi=300)
        plt.show()