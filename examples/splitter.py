import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
import nlopt
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from scipy import special, signal
import pickle
import argparse
from engine import Data, run_opt, get_args, mapping

mp.quiet(quietval=True)
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.44)

def get_filename_prefix(args):
    return "{}splitter_rs-{}_rv-{}".format(args.fprefix,args.r_s,args.r_v)

def setup_sim(args,d):

    d.solid_radius = args.r_s/2
    d.void_radius = args.r_v/2

    waveguide_width = args.wgw
    design_region_width = args.w
    design_region_height = args.h
    waveguide_length = 1.0

    pml_size = args.dpml

    resolution = args.res

    frequencies = np.array([1/1.55])
    d.frequencies = frequencies
    d.design_region_resolution = int(args.dres*resolution)

    Sx = 2*pml_size + design_region_width + 2
    Sy = 2*pml_size + design_region_height + 1
    cell_size = mp.Vector3(Sx,Sy)

    pml_layers = [mp.PML(pml_size)]

    fcen = np.mean(frequencies)
    width = 0.2
    fwidth = width * fcen
    source_center  = [-design_region_width/2 - 0.75,0,0]
    source_size    = mp.Vector3(0,Sy,0)
    kpoint = mp.Vector3(1,0,0)
    src = mp.GaussianSource(frequency=fcen,fwidth=fwidth)
    source = [mp.EigenModeSource(src,
                        eig_band = 1,
                        direction=mp.NO_DIRECTION,
                        eig_kpoint=kpoint,
                        size = source_size,
                        center=source_center)]

    d.Nx = int(d.design_region_resolution*design_region_width) + 1
    d.Ny = int(d.design_region_resolution*design_region_height) + 1
    init = np.ones((d.Nx,d.Ny))*0.5
    design_variables = mp.MaterialGrid(mp.Vector3(d.Nx,d.Ny),SiO2,Si,init,grid_type='U_MEAN',do_averaging=False)
    design_region = mpa.DesignRegion(design_variables,volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(design_region_width, design_region_height, args.wgt)))
    wgy = args.sep/2 + args.wgw/2
    geometry = [
        mp.Block(center=mp.Vector3(x=-Sx/4), material=Si, size=mp.Vector3(Sx/2, args.wgw, args.wgt)), # input waveguide
        mp.Block(center=mp.Vector3(Sx/2,y=wgy), material=Si, size=mp.Vector3(Sx, args.wgw, args.wgt)),  # top waveguide
        mp.Block(center=mp.Vector3(Sx/2,y=-wgy), material=Si, size=mp.Vector3(Sx, args.wgw, args.wgt)),  # top waveguide
        mp.Block(center=design_region.center, size=design_region.size, material=design_variables),
        mp.Block(center=design_region.center, size=design_region.size, material=design_variables,e2=mp.Vector3(y=-1))
    ]

    sim = mp.Simulation(cell_size=cell_size,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=source,
                        default_material=SiO2,
                        resolution=resolution)

    mode = 1
    mon_size = args.wgw+2*args.sep - 0.1
    TE0 = mpa.EigenmodeCoefficient(sim,mp.Volume(center=mp.Vector3(x=-design_region_width/2 - 0.5),size=mp.Vector3(y=Sy)),mode)
    TE_top = mpa.EigenmodeCoefficient(sim,mp.Volume(center=mp.Vector3(design_region_width/2 + 0.5,wgy),size=mp.Vector3(y=mon_size)),mode)
    TE_bot = mpa.EigenmodeCoefficient(sim,mp.Volume(center=mp.Vector3(design_region_width/2 + 0.5,-wgy),size=mp.Vector3(y=mon_size)),mode)
    ob_list = [TE0,TE_top,TE_bot]

    def J(source,top,bot):
        power = npa.abs(top/source)**2 + npa.abs(bot/source)**2
        return 1-power

    opt = mpa.OptimizationProblem(
        simulation = sim,
        objective_functions = J,
        objective_arguments = ob_list,
        design_regions = [design_region],
        frequencies=frequencies
    )

    x_g = np.linspace(-design_region_width/2,design_region_width/2,d.Nx)
    y_g = np.linspace(-design_region_height/2,design_region_height/2,d.Ny)
    X_g, Y_g = np.meshgrid(x_g,y_g,sparse=True,indexing='ij')

    left_wg_mask = (X_g == -design_region_width/2) & (np.abs(Y_g) <= waveguide_width/2)
    top_wg_mask = (X_g == design_region_width/2) & (np.abs(Y_g-wgy) <= waveguide_width/2)
    bot_wg_mask = (X_g == design_region_width/2) & (np.abs(Y_g+wgy) <= waveguide_width/2)
    d.Si_mask = left_wg_mask | top_wg_mask | bot_wg_mask

    border_mask = ((X_g == -design_region_width/2) | 
                (X_g == design_region_width/2) | 
                (Y_g == -design_region_height/2) | 
                (Y_g == design_region_height/2))
    d.SiO2_mask = border_mask.copy()
    d.SiO2_mask[d.Si_mask] = False

    '''init = np.ones((d.Nx,d.Ny))*0.5
    alpha = 1e-5
    beta = 8
    opt.update_design([mapping(init,alpha,beta,args,d)])
    opt.plot2D(True)
    plt.show()
    quit()'''

    return opt

def save_data(v,args,d,opt):
    # visualize
    plt.figure(figsize=(4*run_data.num_groups,4))
    ax = plt.gca()
    eps = np.real(opt.sim.get_array(mp.Dielectric))
    eps = mp.merge_subgroup_data(eps)
    for k in range(run_data.num_groups):
        plt.subplot(1,run_data.num_groups,k+1)
        plt.imshow(np.rot90(eps[:,:,k]),cmap="binary")
        plt.axis("off")
    
    plt.savefig('{}geo.png'.format(d.filename_prefix))

    src, top, bot = opt.get_objective_arguments()
    
    src = mp.merge_subgroup_data(src)
    top = mp.merge_subgroup_data(top)
    bot = mp.merge_subgroup_data(bot)
    
    d.results_history.append([src,top])

    power = np.abs(top/src)**2*100

    plt.figure()
    plt.fill_between(1/d.frequencies,np.min(power,axis=-1),np.max(power,axis=-1),alpha=0.25)
    plt.plot(1/d.frequencies,np.mean(power,axis=-1),'o-')
    plt.grid(True)
    plt.xlabel('Wavelength')
    plt.ylabel("Transmission (%)")
    if mp.am_really_master():
        plt.savefig('{}freq.png'.format(d.filename_prefix))
        with open("{}data.pickle".format(d.filename_prefix),"wb") as f:
            pickle.dump((args,d), f)    
    
    # Evolution plot
    num_iters = len(d.results_history)
    total_power = np.abs(np.array(d.results_history)[:,1]/ np.array(d.results_history)[:,0])**2*100
    axis = (1,2)
    iter_list = np.arange(1,num_iters+1)
    plt.figure()
    plt.fill_between(iter_list,np.min(total_power,axis=axis),np.max(total_power,axis=axis),alpha=0.25)
    plt.plot(iter_list,np.mean(total_power,axis=axis),'o-')
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel("Transmission (%)")
    if mp.am_really_master():
        plt.savefig('{}_history.png'.format(d.filename_prefix))
    plt.close("all")
    mp.comm.barrier()

if __name__ == "__main__":
    num_groups = 1
    my_group = mp.divide_parallel_processes(num_groups)
    
    parser = get_args() # get common arguments

    # geometric parameters
    parser.add_argument('--wgw', type=float, default=0.5, help='Waveguide width')
    parser.add_argument('--wgt', type=float, default=0.22, help='Waveguide thickness')
    parser.add_argument('--w', type=float, default=3.0, help='Width of design region')
    parser.add_argument('--h', type=float, default=3.0, help='Height of design region')
    parser.add_argument('--sep',type=float, default=0.5, help='distance between output arms')
    parser.add_argument('--cladding',type=float, default=2.0, help='cladding thickness')
    parser.add_argument('--dpml',type=float, default=1.0, help='pml thickness')

    # simulation parameters
    parser.add_argument('--res', type=float, default=20.0, help='Resolution')
    parser.add_argument('--dres', type=int, default=2, help='Ratio of design region resolution to meep resolution')
    parser.add_argument('--three_d', action='store_true', default=False, help='3D sim? (default: False)')
    args = parser.parse_args()

    run_data = Data()
    run_data.filename_prefix = get_filename_prefix(args)
    run_data.num_groups = num_groups
    run_data.my_group = my_group
    run_data.my_filter_group = my_group

    opt = setup_sim(args,run_data)

    run_opt(args,run_data,opt,save_data)