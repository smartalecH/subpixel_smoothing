import matplotlib
import matplotlib.pyplot as plt
import meep as mp
import numpy as np
from meep import adjoint as mpa

res = 20                # pixels/μm                                                                               

n = 3.4                 # index of waveguide                                                                      
w = 1                   # width of waveguide                                                                      
pad = 4                 # padding between waveguide and edge of PML                                               
dpml = 2                # thickness of PML                                                                        
filter_radius = 0.2
h_freqs = []
use_smoothing = True
radii = np.linspace(1.8,1.81,6)
for rad in radii: #np.arange(1.800,2.001,0.005):
    sxy = 2*(rad+w+pad+dpml)  # cell size                                                                         

    cell_size = mp.Vector3(sxy,sxy)

    design_shape = mp.Vector3(sxy,sxy,0)
    design_region_resolution = 100
    Nx = int(design_region_resolution*design_shape.x) + 1
    Ny = int(design_region_resolution*design_shape.y) + 1
    x = np.linspace(-0.5*cell_size.x,0.5*cell_size.x,Nx)
    y = np.linspace(-0.5*cell_size.y,0.5*cell_size.y,Ny)
    xv, yv = np.meshgrid(x,y)

    design_params = np.logical_and(np.sqrt(np.square(xv) + np.square(yv)) > rad,
                                   np.sqrt(np.square(xv) + np.square(yv)) < rad+w,
                                   dtype=np.double)
    
    filt_design_params = mpa.conic_filter(design_params,filter_radius,sxy,sxy,design_region_resolution)

    if not use_smoothing:
        filt_design_params = mpa.tanh_projection(filt_design_params,np.inf,0.5)

    matgrid = mp.MaterialGrid(mp.Vector3(Nx,Ny),
                              mp.air,
                              mp.Medium(index=n),
                              weights=design_params,
                              do_averaging=use_smoothing,
                              beta=np.inf if use_smoothing else 0,
                              eta=0.5)

    geometry = [mp.Block(center=mp.Vector3(),
                         size=mp.Vector3(design_shape.x,design_shape.y,0),
                         material=matgrid)]

    # pulse center frequency (from third-order polynomial fit)                                                    
    fcen = -0.018765*rad**3 + 0.137685*rad**2 -0.393918*rad + 0.636202
    # pulse frequency width                                                                                       
    df = 0.02*fcen

    src = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                     component=mp.Hz,
                     center=mp.Vector3(rad+0.1*w)),
           mp.Source(mp.GaussianSource(fcen, fwidth=df),
                     component=mp.Hz,
                     amplitude=-1,
                     center=mp.Vector3(-(rad+0.1*w)))]

    symmetries = [mp.Mirror(mp.X,phase=+1),
                  mp.Mirror(mp.Y,phase=-1)]

    sim = mp.Simulation(cell_size=cell_size,
			geometry=geometry,
			sources=src,
            eps_averaging = use_smoothing,
			resolution=res,
			symmetries=symmetries,
			boundary_layers=[mp.PML(dpml)])
    
    sim.plot2D(eps_parameters={'resolution':100})
    plt.show()
    quit()

    h = mp.Harminv(mp.Hz, mp.Vector3(rad+0.1*w), fcen, df)
    sim.run(mp.after_sources(h),
            until_after_sources=200)
    print("h: ", h.modes[0].freq)
    h_freqs.append(h.modes[0].freq)

if use_smoothing:
    np.savez("ring_matgrid_smoothing.npz",h_freqs=h_freqs,radii=radii)
else:
    np.savez("ring_matgrid_no_smoothing.npz",h_freqs=h_freqs,radii=radii)
