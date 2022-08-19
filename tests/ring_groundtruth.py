import numpy as np
import meep as mp
from meep import adjoint as mpa
import skfmm
from scipy import interpolate
import matplotlib.pyplot as plt

def make_sdf(data):
    '''
    Assume the input, data, is the desired output shape
    (i.e. 1d, 2d, or 3d) and that it's values are between
    0 and 1.
    '''
    # create signed distance function
    sd = skfmm.distance(data- 0.5 , dx = 1)

    # interpolate zero-levelset onto 0.5-levelset
    x = [np.min(sd.flatten()), 0, np.max(sd.flatten())]
    y = [0, 0.5, 1]
    f = interpolate.interp1d(x, y, kind='linear')

    return f(sd)


# ----------------------------------- #
# Simulation setup
# ----------------------------------- #

n = 3.4                 # index of waveguide
w = 1                   # width of waveguide
pad = 4                 # padding between waveguide and edge of PML
dpml = 2                # thickness of PML
res = 10 ## pixels/um 

# ----------------------------------- #
# Material grid setup
# ----------------------------------- #

def get_geometry(rad,smoothing):
        sxy = 2*(rad+w+pad+dpml)

        design_shape = mp.Vector3(rad+w+pad,rad+w+pad,0)
        design_region_resolution = 500
        Nx = int(design_region_resolution*design_shape.x)
        Ny = int(design_region_resolution*design_shape.y)
        x = np.linspace(-0.5*design_shape.x,0.5*design_shape.x,Nx)
        y = np.linspace(-0.5*design_shape.y,0.5*design_shape.y,Ny)
        xv, yv = np.meshgrid(x,y)
        design_params = ((np.sqrt(np.square(xv) + np.square(yv)) <= (rad+w)) &
                (np.sqrt(np.square(xv) + np.square(yv)) >= rad)
        )
        design_params_sdf = make_sdf(design_params)
        if smoothing is False:
                design_params_sdf = mpa.tanh_projection(design_params_sdf,np.inf,0.5)

        # pulse center frequency (from third-order polynomial fit)
        fcen = -0.018765*rad**3 + 0.137685*rad**2 -0.393918*rad + 0.636202
        # pulse frequency width
        df = 0.1*fcen

        src = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                        component=mp.Hz,
                        center=mp.Vector3(rad+0.1*w)),
                mp.Source(mp.GaussianSource(fcen, fwidth=df),
                        component=mp.Hz,
                        center=mp.Vector3(-(rad+0.1*w)),
                        amplitude=-1)]

        geometry = [mp.Cylinder(material=mp.Medium(index=n),
                        radius=rad+w,
                        height=mp.inf,
                        center=mp.Vector3()),
        mp.Cylinder(material=mp.vacuum,
                        radius=rad,
                        height=mp.inf,
                        center=mp.Vector3())]

        sim = mp.Simulation(cell_size=mp.Vector3(sxy,sxy),
                        geometry=geometry,
                        eps_averaging=smoothing,
                        sources=src,
                        resolution=res,
                        symmetries=symmetries,
                        boundary_layers=[mp.PML(dpml)])