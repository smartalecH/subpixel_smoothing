'''vis_smoothing.py
Plot a simple structure and visualize the
effects of the smoothing on the structure.
'''
import meep as mp
from meep import adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product
from enum import Enum
from matplotlib import pyplot as plt

# ---------------------------------------- #
# basic parameters
# ---------------------------------------- #

resolution = 10
design_region_resolution = int(2*resolution)
dx = 2.0
dy = 2.0
filename = "vis_smoothing"
filter_radius = 0.5

# ---------------------------------------- #
# main routine
# ---------------------------------------- #

np.random.seed(1234)

silicon = mp.Medium(epsilon=12)
sxy = 5.0
cell_size = mp.Vector3(sxy,sxy,0)

dpml = 1.0
boundary_layers = [mp.PML(thickness=dpml)]
design_region_size = mp.Vector3(dx,dy)
Nx = int(design_region_resolution*design_region_size.x) + 1
Ny = int(design_region_resolution*design_region_size.y) + 1

def mapping(x):
    x = x.reshape(Nx,Ny)
    x = mpa.conic_filter(x,
                        filter_radius,
                        design_region_size.x,
                        design_region_size.y,
                        design_region_resolution)
    return x.flatten()

init = np.random.rand(Nx,Ny)
init_filter = mapping(init)

matgrid = mp.MaterialGrid(mp.Vector3(Nx,Ny),
                            mp.air,
                            silicon,
                            beta=np.inf,
                            weights=init_filter,
                            do_averaging=False)

matgrid_region = mpa.DesignRegion(matgrid,
                                    volume=mp.Volume(center=mp.Vector3(),
                                                    size=mp.Vector3(design_region_size.x,design_region_size.y,0)))

geometry = [mp.Block(center=matgrid_region.center,
                                size=matgrid_region.size,
                                material=matgrid)]

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    geometry=geometry)

#sim.plot2D(output_plane=mp.Volume(center=mp.Vector3(0.5,0.7),size=mp.Vector3(0.5,0.5)),eps_parameters={'resolution':10})

get_array_metadata(self,
                       vol=None,
                       center=None,
                       size=None,
                       dft_cell=None,
                       return_pw=False)

sim.init_sim()
n = 5
x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)
e2 = np.zeros([n, n], dtype=np.complex128)
e1 = np.zeros([n, n], dtype=np.complex128)
for i in range(len(x)):
    for j in range(len(y)):
        v3 = mp.py_v3_to_vec(sim.dimensions, mp.Vector3(x[i], y[j], 0), sim.is_cylindrical)
        e2[i, j] = sim.fields.get_inveps(mp.Ex, mp.Y, v3)
        e1[i, j] = sim.fields.get_inveps(mp.Ey, mp.X, v3)

plt.show()