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

cell_size = mp.Vector3(1,1,0)

# ----------------------------------- #
# Material grid setup
# ----------------------------------- #

def get_geometry(rad,smoothing):
        design_shape = mp.Vector3(1,1,0)
        design_region_resolution = 5000
        Nx = int(design_region_resolution*design_shape.x)
        Ny = int(design_region_resolution*design_shape.y)
        x = np.linspace(-0.5*cell_size.x,0.5*cell_size.x,Nx)
        y = np.linspace(-0.5*cell_size.y,0.5*cell_size.y,Ny)
        xv, yv = np.meshgrid(x,y)
        design_params = np.sqrt(np.square(xv) + np.square(yv)) < rad
        design_params_sdf = make_sdf(design_params)
        if smoothing is False:
                design_params_sdf = mpa.tanh_projection(design_params_sdf,np.inf,0.5)

        matgrid = mp.MaterialGrid(mp.Vector3(Nx,Ny),
                                mp.air,
                                mp.Medium(index=3.5),
                                weights=design_params_sdf,
                                do_averaging=smoothing,
                                beta=np.inf,
                                eta=0.5)  

        geometry = [mp.Block(center=mp.Vector3(),
                        size=mp.Vector3(design_shape.x,design_shape.y,0),
                        material=matgrid)]

        #geometry = [mp.Cylinder(radius=rad,material=mp.Medium(index=3.5))]
    
        return geometry


res = 30 ## pixels/um  

fcen = 0.3
df = 0.2*fcen
sources = [mp.Source(mp.GaussianSource(fcen,fwidth=df),
                     component=mp.Hz,
                     center=mp.Vector3(-0.1057,0.2094,0))]

k_point = mp.Vector3(0.3892,0.1597,0)

sim = mp.Simulation(resolution=res,
                    cell_size=cell_size,
                    eps_averaging=True,
                    sources=sources,
                    k_point=k_point)

'''sim.plot2D()
plt.show()
quit()'''

# ----------------------------------- #
# Simulation resolution sweep
# ----------------------------------- #

N = 12
radius = np.linspace(0.295,0.305,30)#[0.3,0.305,0.31, 0.315, 0.32]
frequencies_s = []
frequencies_n = []
Q_s = []
Q_n = []

for smoothing in [True, False]:
        for r in radius:
                geometry = get_geometry(r,smoothing)
                sim.geometry = geometry
                sim.eps_averaging = smoothing
                sim.reset_meep()
                h = mp.Harminv(mp.Hz, mp.Vector3(0.3718,-0.2076), fcen, df)

                sim.run(mp.after_sources(h),
                        until_after_sources=200)

                cur_freqs = []
                cur_Q = []
                for m in h.modes:
                        cur_freqs.append(m.freq); cur_Q.append(m.Q)
                        print("harminv:, {}, {}, {}".format(r,m.freq,m.Q))
                if smoothing:
                        frequencies_s.append(cur_freqs)
                        Q_s.append(cur_Q)
                else:
                        frequencies_n.append(cur_freqs)
                        Q_n.append(cur_Q)
if mp.am_master():
        np.savez("pc_continuous_new_fine.npz",radius=radius,Q_s=Q_s,Q_n=Q_n,frequencies_s=frequencies_s,frequencies_n=frequencies_n)
        