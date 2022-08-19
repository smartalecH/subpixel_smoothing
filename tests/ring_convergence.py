import numpy as np
import meep as mp
from meep import adjoint as mpa
import skfmm
from scipy import interpolate
import matplotlib.pyplot as plt
import ring_continuous as rc



# ----------------------------------- #
# Simulation resolution sweep
# ----------------------------------- #

radius = 1.85071346789
N = 25
resolutions = np.logspace(4, 8, N, endpoint=True,base=2) #[10,20,40,80,160]
frequencies_s = []
frequencies_n = []
Q_s = []
Q_n = []

for smoothing in [True, False]:
        for r in resolutions:
                sim, fcen, df = rc.get_geometry(radius,smoothing)
                sim.resolution = r
                sim.reset_meep()
                h = mp.Harminv(mp.Hz, mp.Vector3(radius+0.1*rc.w), fcen, df)
                sim.run(mp.after_sources(h),
                        until_after_sources=300)

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
        np.savez("ring_convergence_new.npz",resolutions=resolutions,Q_s=Q_s,Q_n=Q_n,frequencies_s=frequencies_s,frequencies_n=frequencies_n)
        