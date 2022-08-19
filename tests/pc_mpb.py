import meep as mp
from meep import mpb
import numpy as np
import pc_convergence as pc

rad = 0.293751974
geometry = [mp.Cylinder(rad, material=mp.Medium(index=3.5))]
geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))


resolution = [8,16,32,64,128,512,1024,1024*2,1024*4]
freqs = []
for r in resolution:
    ms = mpb.ModeSolver(num_bands=1,
                        k_points=[mp.Vector3(0.3892,0.1597,0)],
                        geometry=geometry,
                        geometry_lattice=geometry_lattice,
                        resolution=r)

    ms.run_te()
    freqs.append(ms.freqs)
    print(ms.freqs)

print(freqs)
np.savez("mpb_results.npz",freqs=freqs,resolution=resolution)