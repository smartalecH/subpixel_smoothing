import numpy as np
import matplotlib.pyplot as plt

data_no_smoothing = np.load("ring_matgrid_no_smoothing.npz")
data_smoothing = np.load("ring_matgrid_smoothing.npz")

rad = np.linspace(1.8,1.81,1000)
fcen = -0.018765*rad**3 + 0.137685*rad**2 -0.393918*rad + 0.636202

plt.figure()
plt.plot(rad,fcen,label="exact")
plt.plot(data_smoothing['radii'],data_smoothing['h_freqs'],'-o',label="smoothing")
plt.plot(data_no_smoothing['radii'],data_no_smoothing['h_freqs'],'-o',label="no smoothing")
plt.legend()
plt.show()