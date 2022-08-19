import numpy as np
import meep as mp
from meep import adjoint as mpa
import skfmm
from scipy import interpolate
import matplotlib.pyplot as plt

from scipy.interpolate import pade

data = np.load("mpb_results.npz")

freqs = data["freqs"]
resolution = data["resolution"]

plt.figure()
plt.loglog(1/resolution[1:],freqs[1:],'o')
plt.show()