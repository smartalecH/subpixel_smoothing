import numpy as np
import meep as mp
from meep import adjoint as mpa
import skfmm
from scipy import interpolate
import matplotlib.pyplot as plt

plt.figure(figsize=(5.25,3.0),constrained_layout=True)

data = np.load("ring_convergence_new.npz",allow_pickle=True)

print(data["frequencies_s"])
print(data["frequencies_n"])

fs = np.array([a[0] for a in data["frequencies_s"]])
cal = fs[-1]
fs = np.abs(fs - cal) / cal
fn = np.array([a[0] for a in data["frequencies_n"]])
fn = np.abs(fn - cal) / cal

freqs = data["resolutions"][0:-1]

freqs_hat = np.logspace(np.log2(freqs[0]),np.log2(freqs[-1]),100,base=2)
single = 1/freqs_hat*1.5
double = 1/(freqs_hat**2)*5e-1
print(freqs_hat)

plt.subplot(1,2,2)
plt.loglog(freqs_hat,single,'--',color='k')
plt.loglog(freqs_hat,double, color='k')
plt.loglog(freqs,fs[0:-1],'-o')
plt.loglog(freqs,fn[0:-1],'-o')
plt.xlabel("Resolution")
plt.ylabel("Relative error in $\omega$")
plt.tick_params(which='both',direction='in')
plt.xticks([10,50,100])
plt.xlim(freqs[0],freqs[-1])


data_cont = np.load("ring_continuous_new.npz",allow_pickle=True)
radii = data_cont["radius"]
radii_fsc = []
fsc = []
for a,r in zip(data_cont["frequencies_s"],radii):
    if len(a) > 0:
        fsc.append(a[0])
        radii_fsc.append(r)
fsc = np.array(fsc)
fnc = np.array([a[0] for a in data_cont["frequencies_n"]])

plt.subplot(1,2,1)
plt.plot(radii,fsc,'-o')
plt.plot(radii,fnc,'-o')
plt.xlabel("Cylinder radius")
plt.ylabel("Mode frequency $\omega$")
plt.tick_params(which='both',direction='in')
plt.savefig("convergence_ring_new_course.png")
plt.show()

