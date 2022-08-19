import numpy as np
import meep as mp
from meep import adjoint as mpa
import skfmm
from scipy import interpolate
import matplotlib.pyplot as plt

plt.figure(figsize=(5.25,3.0),constrained_layout=True)

data = np.load("pc_convergence_new_longdigits.npz",allow_pickle=True)
data_cyl = np.load("pc_continuous_cyl.npz",allow_pickle=True)

print(data["frequencies_s"])
print(data["frequencies_n"])

fs = np.array([a[0] for a in data["frequencies_s"]])

#cal = 0.2924311007769104#fs[-1]
cal = 0.3040672393669545#fs[-1]


fs = np.abs(fs - cal) / cal
fn = np.array([a[0] for a in data["frequencies_n"]])
fn = np.abs(fn - cal) / cal
fc = np.array([a[0] for a in data_cyl["frequencies_s"]])
fc = np.abs(fc - cal) / cal


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
#plt.loglog(freqs,fc[0:-1],'-o',color="green")
plt.xlabel("Resolution")
plt.ylabel("Relative error in $\omega$")
plt.tick_params(which='both',direction='in')
plt.xticks([10,50,100])
plt.xlim(freqs[0],freqs[-1])


#data_cont = np.load("pc_continuous_new_fixed.npz",allow_pickle=True)
data_cont = np.load("pc_continuous_new_fine.npz",allow_pickle=True)
radii = data_cont["radius"]
fsc = np.array([a[0] for a in data_cont["frequencies_s"]])
fnc = np.array([a[0] for a in data_cont["frequencies_n"]])

plt.subplot(1,2,1)
plt.plot(radii,fsc,'-o')
plt.plot(radii,fnc,'-o')
plt.xlabel("Cylinder radius")
plt.ylabel("Mode frequency $\omega$")
plt.savefig("convergence_pc_new_fine.png")
plt.show()

