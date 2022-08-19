import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product
from enum import Enum
from matplotlib import pyplot as plt
from matplotlib import ticker as tkr
from meep.visualization import plot_boundaries

import sweep_beta_mag as sbm

sbm.opt.design_regions[0].design_parameters.do_averaging = True
sbm.opt.update_design([sbm.mapping(sbm.p,sbm.filter_radius)],beta=np.inf)

# ------------------------------------------------ #
#
# ------------------------------------------------ #
'''ax = plt.subplot(2,3,1)
p = sbm.p
x = np.linspace(-sbm.dx/2,sbm.dx/2,p.size)
plt.plot(x,p)
plt.title('(a)')
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.xaxis.set_minor_locator(tkr.AutoMinorLocator(2))
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.yaxis.set_major_locator(plt.MaxNLocator(6))
ax.yaxis.set_minor_locator(tkr.AutoMinorLocator(2))
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle=':')

ax=plt.subplot(2,3,2)
plt.plot(x,sbm.mapping(sbm.p,sbm.filter_radius))
plt.title('(b)')
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.xaxis.set_minor_locator(tkr.AutoMinorLocator(2))
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.yaxis.set_major_locator(plt.MaxNLocator(6))
ax.yaxis.set_minor_locator(tkr.AutoMinorLocator(2))
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle=':')

ax=plt.subplot(2,3,3)
eps_grid = sbm.sim.get_epsilon_grid(x,np.array([0]),np.array([0]),1/1.55)
plt.plot(x,eps_grid)
plt.title('(c)')
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.xaxis.set_minor_locator(tkr.AutoMinorLocator(2))
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.yaxis.set_major_locator(plt.MaxNLocator(6))
ax.yaxis.set_minor_locator(tkr.AutoMinorLocator(2))
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle=':')'''

# ------------------------------------------------ #
#
# ------------------------------------------------ #
plt.figure(figsize=(2.5,2.5))

sbm.opt.sim.sources[0].src = mp.ContinuousSource(wavelength=1.55)
sbm.opt.sim.run(until=100)
sbm.opt.plot2D(fields=mp.Ez,
plot_boundaries_flag=False,
output_plane=mp.Volume(center=(0,0,0),size=(4,2,0)),
    eps_parameters={'resolution':100},
    field_parameters={'alpha':0.8},
    monitor_parameters={'edgecolor':(0,0,1),'facecolor':(0,0,1,0.15),'alpha':None,'hatch':None},
    boundary_parameters={'edgecolor':'gray','facecolor':'gray','alpha':0.3,'hatch':None})
plt.axis("off")
plt.savefig("steady_state.svg")

# ------------------------------------------------ #
#
# ------------------------------------------------ #

linewidth = 2.0
plt.figure(figsize=(5.5,3.25))

# ------------------------------------------------ #
#
# ------------------------------------------------ #
filename = "u_sweep.npz"
data_old = np.load(filename)
filename = "u_sweep_tight.npz"
data = np.load(filename)
u_bank=data['u_bank']
u_bank_new=data_old['u_bank']
new_method=data_old['new_method']
old_method=data['old_method']

ax = plt.subplot(1,2,1)
plt.plot(u_bank_new,new_method,linewidth=linewidth,label="hybrid level-set")
plt.plot(u_bank,old_method,linewidth=linewidth,label="traditional density method")
plt.xlabel(r'$\rho_k$')
plt.title('(a)')
plt.ylabel(r'$f(\rho)$ (a.u.)')

ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.xaxis.set_minor_locator(tkr.AutoMinorLocator(2))
ax.xaxis.set_minor_formatter(plt.NullFormatter())

ax.yaxis.set_major_locator(plt.MaxNLocator(6))
ax.yaxis.set_minor_locator(tkr.AutoMinorLocator(2))
ax.yaxis.set_minor_formatter(plt.NullFormatter())

ax.tick_params(which='both',right=True,top=True,direction='in')
plt.xlim(0,1)

#ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle=':')
# ------------------------------------------------ #
#
# ------------------------------------------------ #

'''filename = "beta_sweep.npz"

data = np.load(filename)
beta_list = data['beta_list']
l2_new_list = data['smoothing_list']
l2_old_list = data['no_smoothing_list']

plt.subplot(2,2,3)
plt.loglog(beta_list,l2_old_list,'-o')
#plt.loglog(beta_list,l2_new_list,'-o')
plt.grid(True)
plt.ylabel(r'Relative gradient error')
plt.xlabel(r'$\beta$')'''

# ------------------------------------------------ #
#
# ------------------------------------------------ #
import matplotlib.ticker
filename = "beta_mag_sweep.npz"

data = np.load(filename)
beta_list = data['beta_list']
l2_new_list = data['l2_new_list']
l2_old_list = data['l2_old_list']

ax = plt.subplot(1,2,2)
plt.loglog(beta_list,l2_new_list,linewidth=linewidth)
plt.loglog(beta_list,l2_old_list,linewidth=linewidth)

plt.ylabel(r'$|df/d\rho|_2^2$ (a.u.)')
plt.xlabel(r'$\beta$')
plt.title("(b)")
#plt.xticks([1,3,10,30,100,300,1000])
#ax.minorticks_on()
#ax.xaxis.set_major_locator(plt.LogLocator(numticks=10))
#ax.yaxis.set_major_locator(plt.LogLocator(numticks=8))
# Change minor ticks to show every 5. (20/4 = 5)
#ax.xaxis.set_minor_locator(plt.LogLocator(numticks=20))
#ax.xaxis.set_minor_formatter(plt.NullFormatter())
#ax.yaxis.set_minor_locator(plt.LogLocator(numticks=20))
#ax.yaxis.set_minor_formatter(plt.NullFormatter())

y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 6)
ax.yaxis.set_major_locator(y_major)
y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 15)
ax.yaxis.set_minor_locator(y_minor)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

x_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 5)
ax.xaxis.set_major_locator(x_major)
x_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 5)
ax.xaxis.set_minor_locator(x_minor)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.tick_params(which='both',right=True,top=True,direction='in')
plt.xlim(1,1e3)
#ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle=':')
# ------------------------------------------------ #
#
# ------------------------------------------------ #

plt.tight_layout(w_pad=0, h_pad=0)

plt.savefig("hybrid_method.svg",dpi=300)
plt.show()
