#!/usr/bin/python3
# Nicholas Rathmann and Daniel Shapero, 2024

r"""
Test firedrake interface for specfab given a time-constant, non-uniform shear flow in vertical cross-section (xz) domain
"""

import copy, sys, time, code # code.interact(local=locals())
import numpy as np

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import matplotlib.tri as tri
from matplotlib import rc
rc('font',**{'family':'serif', 'sans-serif':['Times'], 'size':15})

import firedrake as fd
from specfabpy.firedrake.ice import IceFabric
from specfabpy import plotting as sfplt

"""
CPO problem setup
"""

Nt = 30 # number of time steps to take
L  = 6 # spectral truncation

modelplane = 'xz'
CPO_kwargs = dict(nu_multiplier=1, nu_realspace=1e-3, modelplane=modelplane)

iota, Gamma0 = +1, None # lattice rotation only
#iota, Gamma0 = 0, 1e-6 # DDRX only

"""
Setup firedrake CPO class
"""

nx = ny = 16
mesh = fd.UnitSquareMesh(nx, ny) #, diagonal='crossed')
x = fd.SpatialCoordinate(mesh)
V = fd.VectorFunctionSpace(mesh, "CG", 1)
Q = fd.FunctionSpace(mesh, "CG", 1) # for projecting scalar fabric measures
T = fd.TensorFunctionSpace(mesh, "CG", 1)

u0, H = 1, 1
expr = fd.as_vector(( u0*(x[1]/H)**2, 0 )) # non-uniform horizontal shear
u = fd.Function(V).interpolate(expr)

# Isotropic ice incoming from left-hand boundary, remaining boundaries are free (no fabric fluxes)
boundaries = (1,2,3,4) 
fabric = IceFabric(mesh, boundaries, L, **CPO_kwargs)
fabric.initialize() # set isotroic CPO field
fabric.set_isotropic_BCs((1,))

tau = fd.project(fd.grad(u), T) # assume coaxial driving stress

"""
Time evolution
"""

# Determine time step using CFL criterion
h_min = 1/nx
v_max = abs(u.vector()[:]).max()
C_CFL = 0.5 # Courant number
dt = C_CFL*h_min/v_max
dt *= 4 # more aggresive time-stepping than CFL

# Get ready
t = 0.0
nn = 0

while nn < Nt:

    nn += 1
    t  += dt
    
    print("*** Step %i :: dt=%.2e, t=%.2e" % (nn, dt, t))
    print("-> Solving fabric evolution")
    fabric.evolve(u, tau, dt, iota=iota, Gamma0=Gamma0) # only lattice rotation supported right now, so takes just velocity field as argument
    
    ### Plot results
        
    if nn==1 or nn%1==0:
    
        fname = 'simpleshear-%s-%02d.png'%('LROT' if Gamma0 is None else 'DDRX', nn)
        print('[->] Plotting model state: %s'%(fname))

        ### Setup figure
        
        figscale = 1
        fig = plt.figure(figsize=(14*figscale, 8*figscale))
        gs = gridspec.GridSpec(1, 3, wspace=0.25, hspace=0.3, left=0.08, right=0.97, top=0.98, bottom=0.37)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[0,2])
        axes = (ax1,ax2,ax3)
        
        kwargs_cb = dict(pad=0.12, aspect=22, fraction=0.09, orientation='horizontal')

        ax = ax1
        h = fd.pyplot.quiver(u, axes=ax, cmap='Reds', width=0.0075)
        cbar = plt.colorbar(h, ax=ax, **kwargs_cb)
        cbar.ax.set_xlabel(r'$\bf u$')
        
        ax = ax2
        J = fd.project(fd.dot(fabric.w,fabric.w)/np.linalg.norm(fabric.nlm_iso)**2, Q)
        h = fd.pyplot.tricontourf(J, axes=ax, levels=np.arange(1, 2 +1e-3, 0.1), extend='both', cmap='YlGnBu')
        cbar = plt.colorbar(h, ax=ax, **kwargs_cb)
        cbar.ax.set_xlabel(r'$J$ (CPO strength)')

        ax = ax3
        E0, E1, dE = (0.5, 1.5, 0.1)
        lvls = np.arange(E0, E1+1e-3, dE)
        divnorm = colors.TwoSlopeNorm(vmin=np.amin(lvls), vcenter=1, vmax=np.amax(lvls))
        kwargs_E = dict(levels=lvls, norm=divnorm, extend='both', cmap='PuOr_r')
        E_CAFFE = fabric.E_CAFFE(u)
        h = fd.pyplot.tricontourf(E_CAFFE, axes=ax, **kwargs_E)
        cbar = plt.colorbar(h, ax=ax, **kwargs_cb)
        cbar.ax.set_xlabel(r'$E$ (CAFFE)')
        
        # plot grids 
        for ax in axes:
            ax.set_xlabel('$x$')
            ax.set_ylabel('$z$')
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            fd.pyplot.triplot(mesh, axes=ax, interior_kw=dict(lw=0.1), boundary_kw=dict(lw=5, clip_on=False))

        ### ODF insets
        
        def plot_ODF(x,y, CPOx,CPOy, mrk, W=0.18):

            geo, prj = sfplt.getprojection(rotation=-90-20, inclination=50)   
            axin = plt.axes([x,y , W, W], projection=prj) 
            axin.set_global()
            for ax in (ax1,ax2):
                points, = ax.plot(CPOx, CPOy, mrk, markersize=12, markeredgewidth=1.1, markeredgecolor='k', markerfacecolor='w')
            sfplt.plotODF(fabric.get_nlm(CPOx, CPOy), fabric.lm, axin, cmap='Greys', cblabel='ODF', lvlset=(np.linspace(0.1, 0.3, 5), lambda x,p:'%.1f'%x), showcb=True)
            sfplt.plotcoordaxes(axin, geo, color='k')
            kwargs = dict(marker='.', ms=7, markeredgewidth=1.0, transform=geo, zorder=10, markerfacecolor=sfplt.c_dred, markeredgecolor=sfplt.c_dred)
            m1 = fabric.eigenframe(CPOx, CPOy)[0][:,0] # pick out m1
            sfplt.plotS2point(axin, +m1, **kwargs)
            sfplt.plotS2point(axin, -m1, **kwargs)
            axin.set_title(r'@ "%s"'%(mrk))

        # Plot for selected locations
        plot_ODF(0.2,0.08, 0.1,0.1, '^')
        plot_ODF(0.4,0.08, 0.5,0.4, 'X')
        plot_ODF(0.6,0.08, 0.9,0.6, 's')
            
        ### Save plot
        
        plt.savefig(fname, dpi=100)
        plt.close()
        print('[OK] Done')
        