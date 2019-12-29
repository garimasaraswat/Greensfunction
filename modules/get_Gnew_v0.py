import numpy as np #Numerical package for arrays, algebra...
from mpl_toolkits.mplot3d import Axes3D #generate 3D plots
from scipy.special import j0,y0 #0th order Bessel function of first and second kind
from modules import constant.py
import time

def return_G0(npix=75,pixwid=5e-10,x_p,y_p,k=0.1):
    G0=np.zeros((npix,npix),dtype=np.complex64)
    x,y=np.indices(G0.shape)
    r=eps+np.sqrt((x-x_p)**2+(y-y_p)**2)*pixwid
    G0=-1*norm*(y0(k*r)+1j*j0(k*r))
    return G0


def return_gaussianV(npix=75,pixwid=5e-10,x_p,y_p,sigma=2.,amp=100.):
    gaussianV=np.zeros((npix,npix),dtype=np.float32)
    x,y=np.indices(gaussianV.shape)
    gaussianV=amp*(1./(2.*np.pi*sigma**2.))*np.exp(-((x-x_p)**2.+(y-y_p)**2.)/(2.*sigma**2.))
    return gaussianV

def return_random_V(npix=75,pixwid=5e-10,n_sites=5,sig_mean=10,sig_width=4.,amp_mean=100.,amp_width=10.,seed=0):
    V=np.zeros((npix,npix),dtype=np.float32)
    x,y=np.indices(V.shape)
    np.random.seed(seed)
    x_idx=np.random.choice(np.arange(npix),n_sites)
    y_idx=np.random.choice(np.arange(npix),n_sites)
    imp_amp=np.random.normal(amp_mean,amp_width,n_sites)
    imp_wid=np.random.normal(sig_mean,sig_width,n_sites)
    for i in range(n_sites):
        randomV=randomV + imp_amp[i]*(1./(2.*np.pi*imp_wid[i]**2.))*np.exp(-((x-x_idx[i])**2.+(y-y_idx[i])**2.)/(2.*imp_wid[i]**2.))
    return randomV

def get_Gnew(npix=75,pixwid=5e-10)
    G0all=np.zeros((npix**2,npix**2),dtype=np.complex)
    s=0
    for itip in range(npix):
        for jtip in range(npix):
            G0all[:,s]=(return_G0(x_p=itip,y_p=jtip,k=.8)).ravel()
            s=s+1
    I0=identity(npix**2,dtype=np.complex)
   #G0V=np.matmul(G0all,Vall) #use when V is not diagonal
    G0V=np.einsum("ij,jj->ij",G0all,Vall) #use when V is diagonal
    Gnew=np.linalg.solve(I0-G0V,G0all)
    return Gnew


