import numpy as np #Numerical package for arrays, algebra...
from scipy.special import j0,y0,j1, y1 #0th and 1st order Bessel function of first and second kind
import constants as cnst #list of fundamental constants used
import time

class init_material(object):
    '''vf= Fermi velocity in m/s
       band_cutoff=Band width in eV
       band_offset=Dirac point in eV
       npix=number of pixels
       pixwid=distance between center of adjacent pixels in m
    '''
    
    def __init__(self,vf=4.6e5,band_cutoff=0.3,band_offset=0.,npix=40,pixwid=1e-9):
        self.pixwid=pixwid
        self.npix=npix
        self.band_cutoff=band_cutoff
        self.band_offset=band_offset
        self.vf=vf
        self.omega=0.
        self.hbarev=cnst.hbar/cnst.e_charge

    def setup_potential(self,vamp=1.,po_type="circle"):
        self.V=np.zeros((self.npix,self.npix),dtype=np.complex)
        for i in range(self.npix):
            for j in range(self.npix):
                radius=np.sqrt((i-self.npix/2)**2+ (j-self.npix/2)**2)*self.pixwid
                if radius >= 10.*self.pixwid and radius <=10*self.pixwid:
                    self.V[i,j]=vamp
        self.Vall=np.kron(np.diag(self.V.ravel()),cnst.sigma0)

    def return_G0(self,x_tip,y_tip,rmask=2,do_vis=False,eps=1e-15):
        f0=np.zeros((self.npix,self.npix),dtype=np.complex)
        f1=np.zeros((self.npix,self.npix),dtype=np.complex)
        x,y=np.indices(f0.shape)
        r=eps+self.pixwid*np.sqrt((x-x_tip)**2.+(y-y_tip)**2.) ; mask=(r>rmask*self.pixwid)

        xi=abs(self.omega)*r/(self.hbarev*self.vf)
        norm=abs(self.omega)/(4*(self.hbarev**2.)*(self.vf**2.))
    
        f0=np.multiply(norm*(np.sign(self.omega)*y0(xi)-1j*j0(xi)*np.heaviside(1-(abs(self.omega)/self.band_cutoff), 1)),mask)
        f1=np.multiply(norm*(1j*y1(xi)+np.sign(self.omega)*j1(xi)*np.heaviside(1-(abs(self.omega)/self.band_cutoff), 1)),mask)
    
    
        limG0=-norm*((np.sign(self.omega)/np.pi)*np.log(abs((self.band_cutoff/self.omega)**2.-1.)) + 1.j*np.heaviside(1-(abs(self.omega)/self.band_cutoff), 1))
    
        if do_vis:
            G0=np.kron(f0,cnst.sigma0)
            G0=G0+np.kron(np.multiply((x-x_tip)*self.pixwid,f1),cnst.sigmax)
            G0=G0+np.kron(np.multiply((y-y_tip)*self.pixwid,f1),cnst.sigmay)
        
            G0=G0+np.kron((limG0*(1-mask)),cnst.sigma0)
        else:
            G0=np.kron(f0.ravel(),cnst.sigma0)
            G0=G0+np.kron(np.multiply((x-x_tip)*self.pixwid,f1).ravel(),cnst.sigmax)
            G0=G0+np.kron(np.multiply((y-y_tip)*self.pixwid,f1).ravel(),cnst.sigmay)
            G0=G0+np.kron((limG0*(1-mask)).ravel(),cnst.sigma0)
        return G0

    def get_G0all(self,E=0.):
        self.omega=E
        self.G0all=np.zeros((2*self.npix**2,2*self.npix**2),dtype=np.complex64)
        s=0
        for itip in range(self.npix):
            for jtip in range(self.npix):
                self.G0all[:,s:s+2]=(self.return_G0(x_tip=itip,y_tip=jtip).T)
                s=s+2

    def get_Gnew(self):
        I0=np.identity(2*self.npix**2,dtype=np.complex)
        G0V=np.einsum("ij,jj->ij",self.G0all,self.Vall)
        self.Gnew=np.linalg.solve(I0-G0V,self.G0all)

    def get_ldos(self,loc=[]):
        LDOS=(-1./np.pi)*np.imag((np.diagonal(self.Gnew)[1::2] + np.diagonal(self.Gnew)[::2]).reshape(self.npix,self.npix))
        if loc:
            x_tip=loc[0] ; y_tip=loc[1]
            return LDOS[x_tip,y_tip]
        else:
            return LDOS



