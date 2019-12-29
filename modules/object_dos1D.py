import numpy as np #Numerical package for arrays, algebra...
from scipy.special import j0,y0 #0th order Bessel function of first and second kind
from modules import constants as cnst #list of fundamental constants used
import time

class dos_calculator(object):
    '''mfac =m*/m_e #pixwid=size of pixel in meters #npix=number of pixels #E=electron energyin eV #E0=surface state binding energy in eV
    '''
    def __init__(self,xnpix,E0=-0.445,E=0.1,mfac=0.38,pixwid=.5,ynpix=1):
        self.pixwid=pixwid
        self.xnpix=xnpix
        if ynpix==[]:
            self.ynpix=xnpix
        else:
            self.ynpix=ynpix
        self.E=E
        self.E0=E0
        self.mfac=mfac

    def return_lambda(self):
        self.dbglambda=2e9*np.pi*cnst.hbar*np.sqrt(1/(2*self.mfac*cnst.m_e*abs(self.E-self.E0)*cnst.e_charge))

    def return_G0(self,x_tip,y_tip):
        self.G0=np.zeros((self.xnpix,self.ynpix),dtype=np.complex)
        x,y=np.indices(self.G0.shape)
        r=(self.pixwid*1e-3)+np.sqrt((x-x_tip)**2+(y-y_tip)**2)*self.pixwid #small value added so that r is nonzero
        k=(2.*np.pi)/self.dbglambda
        self.G0=-1.*((cnst.m_e*self.mfac)/(2.*cnst.hbar**2.))*(y0(k*r)+1j*j0(k*r)) # for free patricle in 2D

    def return_G01D(self,x_tip):
        self.G0=np.zeros((self.xnpix),dtype=np.complex)
        x=np.indices(self.G0.shape)
        r=np.abs(x-x_tip)*self.pixwid
        k=(2.*np.pi)/self.dbglambda
        self.G0=-1j*((cnst.m_e*self.mfac)/(2.*cnst.hbar**2.))*(np.exp(1j*k*r)/k) # for free patricle in 1D
    
    
    
    def return_G0all1D(self):
        self.G0all=np.zeros((self.xnpix,self.xnpix),dtype=np.complex)
        s=0
        for itip in range(self.xnpix):
                self.return_G01D(x_tip=itip)
                self.G0all[:,s]=self.G0.ravel()
                s=s+1
        self.G0all=self.G0all*cnst.e_charge

    def get_Gnew(self,V):
        I0=np.identity(self.xnpix,dtype=np.complex)
        Vall=np.diag(V.ravel())
        #G0V=np.matmul(self.G0all,Vall) #use when V is not diagonal
        G0V=np.einsum("ij,jj->ij",self.G0all,Vall) #use when V is diagonal
        self.Gnew=np.linalg.solve(I0-G0V,self.G0all)
        self.dos=(-1/np.pi)*np.imag(np.diag(self.Gnew)).reshape(self.xnpix,self.ynpix)
#return Gnew
#
#
#
#
