import time
#Numerical package for arrays, algebra...
import numpy as np
#0th and 1st order Bessel function of first and second kind
from scipy.special import j0,y0,j1, y1
from scipy.sparse import csr_matrix
#list of fundamental constants used
import constants as cnst
#import multiprocessing as mp
import itertools
from scipy.ndimage import gaussian_filter as gaussfil


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
        self.hbarev=cnst.hbar/cnst.e_charge

    def get_sidx_given_xy(self,x_tip,y_tip):
        return x_tip*self.npix+y_tip

    def get_xy_given_sidx(self,sidx):
        y_tip=np.mod(sidx,self.npix)
        x_tip=(sidx-y_tip)/self.npix
        return x_tip,y_tip
    #######################################################
    #######################################################
    def setup_potential(self,V0=1.5,rad=2e-9,po_type="circ",edgewid=0.05,smwid=0.0,img=[],img_thr=0.5):
        '''po_type="circ","circisland",hex","hexisland","image"
           edgewid= percentage of radius for width of circumference
        '''
        self.V=np.zeros((self.npix,self.npix),dtype=np.complex)
        V0=V0*(self.pixwid**2.)
        #############
        if po_type=="circ":
            for i in range(self.npix):
                for j in range(self.npix):
                    radius=np.sqrt((i-self.npix/2)**2+ (j-self.npix/2)**2)*self.pixwid
                    if radius >= rad and radius <=rad*(1+edgewid):
                        self.V[i,j]=V0
        #############
        if po_type=="circisland":
            for i in range(self.npix):
                for j in range(self.npix):
                    radius=np.sqrt((i-self.npix/2)**2+ (j-self.npix/2)**2)*self.pixwid
                    if radius <=rad:
                        self.V[i,j]=V0
        
        ##############
        if po_type=="hex":
            x0=np.linspace(-self.npix/2,self.npix/2,self.npix)
            y0=np.linspace(-self.npix/2,self.npix/2,self.npix)
            cidx=[t for t in itertools.product(np.arange(self.npix),np.arange(self.npix))]
            coord=[t for t in itertools.product(x0,y0)]
            
            
            def hexagon(pos,rad):
                x, y = map(abs, pos) ; x=x*self.pixwid ; y=y*self.pixwid
                return y <= 3**0.5 * min(rad - x, rad/2)
            
            tempV=np.zeros(self.npix**2,np.complex)
            for i,xy in enumerate(coord):
                if hexagon(xy,rad):
                    tempV[i]=V0
                if hexagon(xy,rad*(1.+edgewid)):
                    tempV[i]=V0-tempV[i]
                        
            self.V=tempV.reshape(self.npix,self.npix)
        ##############
        if po_type=="hexisland":
            x0=np.linspace(-self.npix/2,self.npix/2,self.npix)
            y0=np.linspace(-self.npix/2,self.npix/2,self.npix)
            cidx=[t for t in itertools.product(np.arange(self.npix),np.arange(self.npix))]
            coord=[t for t in itertools.product(x0,y0)]

            
            def hexagon(pos):
                x, y = map(abs, pos) ; x=x*self.pixwid ; y=y*self.pixwid
                return y <= 3**0.5 * min(rad - x, rad/2)
            
            tempV=np.zeros(self.npix**2,np.complex)
            for i,xy in enumerate(coord):
                if hexagon(xy):
                    tempV[i]=V0

            self.V=tempV.reshape(self.npix,self.npix)
        ##############
        if po_type=="image":
            def rebin(arr):
                shape = (self.npix, arr.shape[0] // self.npix,self.npix, arr.shape[1] // self.npix)
                return arr.reshape(shape).mean(-1).mean(1)
            self.V=rebin(img)
            self.V=self.V/np.max(self.V)
            self.V[self.V<img_thr]=0.
            self.V=self.V*V0
        ##############
        
        
        if smwid!=0.0:
            self.V=gaussfil(np.real(self.V),smwid) + 1j*gaussfil(np.imag(self.V),smwid)

        self.Vall=np.kron(np.diag(self.V.ravel()),cnst.sigma0)

        
    def return_G0(self,E,x_tip,y_tip,rmask=2,do_vis=False,eps=1e-15):
        f0=np.zeros((self.npix,self.npix),dtype=np.complex)
        f1=np.zeros((self.npix,self.npix),dtype=np.complex)
        x,y=np.indices(f0.shape)
        r=eps+self.pixwid*np.sqrt((x-x_tip)**2.+(y-y_tip)**2.) ; mask=(r>rmask*self.pixwid)

        xi=abs(E)*r/(self.hbarev*self.vf)
        norm=abs(E)/(4*(self.hbarev**2.)*(self.vf**2.))

        f0=np.multiply(norm*(np.sign(E)*y0(xi)-1j*j0(xi)*np.heaviside(1-(abs(E)/self.band_cutoff), 1)),mask)
        f1=np.multiply(norm*(1j*y1(xi)+np.sign(E)*j1(xi)*np.heaviside(1-(abs(E)/self.band_cutoff), 1)),mask)


        limG0=-norm*((np.sign(E)/np.pi)*np.log(abs((self.band_cutoff/E)**2.-1.)) + 1.j*np.heaviside(1-(abs(E)/self.band_cutoff), 1))

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

    def return_G0_new(self,E,sidx,rmask=2,do_vis=False,eps=1e-15):
        x_tip,y_tip=self.get_xy_given_sidx(sidx)
        f0=np.zeros((self.npix,self.npix),dtype=np.complex)
        f1=np.zeros((self.npix,self.npix),dtype=np.complex)
        x,y=np.indices(f0.shape)
        r=eps+self.pixwid*np.sqrt((x-x_tip)**2.+(y-y_tip)**2.) ; mask=(r>rmask*self.pixwid)
        
        xi=abs(E)*r/(self.hbarev*self.vf)
        norm=abs(E)/(4*(self.hbarev**2.)*(self.vf**2.))
        
        f0=np.multiply(norm*(np.sign(E)*y0(xi)-1j*j0(xi)*np.heaviside(1-(abs(E)/self.band_cutoff), 1)),mask)
        f1=np.multiply(norm*(1j*y1(xi)+np.sign(E)*j1(xi)*np.heaviside(1-(abs(E)/self.band_cutoff), 1)),mask)
        
        
        limG0=-norm*((np.sign(E)/np.pi)*np.log(abs((self.band_cutoff/E)**2.-1.)) + 1.j*np.heaviside(1-(abs(E)/self.band_cutoff), 1))
        
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

    def get_G0all(self,E):
        G0all=np.zeros((2*self.npix**2,2*self.npix**2),dtype=np.complex64)
        s=0
        for itip in range(self.npix):
            for jtip in range(self.npix):
                G0all[:,s:s+2]=(self.return_G0(E=E,x_tip=itip,y_tip=jtip).T)
                s=s+2
        return G0all



    def get_Gnew(self,E):
        G0all=self.get_G0all(E=E)
        I0=np.identity(2*self.npix**2,dtype=np.complex)
        G0V=np.einsum("ij,jj->ij",G0all,self.Vall)
        Gnew=np.linalg.solve(I0-G0V,G0all)
        return Gnew


    def get_Gnew_fromT(self,E):
        '''
            Requires testing
        '''
        G0all=self.get_G0all(E=E)
        I0=np.identity(2*self.npix**2,dtype=np.complex)
        G0V=np.einsum("ij,jj->ij",G0all,self.Vall)
        
        nz_idx=np.nonzero(self.Vall)[0]
        temp=np.linalg.solve((I0-G0V).T,self.Vall[:,nz_idx])
        
        Tt=np.zeros_like(self.Vall)
        for i,idx in enumerate(nz_idx):
            Tt[:,idx]=temp[:,i]

        Tt_sparse=csr_matrix(Tt)
        TG0=Tt_sparse.dot(G0all) ; nz_idx=np.nonzero(TG0[:,0])[0]
        G0TG0=np.matmul(G0all[:,nz_idx],TG0[nz_idx,:])
        Gnew= G0all + G0TG0
        return Gnew
		
    def get_ldos_Gnew(self,E,method="T",loc=[]):
        '''method T is for T-matrix, else original'''
        if method=="T":
            G=self.get_Gnew_fromT(E=E)
        else:
            G=self.get_Gnew(E=E)
        LDOS=(-1./np.pi)*np.imag((np.diagonal(G)[1::2] + np.diagonal(G)[::2]).reshape(self.npix,self.npix))
        if loc:
            x_tip=loc[0] ; y_tip=loc[1]
            return LDOS[x_tip,y_tip]
        else:
            return LDOS



    def get_ldos_G0(self,E,loc=[]):
        G=self.get_G0all(E=E)
        LDOS=(-1./np.pi)*np.imag((np.diagonal(G)[1::2] + np.diagonal(G)[::2]).reshape(self.npix,self.npix))
        if loc:
            x_tip=loc[0] ; y_tip=loc[1]
            return LDOS[x_tip,y_tip]
        else:
            return LDOS

# To do
#    Parallel verion of getting G0all.
#    def get_G0all(self,E,numprocs=2):
#        site_adr=np.arange(self.npix**2)
#
#        def run_G0(sidx):
#            G0=self.return_G0_new(E=E,sidx=sidx)
#            return G0.T
#
#        def run_allG0_in_parallel(numprocs):
#            pool=mp.Pool(processes=numprocs)
#            allG0=pool.map(run_G0,site_adr)
#            pool.close()
#            pool.join()
#            return allG0
#
#        def __getstate__(self):
#            self_dict = self.__dict__.copy()
#            del self_dict['pool']
#            return self_dict
#
#        allG0=run_allG0_in_parallel(numprocs=numprocs)
#        G0all=np.concatenate((allG0),axis=1)
#
#        return G0all

