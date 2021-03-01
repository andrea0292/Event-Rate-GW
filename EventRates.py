import numpy as np
from numpy import sqrt
from numpy import sin
from numpy import pi
from numpy import cos
from numpy import exp
from numpy import tanh
from numpy import zeros
from numpy import arccos
from numpy import log10
from matplotlib import pyplot as plt
import scipy.optimize as opt
import emcee
from pylab import plot
from scipy import integrate
import corner
import random
from collections import Counter
import time
import scipy
from scipy.stats import poisson # use as poisson.pmf(number of events , mean value)
from scipy import optimize
import multiprocessing as mp
from multiprocessing import Pool

import astropy.units as u
import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM, z_at_value
from tqdm import *
from astropy.cosmology import Planck13 as cosmo
from astropy import constants as const
import h5py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cmasher as cmr
import matplotlib.colors as clrs
import matplotlib.cm as cmap
import cmath
from scipy.optimize import root_scalar

import LisaNoise
from LisaNoise import * 

# Now import the mass data

eobdata = np.loadtxt('EOBmasses.dat')
NRdata = np.loadtxt('NRmasses.dat')
Phendata = np.loadtxt('PHENmasses.dat')

data = h5py.File('/Users/andreacaputo/Desktop/Phd/BinaryLISALIGO/Event_SamplesWaveTransient/all_events/o1o2o3_mass_c_iid_mag_two_comp_iid_tilt_powerlaw_redshift_mass_data.h5', 'r')
mass_ppd = data["ppd"]
mass_lines = data["lines"]

# Create a flat copy of the array and calculate also the rate
flat = mass_ppd[:].flatten() /  np.sum(mass_ppd[:].flatten())
mass_1 = np.linspace(3, 100, 1000)
mass_ratio = np.linspace(0.1, 1, 500)
ratePL = np.trapz(np.trapz(mass_ppd, mass_ratio, axis=0), mass_1, axis = 0)


class GWfunctions:

    def __init__(self):


        # Set constants
        # we work in geometrized units
        self.Msol = 1.47680000000000000000000000000000000 # in km
        self.c = 1 # we have actually put them to one everywhere in the code already
        self.G = 1
        self.year = 60*60*24*(365.250000000000000000000000000); # s
        self.cc = 2.9980000000000000000000000000000*10**5 # velocity of light in Km/s
        self.pc = 3.0856770000000000000000000000*10**13 # km
        self.H0 = 67.9 # km /s / Mpc # changed to PLANCK 2015
        self.Omegam = 0.306
        self.OmegaL = 0.694
        self.Omegar = 0.

    
    # some useful functions 
    
    def Ez(self, z):

        return sqrt(self.Omegar*(1+z)**4+self.Omegam*(1+z)**3+self.OmegaL)

    def invE(self, z):
        return 1. / self.Ez(z)

    def distL(self, z):

        return (1+z) * integrate.quad(self.invE,0,z)[0] *self.cc / self.H0 *10**6*self.pc # km

    def primopezzo(self, zp):
        return 1/self.Ez(zp)*(integrate.quad(self.invE,0,zp)[0])**2

    def dVdz(self, z):
        return 4*np.pi*(self.cc/self.H0)**3*self.primopezzo(z) # Mpc**3

    def dtdf(self, f,m1,m2):  # Hz, km, km, gravitational wave frequency
        return 5/(96*np.pi**(8/3))*(1./self.cc*(m1 + m2)*(m1*m2/(m1 + m2)**2)**(3/5))**(-5/3)*f**(-11/3)  # s^2, assuming f in Hz and m in km

    def solof(self, f):

        return f**(-11/3)

    def nu(self, m1,m2):

        return m1*m2/((m1+m2)**2)

    def Mc(self, m1,m2):

        return (m1 + m2) * (m1*m2/((m1+m2)**2))**(3/5)


    def tau(self, ma,g): 
        # axion lifetime, pass the mass in eV and the coupling in GeV**-1, the lifetime will be
        # then in 1/eV
        return (ma**3 * (g * 1e-9)**2/64/np.pi)**-1

    # frequency functions 

    def fMIN2(self, fmax0,m1,m2,Tobs): # gravitational wave frequency

        return 1/(1/fmax0**(8/3)+256/5*(self.Mc(m1,m2)/self.cc)**(5/3)*np.pi**(8/3)*(Tobs))**(3/8)

    def fmax(self, m1,m2,fmin,Tobs): # gravitational wave frequency

        return fmin*((5*self.cc)/(5*self.cc-256*fmin**(8/3)*np.pi**(8/3)*Tobs*self.Mc(m1,m2)*(self.Mc(m1,m2)/self.cc)**(2/3)))**(3/8)

    def getfmax(self, m1,m2,fmin,Tobs): # gravitational wave frequency
        if self.fMIN2(1,m1,m2,Tobs)>fmin:

            return self.fmax(m1,m2,fmin,Tobs)

        else: 
            return 1

    def Tmerger(self, m1,m2,fmin): # merger time

        return 5. * (m1 + m2)**(1./3.) * self.cc**(5./3.)/ (256. * fmin**(8./3.) * m1 * m2 * np.pi**(8./3.))


    def findFmin(self, timemerger,m1,m2):

        def condition(fmin):
            return timemerger-self.Tmerger(m1,m2,fmin)

        return root_scalar(condition,bracket=(10**-4,1)).root

    # waveform

    def ampl(self, m1,m2,d): 

        return (self.Mc(m1,m2)*self.G)**(5/6)*np.sqrt(5/24)/(np.pi**(2/3)*d*self.c**(3/2))
    
    def habs(self, m1,m2,d,f): # gravitational wave frequency
        return self.ampl(m1,m2,d)*f**(-7/6)

    # Here we define the SNR; 

    def SNR(self, iota,psi,theta,phi,fmin,fmax,m1,m2,d, len): 
        
        return sqrt(4*integrate.quad(lambda x: (factorskySNR(iota,psi,theta,phi)*self.habs(m1,m2,d, x/cc))**2/(Pn(x, len)*(self.cc**2)), fmin, fmax)[0])

    def SNRAverage(self, fmin,fmax,m1,m2,d, len): 
        
        return sqrt(4*integrate.quad(lambda x: (skyav*self.habs(m1,m2,d, x/self.cc))**2/(Pn(x, len)*(self.cc**2)), fmin, fmax)[0])

class NumberIntrinsic(GWfunctions):

    def __init__(self, **kwargs):
        GWfunctions.__init__(self, **kwargs)
        self.year = 60*60*24*(365.250000000000000000000000000); # s
    
    def gendistrIntrinsic(self, N,iteration, massoption,SNRth, Tobs, duty, ttmin, ttmax, leng):
    
        N=int(N)
        # Define functions to store intermediate products; I haven't decided on the most convenient format for the final catalogs yet...

        massmin=50*Msol
        massmax=100*Msol

        # Largest horizon redshift; don't waste computing time above this
        zmax = 0.512
        zmin = 0.1
        # Largest comoving distance for sampling
        # largest merger time (yrs)
        tmax = ttmax
        tobs = Tobs
        # ground-based duty cycle
        dutycycle= duty

        data=[]
        for i in range(N):

            z = np.random.uniform(zmin,zmax)
            dl = self.distL(z)

            # Mass spectrum
        

            if massoption == 'EOB':
                index = np.random.choice(len(eobdata))
                m1 = Msol*eobdata[index][0]
                m2 = Msol*eobdata[index][1]
                
        
            if massoption == 'PHEN':
                index = np.random.choice(len(Phendata))
                m1 = Msol* Phendata[index][0]
                m2 = Msol*Phendata[index][1]
                
            
            if massoption == 'NR':
                index = np.random.choice(len(NRdata))
                m1 = Msol*NRdata[index][0]
                m2 = Msol*NRdata[index][1]
           
            
            if massoption == 'breakPL' or massoption == 'breakPLcut':
            
                sample_index = np.random.choice(a=flat.size, p=flat)
                adjusted_index = np.unravel_index(sample_index, mass_ppd[:].shape)
                m1 = Msol * mass_1[adjusted_index[1]]
                m2 = Msol * mass_1[adjusted_index[1]] * mass_ratio[adjusted_index[0]]
            
            if massoption == 'PL_error':

                index1 = np.random.choice(len(mass_lines['mass_ratio'][:]))
                mass_ppd1 = np.outer(mass_lines['mass_ratio'][:][5166], mass_lines['mass_1'][:][5166] )
                flat1 = mass_ppd1.flatten() /  np.sum(mass_ppd1.flatten())
            
                sample_index = np.random.choice(a=flat1.size, p=flat1)
                adjusted_index = np.unravel_index(sample_index, mass_ppd1.shape)
                m1 = Msol * mass_1[adjusted_index[1]]
                m2 = Msol * mass_1[adjusted_index[1]] * mass_ratio[adjusted_index[0]]

        
            # Sky-location, inclination, polarization, initial phase
            cosiota = random.uniform(-1.,1.)
            psi = random.uniform(0,2*np.pi)
            costheta=random.uniform(-1.,1.)
            phi=random.uniform(0,2*np.pi)
            iota=np.arccos(cosiota)
            theta=np.arccos(costheta)
        
            # Merger time
            tmerger=np.random.uniform(ttmin*year,tmax*year)
        
            #fmin=np.random.uniform(1e-5,0.01)
            fmin=self.findFmin(tmerger,m1,m2)
            Fmax= self.getfmax(m1*(1+z),m2*(1+z),fmin/(1+z),tobs*year)  #fmax(m1,m2,fmin,tobs*year)
           
            snr=np.sqrt(dutycycle)*self.SNR(iota,psi,theta,phi,fmin/(1+z),Fmax,m1*(1+z),m2*(1+z),dl, leng)


            dVcdz = cosmo.comoving_volume(z).value 
        
            
            integralbulk = (tmax-ttmin) * (zmax-zmin) * dVcdz   *np.heaviside(snr-SNRth,0) * (1./(1.+z))

            
            data.append(np.array([m1,m2,z,fmin,integralbulk,snr]))

        return np.array(data).T

    def consolidatedistrIntrinsic(self, Nsingle,iterations, massoption,SNRth, Tobs, duty, ttmin, ttmax, leng):

        Nsingle = int(Nsingle)
        iterations = int(iterations)

        Ntot = Nsingle*iterations
        data=[]

        for it in range(0,iterations):
            data.append(self.gendistrIntrinsic(Nsingle,iterations, massoption,SNRth, Tobs, duty, ttmin, ttmax, leng))

        data=np.array(data)
    
        m1 = np.concatenate(data[:,0])
        m2 = np.concatenate(data[:,1])
        z = np.concatenate(data[:,2])
    
        fmin = np.concatenate(data[:,3])
        integralbulk = np.concatenate(data[:,4])
        SNR10 = np.concatenate(data[:,5])

        return m1,m2,z,fmin,integralbulk,SNR10

    def NeventsIntrinsic(self, Nsingle,iterations, massoption,SNRth, Tobs, duty, ttmin, ttmax, leng):

        bigdata = self.consolidatedistrIntrinsic(Nsingle,iterations, massoption,SNRth, Tobs, duty, ttmin, ttmax, leng)

        # bigdata is the output of consolidatedistr
        m1,m2,z,fmin,integralbulk,SNR10 = bigdata 

        Ntot = len(m1)

        montecarlo_contributions = integralbulk / Ntot

        return np.sum(montecarlo_contributions) * 1e-9

class NumberRates(NumberIntrinsic):

    def __init__(self, **kwargs):
        NumberIntrinsic.__init__(self, **kwargs)

    def Poisson(self, Nmed, N1, N2):

        arr1 = np.array([np.random.poisson(np.array([np.random.gamma(2, 0.596*0.13)*Nmed for n in range(N1)])) for m in range(N2)])
        arr = arr1.flatten()
    
        return np.median(arr), np.percentile(arr,95)- np.median(arr), np.percentile(arr,5)- np.median(arr)

    def NumberEvents(self, N1, N2, Nsingle,iterations, massoption,SNRth, Tobs, duty, ttmin, ttmax, leng):

        NmedToUse = self.NeventsIntrinsic(Nsingle,iterations, massoption,SNRth, Tobs, duty, ttmin, ttmax, leng)

        return self.Poisson(NmedToUse, N1, N2)
