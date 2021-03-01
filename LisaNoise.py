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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cmasher as cmr
import matplotlib.colors as clrs
import matplotlib.cm as cmap
import cmath
from scipy.optimize import root_scalar

# we work in geometrized units

Msol = 1.47680000000000000000000000000000000 # in km
c = 1 # we have actually put them to one everywhere in the code already
G = 1
year = 60*60*24*(365.250000000000000000000000000); # s
cc = 2.9980000000000000000000000000000*10**5 # velocity of light in Km/s
pc = 3.0856770000000000000000000000*10**13 # km

H0 = 67.9 # km /s / Mpc # changed to PLANCK 2015
Omegam = 0.306
OmegaL = 0.694
Omegar = 0.



#here we want to add also the part with the SNR 

# LISA noise parameters
Amp = 5/10*18/10*10**-44
alpha = 138/1000
beta = -221
kappa = 521
gamma = 1680
fk = 113/10**5
L0 = 25/10*10**9
fstar = 1909/100*10**-3
pm = 10**-12

skyav=(4/5*sqrt(2)*sin(pi/3))

# noise function, checked with Mathematica

def Sacc(f):
    return (3*10**-15)**2*(1+(4/10*10**-3/f)**2)*(1+(f/(8*10**-3))**4)

def Sgal(f):
    return Amp*f**(-7/3)*exp(-f**alpha+beta*f*sin(kappa*f))*(1+tanh(gamma*(fk-f)))

def Soms(f, len):
    return (len *pm)**2*(1+(2*10**-3/f)**4)  

def Sacc(f):
    return (3*10**-15)**2*(1+(4/10*10**-3/f)**2)*(1+(f/(8*10**-3))**4)
    
def SnSA(f, len):
    return 10/3*(4*Sacc(f)/(2*pi*f)**4+Soms(f, len))/(L0**2)*(1+6/10*(f/fstar)**2)

def R(f):
    return 3/20*2

def Sn(f, len):
    return abs(SnSA(f, len)+Sgal(f))

def Pn(f, len): # gravitational wave frequency
    return abs(Sn(f, len)*R(f))

def Aplus(iota,psi):
    return -(1+cos(iota)**2)*cos(2*psi)-2*cos(iota)*sin(2*psi)

def Across(iota,psi):
    return (1+cos(iota)**2)*sin(2*psi)-2*cos(iota)*cos(2*psi)

def Fplus(theta,phi):
    return 1/2*(1+cos(theta)**2)*cos(2*phi)

def Fcross(theta,phi):
    return cos(theta)*sin(2*phi)

def factorsky(iota,psi,theta,phi):
    return Fplus(theta,phi)*Aplus(iota,psi)+1j*Fcross(theta,phi)*Across(iota,psi)

def factorskySNR(iota,psi,theta,phi):
    return abs(sqrt(2)*sin(pi/3)*(Fplus(theta,phi)*Aplus(iota,psi)+1j*Fcross(theta,phi)*Across(iota,psi)))

