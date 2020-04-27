"""

CO(g) + *   <->  CO*      (1)
O2(g) + *   <->  O2*      (2)
CO*   + O*  <->  CO2(g)   (3)
O2*   + *   <->  2O*      (4)
2O*          ->  O2(g)    (5)

Herein database.py goes the parameters(constants) and
rate functions that later on is appened to
a list.
"""

import numpy as np
from constants import *
from prefactors import *

print("Loading Pt(111) harmonic database .....")
# Vibration Modes on Pd(100):
El = 12.3984  # 100 cm^-1 in meV

Vib_O = 1E-3 * np.array([47.3, 47.6, 55.1])
Vib_CO = 1E-3 * np.array([17.8, 18.3, 36.3, 36.3, 39.7, 216.8])

# Vibrations in gas phase:
Vib_O2_gas = 1E-3 * np.array([El, El, El, El, El, 191.2])
Vib_CO2_gas = 1E-3 * np.array([El, El, El, El, El, 78.4, 78.4, 163.5, 293.3])
Vib_CO_gas = 1E-3 * np.array([El, El, El, El, El, 265.7])

# Energies without ZPE corrections plus the ZPE correction:
EO2gas = -10.542858 + Vib_O2_gas.sum() / 2.
ECO2gas = -23.6375 + Vib_CO2_gas.sum() / 2.
ECOgas = -15.308574 + Vib_CO_gas.sum() / 2.

EO = -94.900299 + Vib_O.sum() / 2.
ECO = -105.375296 + Vib_CO.sum() / 2.

Eclean = -88.654122

Vib_TS_COOx = 1E-3 * np.array([El, 19.9, 36.4, 40.4, 50.5, 54.6, 65.1, 245.])
Vib_TS_COads = 1E-3 * np.array([El, El, El, El, 262.7])
Vib_TS_O2ads = 1E-3 * np.array([El, El, El, El, 107.5])

# Barriers for reactions:
# CO(g) + * <-> CO*
Ea1f = 0.001
Ea1b = Ea1f + (ECOgas - ECO + Eclean)

# O2(g)+2* <-> 2O*
Ea2f = 0.001
Ea2b = Ea2f - (2. * EO - 2. * Eclean - EO2gas)

# CO*+O* <-> CO2(g) + 2*
Ea3f = 1.06 + (Vib_TS_COOx.sum() / 2. - Vib_CO.sum() - Vib_O.sum()) / 2.  #
Ea3b = Ea3f - (ECO2gas - EO - ECO + 2. * Eclean)

EdiffO = 0.58
EdiffCO = 0.08

# print 2.*(EO-Eclean-0.5*EO2gas)
# print("CO", ECO-Eclean-ECOgas)

# print("Warning Wrong EdiffO2 value")

# Repulsion /attraction slopes
betaO_CO = 1.15  # 1.15#1.99#how O affects CO
betaCO_O = 2. * 2.29  # 2.*2.29
beta = 0.0

print("Nsites = ", Nsites, " A = ", A, " b = ", b)

# Unused modes in the Hindered formalism:
# ----------------------------------------
COmodes = 1E-3 * np.array([38.7, 38.9, 42.5, 220.1])
O2modes = 1E-3 * np.array([28.7, 49.5, 67.8, 107.5])


# -----------------------------------
# Rate functions
def EbCO(thetaCO):
    """
    Returns the binidng energy of CO with the CO-CO lateral
    repulsions, no cross-terms
    """
    # return 1.392696-0.04787676*np.exp(3.72541193*thetaCO)
    return 1.36325 - 0.04787676 * np.exp(3.72541193 * thetaCO)


def Eb2O(thetaO):
    """
    Returns the binidng energy of 2O with the O-O lateral
    repulsions, no cross-terms
    """
    # return 1.927692-0.36266796*np.exp(2.98937029*thetaO)
    return 1.895069 - 0.36266796 * np.exp(2.98937029 * thetaO)


def W1f(T, pCO, thetaCO, thetaO, thetastar):  # CO(g) + * ->    CO*
    return 0.9 * Asite * pCO / np.sqrt(2. * np.pi * mCO * kB * eV2J * T)


def W1b(T, pCO, thetaCO, thetaO, thetastar):
    Nfree = max(1., thetastar * Nsites)
    Ecur = -(EbCO(thetaCO) - betaO_CO * thetaO)
    if Ecur > 0.:
        Ecur = -0.0001

    Sgas = get_entropy_CO(T, pCO)
    Ssurf = get_entropy_ads(Vib_CO, T) + kB * np.log(Nsites)
    K = np.exp(-Ecur / (kB * T)) * np.exp((Ssurf - Sgas) / kB) / Nsites
    return W1f(T, pCO, thetaCO, thetaO, thetastar) / K


def W2f(T, pO2, thetaCO, thetaO, thetastar):  # O2(g)+* -> 2O*
    return 0.1 * Asite * pO2 / np.sqrt(2. * np.pi * mO2 * kB * eV2J * T)


def W2b(T, pO2, thetaO, thetaCO, thetastar):  # 2O* -> O2(g) + *
    Ecur = -(Eb2O(thetaO) - betaCO_O * thetaCO)
    if Ecur > 0.:
        Ecur = -0.0001
    Sgas = get_entropy_O2(T, pO2)
    Ssurf = 2. * get_entropy_ads(Vib_O, T) + kB * np.log(Nsites)
    K = np.exp(-Ecur / (kB * T)) * np.exp((Ssurf - Sgas) / kB) / Nsites
    # print "K =", K
    return W2f(T, pO2, thetaO, thetaCO, thetastar) / K


def W3f(T, thetaCO, thetaO, thetastar):  # CO* + O* -> CO2(g) + 2*
    Zini = get_Zvib(Vib_CO, T) * get_Zvib(Vib_O, T)
    Zts = get_Zvib(Vib_TS_COOx, T)
    return np.exp(-Ea3f / (kB * T)) * kB * T * Zts / (h * Zini)