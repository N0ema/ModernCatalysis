"""
Calculates pre-factors of reaction steps through the vibrational
partition functions and also the translational and rotational ones.
"""

from constants import *
import numpy as np
from math import factorial
# Integer order modified bessel function
# of the second kind
from scipy.special import i0 as I0
from scipy.special import i1 as I1

# Constants ----------------------------------------
trans_fact = (2. * np.pi * kB * eV2J) ** (3. / 2.) / (h * eV2J) ** 3.0
a0 = 4.00E-10
ICO = 1.50752694e-46  # kgm^2 Moment of Inertia CO.
sigmaCO = 2.


def get_Zrot(T, Natoms, I, sigma):
    """
    Get the rotational part of the partition function.
    Takes the temperature(T[K]), the moment of inertia(possible 3 element
    array of moments along principal axes x,y and z. sigma is the symmetry factor,
    that must be looked up for the given molecule..
    """
    if Natoms == 2:
        assert len(np.array(I)) == 1
        return 18. * np.pi ** 2.0 * I[0] * kB * eV2J * T / (h * eV2J) ** 2. / sigma
    if Natoms == 1:
        return 1
    if Natoms > 2 and len(np.array([I])) > 1:
        return (8. * np.pi * kB * eV2J * T / (eV2J * h) ** 2.) ** (3. / 2.) * np.sqrt(
            np.pi * I[0] * I[1] * I[2]) / sigma  # I0 = Ix, I1 = Iy, I2=Iz


def get_Ztrans(T, V, m):
    """
    Calculates the translational parition function
    given the number of translation DOF(Nmodes), the
    Volume it can move in(V[m^3]) and the molecular mass m[kg].
    """
    return trans_fact * V * (m * T) ** (3. / 2.)


def get_Zvib(modes, T):
    """
    Calculate the vibrational part of the parition function.
    needs the temperature(T[K]) and the modes energies(modes[eV]).
    modes must be a numpy array(ndarray).
    """
    beta = 1. / (kB * T)
    ZPE = modes.sum() / 2.
    return np.exp(-ZPE * beta) * ((1. / (1. - np.exp(-modes * beta))).prod())


def get_entropy_ads(modes, T):
    """
    Get the entropy[eV/K] of an adsorbed molecule for a given temperature T[K],
    and the vibrational energies(modes(np.array[eV]).
    """
    S = 0.
    beta = 1. / (kB * T)
    # print "Zvib ", get_Zvib(modes,T), " modes sum/2", modes.sum()/2.
    for m in modes:
        S += -kB * np.log(1. - np.exp(-m * beta)) + m * np.exp(-beta * m) / (T * (1. - np.exp(-m * beta)))

    return S


def get_Z_CO(T, pCO):
    """
    Get the entopy[eV/K] of CO in the gas phase. It is a function
    of the temperature[T].
    """
    modes = 1E-3 * np.array([263.4])  # Vibrational modes of CO
    I = 1.50752694e-46  # Moments of inertia along princpal axes(x,y,z)(from ASE)
    sigma = 1.  # Symmetry factor of methane.
    m = 28.01E-3 / Na  # mass[kg]
    V = kB * eV2J * T / pCO  # 2 ang cubic volume
    Zrot = 8. * np.pi ** 2. * I * kB * eV2J * T / sigma / (h * eV2J) ** 2.  # I0 = Ix, I1 = Iy, I2=Iz
    Zvib = get_Zvib(modes, T)
    Ztrans = get_Ztrans(T, V, m)
    Z = Ztrans * Zvib * Zrot
    # print "ZT",Ztrans,Zrot,Zvib
    return Z


def get_Z_O2(T, pO2):
    sigma = 2.  # 1 possible simple rotation leaving unchanged
    I = 2.06218774e-46  # Moment of Inertia
    V = kB * eV2J * T / pO2
    m = 15.9994E-3 * 2. / Na
    modes = 1E-3 * np.array([194.4])

    Ztrans = get_Ztrans(T, V, m)
    Zvib = get_Zvib(modes, T)
    Zrot = 8. * np.pi ** 2. * I * kB * eV2J * T / sigma / (h * eV2J) ** 2.  # Valid for T > h^2/(8pi^2IkB)

    Z = Ztrans * Zvib * Zrot

    return Z


def get_Z_CO2(T, pCO2):
    sigma = 2.  # 2 possible simple rotations leaving unchanged
    I = 7.38173408e-46  # Moment of Inertia
    V = kB * eV2J * T / pCO2
    m = 44.01E-3 / Na
    modes = 1E-3 * np.array([78.4, 78.4, 163.5, 293.3])

    Ztrans = get_Ztrans(T, V, m)
    Zvib = get_Zvib(modes, T)
    Zrot = (8. * np.pi ** 2. * I * kB * eV2J * T / (sigma * h ** 2. * eV2J ** 2.))

    Z = Ztrans * Zvib * Zrot
    return Z


def get_Zhindtrans(T, Ea, b, m, Nfree):
    """
    Get the partition function of a hindered
    translator on Pt(111).
    """
    mode = np.sqrt(Ea * eV2J / (m * np.sqrt(3.) * b ** 2.)) * h
    M = Nfree  # How many sites ads. is free to move over.
    Tx = kB * T / mode
    rx = Ea / mode

    frac1 = M * (np.pi * rx / Tx) * np.exp(-rx / Tx) * np.exp(-1. / Tx) * I0(rx / 2. / Tx) ** 2. * np.exp(
        2. / Tx / (2. + 16. * rx))
    frac2 = (1. - np.exp(-1. / Tx)) ** 2.

    return frac1 / frac2


def get_ZhindtransWOC(T, Ea, b, m, Nfree):
    """
    Get the partition function of a hindered
    translator on Pt(111) without the ZPE correction factor.
    """
    mode = np.sqrt(Ea * eV2J / (m * np.sqrt(3.) * b ** 2.)) * h
    M = Nfree  # How many sites ads. is free to move over.
    Tx = kB * T / mode
    rx = Ea / mode

    frac1 = M * (np.pi * rx / Tx) * np.exp(-rx / Tx) * np.exp(-1. / Tx) * I0(rx / 2. / Tx) ** 2.
    frac2 = (1. - np.exp(-1. / Tx)) ** 2.

    return frac1 / frac2


def get_Shindered(T, Ea, b, m, Nfree, intmodes):
    """
    Get the hindered translator entropy. The mode should be calculated as
    by Campbell. That is the HMO int-vib modes still contribute, but also
    the hindered translator modes contribute to the entropy.
    """
    mode = np.sqrt(Ea * eV2J / (m * np.sqrt(3.) * b ** 2.)) * h
    # print "Mode eV ", mode*J2eV, " Ea = ", Ea, "m = ", m, " b = ", b, "h = ", h
    M = Nfree  # How many sites ads. is free to move over.
    Tx = kB * T / mode
    rx = Ea / mode
    SHOhind = 2. * get_entropy_ads(np.array([mode]), T)
    SHOint = get_entropy_ads(intmodes, T)

    Shind = 2. * kB * (-1. / 2. - rx * I1(rx / 2. / Tx) / (2. * Tx * I0(rx / 2. / Tx)) + np.log(
        np.sqrt(np.pi * rx / Tx) * I0(rx / 2. / Tx)))
    Sconf = kB * np.log(M)
    return Shind + SHOint + Sconf + SHOhind


def get_entropy_O2(T, pO2):
    """
    Return the entropy of O2 in 3D gas phase.
    """
    sigma = 1.  # 1 possible simple rotation leaving unchanged
    I = 2.06218774e-46  # Moment of Inertia
    V = kB * eV2J * T / pO2
    m = 15.9994E-3 * 2. / Na
    modes = 1E-3 * np.array([194.4])

    Ztrans = get_Ztrans(T, V, m)
    Zvib = get_Zvib(modes, T)
    Zrot = 8. * np.pi ** 2. * I * kB * eV2J * T / sigma / (h * eV2J) ** 2.  # Valid for T > h^2/(8pi^2IkB)

    Strans = kB * np.log(Ztrans) + 3. / 2. * kB
    Svib = get_entropy_ads(modes, T)
    Srot = kB * np.log(Zrot) + kB

    return Strans + Svib + Srot


def get_entropy_CO(T, pCO):
    """
    Return the entropy of CO in 3D gas phase.
    """
    modes = 1E-3 * np.array([263.4])  # Vibrational modes of CO
    I = 1.50752694e-46  # Moments of inertia along princpal axes(x,y,z)(from ASE)
    sigma = 2.  # Symmetry factor of CO.
    m = 28.01E-3 / Na  # mass[kg]
    V = kB * eV2J * T / pCO  # 2 ang cubic volume

    Ztrans = get_Ztrans(T, V, m)
    Zvib = get_Zvib(modes, T)
    Zrot = 8. * np.pi ** 2. * I * kB * eV2J * T / sigma / (h * eV2J) ** 2.  # Valid for T > h^2/(8pi^2IkB)

    Strans = kB * np.log(Ztrans) + 3. / 2. * kB
    Svib = get_entropy_ads(modes, T)
    Srot = kB * np.log(Zrot) + kB

    return Strans + Svib + Srot


def get_entropy_CO2(T, pCO2):
    """
    Return the entropy of CO2 in 3D gas phase.
    """
    sigma = 2.  # Symmetry factor of CO.
    I = 7.38173408e-46  # Moment of Inertia
    V = kB * eV2J * T / pCO2
    m = 44.01E-3 / Na
    modes = 1E-3 * np.array([78.4, 78.4, 163.5, 293.3])

    Ztrans = get_Ztrans(T, V, m)
    Zvib = get_Zvib(modes, T)
    Zrot = (8. * np.pi ** 2. * I * kB * eV2J * T / (sigma * h ** 2. * eV2J ** 2.))

    Strans = kB * np.log(Ztrans) + 3. / 2. * kB
    Svib = get_entropy_ads(modes, T)
    Srot = kB * np.log(Zrot) + kB

    return Strans + Svib + Srot


def get_Z_2dgasCO(A, T):
    """
    Returns the translational entropy of a 2D ideal gas
    of 1 molecule with mass M[kg],  which is free to move
    over an area A[M], at temperature T[K].
    """

    Ztrans2D = 2. * np.pi * mCO * kB * eV2J * T * A / (h * eV2J) ** 2.
    return Ztrans2D


def get_Z_2dgasO(A, T):
    """
    Returns the translational entropy of a 2D ideal gas
    of 1 molecule with mass M[kg],  which is free to move
    over an area A[M], at temperature T[K].
    """
    Ztrans2D = 2. * np.pi * mO * kB * eV2J * T * A / (h * eV2J) ** 2.
    return Ztrans2D  # *Zvib


def get_Z_2dgasO2(A, T):
    """
    Returns the translational entropy of a 2D ideal gas
    of 1 molecule with mass M[kg],  which is free to move
    over an area A[M], at temperature T[K].
    """
    Ztrans2D = 2. * np.pi * mO2 * kB * eV2J * T * A / (h * eV2J) ** 2.
    return Ztrans2D


def get_S_2dgasO(A, T):
    """
    Return the entropy of 2 free translation modes for O
    """
    return kB * np.log(get_Z_2dgasO(A, T)) + kB


def get_S_2dgasCO(A, T):
    """
    Return the entropy of 2 free translation modes for CO
    """
    return kB * np.log(get_Z_2dgasCO(A, T)) + kB


def get_Zhindrot(T, Wr, rotmode):
    Tr = kB * T / rotmode
    rr = Wr / kB / T
    frac1 = np.sqrt(np.pi * rr / Tr) * np.exp(-rr / 2. / Tr) * np.exp(-1. / 2 / Tr) * I0(rr / 2. / Tr) * np.exp(
        1. / Tr / (2. + 16. * rr))
    frac2 = (1. - np.exp(-1. / Tr))
    return frac1 / frac2


def get_Ztrans1D(T, l, m):
    """
    Calculates the translational parition function
    given the number of translation DOF(Nmodes), the
    Volume it can move in(V[m^3]) and the molecular mass m[kg].
    """
    return trans_fact ** (1. / 3.) * l * (m * T) ** (1. / 2.)


get_entropy_CO2(500, 1E1)

"""
import matplotlib.pyplot as plt
Ea = 0.1
T = np.linspace(1.,900.,100)
mode = 69E-3
Tx = kB*T/(mode)


plt.plot(Tx,Ptrans(T,Ea,mode))
plt.show()
"""