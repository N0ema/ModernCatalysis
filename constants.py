import numpy as np

# Conversion FACTORS
eV2J = 1.60217662E-19  # Joule to eV conversion
J2eV = 1.0 / eV2J  # eV to Joule

# Constants
kB = 8.6173324E-5  # eV/K
R = 8.3144598 * J2eV  # Gas constant in eV/(K*mol)
Na = 6.0221409E23  # Avogadro's number mol^-1
h = 6.62607004E-34 * J2eV  # Planck's Constant in eV*s
hbar = 1.0545718E-34 * J2eV  # Hbar = h/2Pi
e = 1.60217662E-19  # Coulombs
pi = 3.14159265359  # pi

mCO = 28.01E-3 / Na
mO2 = 2. * 15.9994E-3 / Na
mO = 15.9994E-3 / Na

# Surface Geometry
a0 = 4.00E-10
b = a0 / np.sqrt(2)  # NN distance
Asite = ((4.00E-10) ** 2) * np.sqrt(3) / 4
# A = (10E-10)**2. # The ca. area of a facet on NP.
A = Asite
Nsites = A / Asite

sigma = 2. * (1E-2) ** 2. / (b ** 2. * np.sqrt(3) / 2.)
