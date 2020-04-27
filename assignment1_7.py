"""
The program plot the fractional coverages of CO, O, free sites and the rate for CO oxidation.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton_krylov


def myFunction(z, pCO, pO2, rates):
    """Return the derivatives of CO and O coverage
    Input = z
        pCO : CO pressure
        pO : O pressure
        rates : elementary steps' rate constants
        """
    x = z[0]
    y = z[1]

    k1_plus, k1_minus, k2_plus, k2_minus, k3_plus = rates
    F = np.empty(2)

    F[0] = k1_plus * pCO * (1 - x - y) - k1_minus * x - k3_plus * x * y
    F[1] = k2_plus * pO2 * pow((1 - x - y), 2) - k2_minus * pow(y, 2) - k3_plus * x * y

    return F


pO2 = np.linspace(0.01, 1.0, 1000)
pCO = np.linspace(1., 0.01, 1000)

zGuess = np.array([0., 0.1])

alpha = pCO / (pCO + pO2)
alpha_down = np.flip(alpha)
print(alpha, alpha_down)

cov_CO = np.zeros(len(alpha))
cov_O = np.zeros(len(alpha))

rate_constants = [7651497.962993699, 7649114.857161215, 122549.93269798308, 121357.91938683724, 0]

eps = 1e-12
check = []
for i in range(len(alpha)):
    z = newton_krylov(lambda z: myFunction(z, pCO[i], pO2[i], rate_constants), zGuess)

    zGuess = z
    cov_CO[i] = z[0]
    cov_O[i] = z[1]
    check.append(abs(myFunction(z, pCO[i], pO2[i], rate_constants)[1]) < eps)

if all(item == True for item in check):
    print('Solution found for all values')
else:
    print('Solution not found for ' + str(sum([1 for item in check if not item])) + ' values')

rate = 2 * cov_CO * cov_O

plt.plot(alpha, rate, label='Rate')
plt.plot(alpha_down, rate, label='Rate')
plt.plot(alpha, cov_CO, label='CO')
plt.plot(alpha, cov_O, label='O')
plt.plot(alpha, (1 - cov_O - cov_CO), '--', label='empty sites')
plt.xlabel("$\\alpha=\\frac{p_{CO}}{p_{CO} + p_{O_{2}}}$", fontsize=14)
plt.ylabel('Coverage', fontsize=14)
plt.legend(loc='upper right')
plt.grid()
plt.savefig('./assignment_17.eps', format='eps', bbox_inches='tight')
plt.show()