"""
The program plot the fractional coverages of CO, O, free sites and the rate for CO oxidation
at steady state conditions.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton_krylov


# def myFunction(z, pCO, pO2, rates):
#     """Return the derivatives of CO and O coverage
#     Input = z
#         pCO : CO pressure
#         pO : O pressure
#         rates : elementary steps' rate constants
#         """
#     x = z[0]
#     y = z[1]
#
#     k1_plus, k1_minus, k2_plus, k2_minus, k3_plus = rates
#     F = np.empty(2)
#
#     F[0] = k1_plus * pCO * (1 - x - y) - k1_minus * x - k3_plus * x * y
#     F[1] = k2_plus * pO2 * pow((1 - x - y), 2) - k2_minus * pow(y, 2) - k3_plus * x * y
#
#     return F

import numpy as np
from scipy.integrate import ode
from database_pt111 import *
import matplotlib.pyplot as plt


def derivatives(t, ini, conditions, rates):
    T, pCO, pO2 = conditions
    theta_CO, theta_O = ini

    theta_star = 1 - theta_CO - theta_O

    k1_plus, k1_minus, k2_plus, k2_minus, k3_plus = rates
    dCOdt = k1_plus * theta_star - k1_minus * theta_CO - k3_plus * theta_CO * theta_O
    dOdt = k2_plus * pow(theta_star, 2) - k2_minus * pow(theta_O, 2) - k3_plus * theta_CO * theta_O

    return [dCOdt, dOdt]


def solve_cov(conditions, ini, tini=0., tfin=1E4, dt=1E2, ret_all_t=False):
    """
    Get the coverages for a certain set of conditions,
    conditions is a list of pressures and temperatures,
    ini is a list of initial guesses of coverages,
    t is a list of times, and pret_all_t returns coverage vs. time
    if true.
    """
    o = ode(derivatives)

    # BDF method suited to stiff systems of ODEs
    o.set_integrator('lsoda', nsteps=1E9, method='bdf', max_hnil=30, min_step=1E-50)
    o.set_initial_value(ini, tini)
    o.set_f_params(conditions)
    res = []
    while o.successful() and o.t < tfin:
        res.append(o.integrate(o.t + dt))

    cov_CO = [v[0] for v in res]
    cov_O = [v[1] for v in res]

    coverages = [cov_CO, cov_O]

    if not ret_all_t:
        return [o[-1] for o in coverages]
    else:
        return coverages


def get_TOF(coverages, conditions):
    theta_CO, theta_O = coverages
    T, pCO, pO2 = conditions

    TOF = np.zeros(len(T))

    for i in range(len(T)):
        theta_star = 1 - theta_CO[i] - theta_O[i]
        TOF[i] = W3f(T[i], theta_CO[i], theta_O[i], theta_star) * theta_CO[i] * theta_O[i]

    return TOF


def temp_solve(conditions, ini_cov):
    T, pCO, pO2 = conditions
    cov_CO = np.zeros(len(T))
    cov_O = np.zeros(len(T))

    for i, T in enumerate(T):
        conditions = [T, pCO, pO2]
        covs = solve_cov(conditions, ini_cov, tfin=tfin, dt=dt)
        print(conditions, T, covs, ini_cov)

        ini_cov = covs

        cov_CO[i] = covs[0]
        cov_O[i] = covs[1]

    coverages = [cov_CO, cov_O]

    return coverages


pCO = np.linspace(0.01, 1.0, 1000)
pO2 = np.linspace(1., 0.01, 1000)
alpha = pCO / (pCO + pO2)

zGuess = np.array([0.1, 0.5])


cov_CO = np.zeros(len(alpha))
cov_O = np.zeros(len(alpha))

# rate_constants = [13e4, 6e4, 9e8, 8e8, 2e1]
rate_constants = [7651497.962993699, 7649114.857161215, 122549.93269798308, 121357.91938683724, 2383.4774406494134]

# eps = 1e-12
# check = []
# scipy.integrate.ode(f).set_integrator('vode', method='bdf', order=15)
# for i in range(len(alpha)):
#     z = newton_krylov(lambda z: myFunction(z, pCO[i], pO2[i], rate_constants), zGuess)
#
#     zGuess = z
#     cov_CO[i] = z[0]
#     cov_O[i] = z[1]
#     check.append(abs(myFunction(z, pCO[i], pO2[i], rate_constants)[1]) < eps)
#
# if all(item == True for item in check):
#     print('Solution found for all values')
# else:
#     print('Solution not found for ' + str(sum([1 for item in check if not item])) + ' values')


rate = rate_constants[4] * cov_CO * cov_O
rate_norm = (rate - min(rate))/(max(rate)-min(rate))
plt.plot(alpha, rate_norm, label='Normalized Rate')
plt.plot(alpha, cov_CO, label='CO')
plt.plot(alpha, cov_O, label='O')
plt.plot(alpha, (1 - cov_O - cov_CO), '--', label='empty sites')
plt.xlabel("$\\alpha=\\frac{p_{CO}}{p_{CO} + p_{O_{2}}}$", fontsize=14)
plt.ylabel('Coverage', fontsize=14)
plt.legend(loc='upper right')
plt.grid()
plt.savefig('./assignment_16.eps', format='eps', bbox_inches='tight')
plt.show()