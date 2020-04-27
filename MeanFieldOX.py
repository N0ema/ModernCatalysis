import numpy as np
from scipy.integrate import ode
from database_pt111 import *
import matplotlib.pyplot as plt


def derivatives(t, ini, conditions):

    T, pCO, pO2 = conditions
    theta_CO, theta_O = ini

    theta_star = 1 - theta_CO - theta_O

    k1_plus = W1f(T, pCO, theta_CO, theta_O, theta_star)
    k1_minus = W1b(T, pCO, theta_CO, theta_O, theta_star)
    k2_plus = W2f(T, pO2, theta_CO, theta_O, theta_star)
    k2_minus = W2b(T, pO2, theta_O, theta_CO, theta_star)
    k3_plus = W3f(T, theta_CO, theta_O, theta_star)
    
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


if __name__ == "__main__":

    T = 700.
    pCO = .5e2
    pO2 = 3e2

    conditions = [T, pCO, pO2]
    tini = 0
    tfin = 1e4
    dt = tfin / 10.
    t = np.arange(tini, tfin, dt)

    ini_cov = [0.8, 0.0]

    species = ["CO", "O"]

    ####Plot steady state coverages####
    # covs = solve_cov(conditions, ini_cov, ret_all_t=True, tini=0., tfin=tfin, dt=dt)
    # for i in range(len(species)):
    #     plt.plot(t, covs[i], '-o', label=species[i])
    # plt.ylim([-0.01, 1.0])
    # plt.grid()
    # plt.legend()
    # plt.show()

    ###Solving in T####

    T_up = np.linspace(400., 1000., 20)
    T_down = np.linspace(1000., 400., 20)
    conditions_up = [T_up, pCO, pO2]
    conditions_down = [T_down, pCO, pO2]

    z = temp_solve(conditions_up, ini_cov)
    y = temp_solve(conditions_down, ini_cov)

    plt.plot(T_up, z[0], label='CO')
    plt.plot(T_down, y[0], label='CO')
    plt.plot(T_up, z[1], label='O')
    plt.plot(T_down, y[1], label='O')
    plt.plot(T_up, (1 - z[0] - z[1]), '--', label='O')
    plt.ylim([-0.01, 1.0])
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(T_up, get_TOF(z, conditions_up), '-o')
    plt.plot(T_down, get_TOF(y, conditions_down), '--')
    plt.show()
