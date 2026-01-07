# Расчетный модуль:
# 1. eyring_k(T, dG) - уравнение Эйринга
# 2. dydt(t, y, reactions) - Система диф. уравнений
# 3. calculate(p, c_b_0, c_e_0, c_p_0, c_mb_0, t_max, T, G_M) - численное решение системы
# 4. optimal_T(p, c_b_0, c_e_0, c_p_0, c_mb_0, t_max, T_min=350.0, T_max=450.0, step=10) - подбор оптимальной температуры по C_EB

import numpy as np
from scipy.integrate import solve_ivp

from Alkyl_Consts import (
    R, k_B_const, h, molar_masses, Reaction, substances,
    cp_B, cp_E, cp_EB, cp_P, cp_PB, cp_MB, cp_MEB, cp_DFE, cp_DEB, cp_H,
    cp_coeffs, int_cp_from_poly, H298r, T_ref_cp
)


def eyring_k(T, dG):
    return (k_B_const * T / h) * np.exp(-dG / (R * T))


def dydt(t, y, reactions):
    r_11, r_12, r_21, r_22, r_31, r_32, r_41, r_42, r_51, r_52 = reactions  # распаковка массива реакций
    C_B, C_E, C_EB, C_P, C_PB, C_MB, C_MEB, C_DFE, C_DEB, C_H = y  # распаковка массива концентраций

    k_B = 0.16
    z = (1 + k_B) ** 2  # коэффициент адсорбции

    # скорости реакций
    w1 = (r_11.k * C_B * C_E - r_12.k * C_EB) / z
    w2 = (r_21.k * C_B * C_P - r_22.k * C_PB) / z
    w3 = (r_31.k * C_MB * C_E - r_32.k * C_MEB) / z
    w4 = (r_41.k * C_B * C_EB - r_42.k * C_DFE * C_H) / z
    w5 = (r_51.k * C_EB * C_E - r_52.k * C_DEB) / z

    dC_B   = -w1 - w2 - w4
    dC_E   = -w1 - w3 - w5
    dC_EB  =  w1 - w4 - w5
    dC_P   = -w2
    dC_PB  =  w2
    dC_MB  = -w3
    dC_MEB =  w3
    dC_DFE =  w4
    dC_DEB =  w5
    dC_H   =  w4

    return [dC_B, dC_E, dC_EB, dC_P, dC_PB, dC_MB, dC_MEB, dC_DFE, dC_DEB, dC_H]


def heat_balance_for_state(C_vec, y0, T, T1):
    # исходные и конечные концентрации
    C_B_0, C_E_0, _, C_P_0, _, C_MB_0, _, _, _, _ = y0
    C_B_k, C_E_k, C_EB_k, C_P_k, C_PB_k, C_MB_k, C_MEB_k, C_DFE_k, C_DEB_k, C_H_k = C_vec

    T0 = T_ref_cp  # базовая температура

    def I_all(Tloc):
        I_B   = int_cp_from_poly(cp_coeffs['C_B'],   Tloc, T0)
        I_E   = int_cp_from_poly(cp_coeffs['C_E'],   Tloc, T0)
        I_EB  = int_cp_from_poly(cp_coeffs['C_EB'],  Tloc, T0)
        I_P   = int_cp_from_poly(cp_coeffs['C_P'],   Tloc, T0)
        I_PB  = int_cp_from_poly(cp_coeffs['C_PB'],  Tloc, T0)
        I_MB  = int_cp_from_poly(cp_coeffs['C_MB'],  Tloc, T0)
        I_MEB = int_cp_from_poly(cp_coeffs['C_MEB'], Tloc, T0)
        I_DFE = int_cp_from_poly(cp_coeffs['C_DFE'], Tloc, T0)
        I_DEB = int_cp_from_poly(cp_coeffs['C_DEB'], Tloc, T0)
        I_H   = int_cp_from_poly(cp_coeffs['C_H'],   Tloc, T0)
        return I_B, I_E, I_EB, I_P, I_PB, I_MB, I_MEB, I_DFE, I_DEB, I_H

    # интегралы при T и T1
    I_B_T, I_E_T, I_EB_T, I_P_T, I_PB_T, I_MB_T, I_MEB_T, I_DFE_T, I_DEB_T, I_H_T = I_all(T)
    I_B_1, I_E_1, I_EB_1, I_P_1, I_PB_1, I_MB_1, I_MEB_1, I_DFE_1, I_DEB_1, I_H_1 = I_all(T1)

    # нагрев реагентов от T0 до T
    Qin = (
        C_B_0  * I_B_T +
        C_E_0  * I_E_T +
        C_P_0  * I_P_T +
        C_MB_0 * I_MB_T
    )

    # ΔCp для реакций при T
    dI1 = I_EB_T - (I_B_T + I_E_T)                     # Б+Э -> ЭБ
    dI2 = I_PB_T - (I_B_T + I_P_T)                     # Б+П -> ПБ
    dI3 = I_MEB_T - (I_MB_T + I_E_T)                   # МБ+Э -> МЭБ
    dI4 = (I_DFE_T + I_H_T) - (I_B_T + I_EB_T)         # Б+ЭБ -> ДФЭ+H2
    dI5 = I_DEB_T - (I_EB_T + I_E_T)                   # ЭБ+Э -> ДЭБ
    dI = np.array([dI1, dI2, dI3, dI4, dI5])

    # степени протекания реакций
    w1 = abs(C_EB_k)
    w2 = abs(C_PB_k)
    w3 = abs(C_MEB_k)
    w4 = abs(C_DFE_k)
    w5 = abs(C_DEB_k)
    w = np.array([w1, w2, w3, w4, w5])

    H_T = H298r + dI
    Qreact = np.sum(H_T * w)
    Qout = (
        C_B_k   * I_B_1   +
        C_E_k   * I_E_1   +
        C_EB_k  * I_EB_1  +
        C_P_k   * I_P_1   +
        C_PB_k  * I_PB_1  +
        C_MB_k  * I_MB_1  +
        C_MEB_k * I_MEB_1 +
        C_DFE_k * I_DFE_1 +
        C_DEB_k * I_DEB_1 +
        C_H_k   * I_H_1
    )

    Qbalance = (Qin + Qreact - Qout) / Qout * 100.0 if Qout != 0 else 0.0

    return Qin, Qreact, Qout, Qbalance


def calculate(p, c_b_0, c_e_0, c_p_0, c_mb_0, t_max, T, G_M):
    # описание реакций
    r_11 = Reaction(G=(-67.6 + R * T * np.log(p) - 105.87 + 0.13 * T - 4.84e-6 * T ** 2),
                    k=eyring_k(T, 57.5 + 0.03 * T))
    r_12 = Reaction(G=0, k=r_11.k / np.exp(-r_11.G / (R * T)))

    r_21 = Reaction(G=(R * T * np.log(p) - 54.9 - 98.9434 + 0.15 * T - 6.16e-6 * T ** 2),
                    k=eyring_k(T, 90 + 0.05 * T))
    r_22 = Reaction(G=0, k=r_21.k / np.exp(-r_21.G / (R * T)))

    r_31 = Reaction(G=(R * T * np.log(p) - 62.53 - 104.25 + 0.14667 * T - 4.666e-6 * T ** 2),
                    k=eyring_k(T, 65 + 0.02 * T))
    r_32 = Reaction(G=0, k=r_31.k / np.exp(-r_31.G / (R * T)))

    r_41 = Reaction(G=(R * T * np.log(p) + 37.395 + 29.407 + 0.03 * T - 1.2882e-6 * T ** 2),
                    k=eyring_k(T, 130 + 0.145 * T))
    r_42 = Reaction(G=0, k=r_41.k / np.exp(-r_41.G / (R * T)))

    r_51 = Reaction(G=(R * T * np.log(p) - 118.87 - 130.69 + 0.03667 * T - 8.3217e-6 * T ** 2),
                    k=eyring_k(T, 95 + 0.075 * T))
    r_52 = Reaction(G=0, k=r_51.k / np.exp(-r_51.G / (R * T)))

    reactions = (r_11, r_12, r_21, r_22, r_31, r_32, r_41, r_42, r_51, r_52)

    # начальные концентрации
    y0 = [c_b_0, c_e_0, 0.0, c_p_0, 0.0, c_mb_0, 0.0, 0.0, 0.0, 0.0]

    t_span = (0.0, t_max)
    t_eval = np.linspace(0.0, t_max, 1000)

    solution = solve_ivp(
        lambda t, y: dydt(t, y, reactions),
        t_span, y0, t_eval=t_eval, method="Radau", rtol=1e-9, atol=1e-9
    )

    # исходная общая масса
    start_mass = float(np.dot(molar_masses, y0))
    # конечная общая масса
    now_mass = np.dot(molar_masses, solution.y)
    # отклонение массы
    diff_mass = start_mass - now_mass


    # первое приближение теплоты продуктов
    T1 = T
    C_end = solution.y[:, -1]

    for _ in range(30):
        Qin, Qreact, Qout, Qbal = heat_balance_for_state(C_end, y0, T, T1)
        F = Qin + Qreact - Qout
        if abs(F) < 1e-6:
            break
        dT = 1.0
        _, _, Qout2, _ = heat_balance_for_state(C_end, y0, T, T1 + dT)
        F2 = Qin + Qreact - Qout2
        dF = (F2 - F) / dT
        if abs(dF) < 1e-8:
            break
        T1 = T1 - F / dF
        print(T1)

    # баланс при найденной T1
    Qin, Qreact, Qout, Qbalance = heat_balance_for_state(C_end, y0, T, T1)

    # процентная погрешность по времени
    Q_balance_time = []
    for k in range(solution.y.shape[1]):
        C_vec_k = solution.y[:, k]
        _, _, _, Qbal_k = heat_balance_for_state(C_vec_k, y0, T, T1)
        Q_balance_time.append(Qbal_k)

    Q_balance_time = np.array(Q_balance_time)

    Q_tuple = (Qin, Qreact, Qout, Qbalance, T1, Q_balance_time)
    return solution, start_mass, diff_mass, reactions, y0, Q_tuple


def optimal_T(p, c_b_0, c_e_0, c_p_0, c_mb_0, t_max, T_min=195.0, T_max=257.0, step=5):
    T_min += 273
    T_max += 273
    Ts = np.linspace(T_min, T_max, 100//step)
    C_EB_end = []

    for T in Ts:
        solution_T, start_mass_T, diff_mass_T, _, _, _ = calculate(
            p, c_b_0, c_e_0, c_p_0, c_mb_0, t_max, T, G_M = 1
        )
        C_EB_end.append(solution_T.y[2, -1])

    T_opt = Ts[np.argmax(C_EB_end)]

    return T_opt, Ts, C_EB_end
