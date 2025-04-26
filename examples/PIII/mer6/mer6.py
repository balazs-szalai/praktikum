# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:10:24 2024

@author: balazs
"""

import numpy as np
import scipy.special as spec
import cv2
from math import sin
import praktikum as p
import matplotlib.pyplot as plt
import scipy.optimize as opt
from math import sqrt, cos, sin
import pandas as pd

data = pd.read_excel('mer6.xlsx')

alpha = np.arange(0, 51, 5)

I1s = np.array(data.iloc[2:13, 4:9]).astype(np.float64)
I2s = np.array(data.iloc[16:27, 4:9]).astype(np.float64)

I1p1 = np.array(data.iloc[2:13, 12:15]).astype(np.float64)
I2p1 = np.array(data.iloc[16:27, 12:15]).astype(np.float64)

alpha1p2 = np.arange(25, 36)
I1p2 = np.array(data.iloc[2:13, 21:26]).astype(np.float64)[:, 2:]

alpha2p2 = np.arange(28, 41)
I2p2 = np.array(data.iloc[16:, 21:25]).astype(np.float64)


def r_s(theta_i, n1, n2):
    return (n1 * np.cos(theta_i) - n2 * np.sqrt(1 - (n1 / n2) ** 2 * np.sin(theta_i) ** 2)) / (n1 * np.cos(theta_i) + n2 * np.sqrt(1 - (n1 / n2) ** 2 * np.sin(theta_i) ** 2))

def r_p(theta_i, n1, n2):
    return (n1 * np.sqrt(1 - (n1 / n2) ** 2 * np.sin(theta_i) ** 2) - n2 * np.cos(theta_i)) / (n1 * np.sqrt(1 - (n1 / n2) ** 2 * np.sin(theta_i) ** 2) + n2 * np.cos(theta_i))

def t_s(theta_i, n1, n2):
    return 2 * n1 * np.cos(theta_i) / (n1 * np.cos(theta_i) + n2 * np.sqrt(1 - (n1 / n2) ** 2 * np.sin(theta_i) ** 2))

def t_p(theta_i, n1, n2):
    return 2 * n1 * np.cos(theta_i) / (n1 * np.sqrt(1 - (n1 / n2) ** 2 * np.sin(theta_i) ** 2) + n2 * np.cos(theta_i))

def fit_Rp(x, A, B, n):
    theta = x*np.pi/180
    return A*r_p(theta, 1, n)**2 + B

def fit_Rs(x, A, B, n):
    theta = x*np.pi/180
    return A*r_s(theta, 1, n)**2 + B

#%%

I_1s = np.mean(I1s, axis = 1)
s_I1s = np.sqrt(1/len(I1s[:, 0]) + np.std(I1s, axis = 1)**2)
pa_1s, _ = opt.curve_fit(fit_Rs, 90-alpha, I_1s, p0 = [1000, 10, 1.5])

I_2s = np.mean(I2s, axis = 1)
s_I2s = np.sqrt(1/len(I2s[:, 0]) + np.std(I2s, axis = 1)**2)
pa_2s, _ = opt.curve_fit(fit_Rs, 90-alpha, I_2s, p0 = [1000, 10, 1.5])

I_1p1 = np.mean(I1p1, axis = 1)
s_I1p1 = np.sqrt(1/len(I1p1[:, 0]) + np.std(I1p1, axis = 1)**2)
pa_1p1, _ = opt.curve_fit(fit_Rp, 90-alpha, I_1p1, p0 = [1000, 10, 1.5])

I_2p1 = np.mean(I2p1, axis = 1)
s_I2p1 = np.sqrt(1/len(I2p1[:, 0]) + np.std(I2p1, axis = 1)**2)
pa_2p1, _ = opt.curve_fit(fit_Rp, 90-alpha, I_2p1, p0 = [1000, 10, 1.5])

p.default_plot([90-alpha, 90-alpha], [I_1s, I_1p1], '$\\alpha_1$ [°]', 'I [-]',
               legend = ['1. vzorek - polarizácia s', '1. vzorek - polarizácia p'],
               xerror = [[0]*len(alpha)]*2,
               yerror = [s_I1s, s_I1p1],
               fit = [[fit_Rs, *pa_1s], [fit_Rp, *pa_1p1]])

p.default_plot([90-alpha, 90-alpha], [I_2s, I_2p1], '$\\alpha_1$ [°]', 'I [-]',
               legend = ['2. vzorek - polarizácia s', '2. vzorek - polarizácia p'],
               xerror = [[0]*len(alpha)]*2,
               yerror = [s_I2s, s_I2p1],
               fit = [[fit_Rs, *pa_2s], [fit_Rp, *pa_2p1]])

# p.default_plot([90-alpha, 90-alpha, 90-alpha, 90-alpha], 
#                [I_1s, I_1p1, I_2s, I_2p1], '$\\alpha_1$ [°]', 'I [-]',
#                legend = ['1. vzorek - polarizácia s', '1. vzorek - polarizácia p', '2. vzorek - polarizácia s', '2. vzorek - polarizácia p'],
#                xerror = [[0]*len(alpha)]*4,
#                yerror = [s_I1s, s_I1p1, s_I2s, s_I2p1])

table1 = p.default_table(pd.DataFrame({
    '$\\alpha_1$ [°]': p.readable(90-alpha, [0.02]*len(alpha)),
    '$I_s$': p.readable(I_1s, s_I1s),
    '$I_p$': p.readable(I_1p1, s_I1p1),
    '$I_s$ ': p.readable(I_2s, s_I2s),
    '$I_p$ ': p.readable(I_2p1, s_I2p1),
    }), 'table1', 'neco', header=[(1, ""), (2, '1. vzorek'), (2, '2. vzorek')])

I_1p2 = np.mean(I1p2, axis = 1)
s_I1p2 = np.sqrt(1/len(I1p2[:, 0]) + np.std(I1p2, axis = 1)**2)
pa_1p2, _ = opt.curve_fit(fit_Rp, 90-alpha1p2, I_1p2, p0 = [2000, 10, 1.5])

I_2p2 = np.mean(I2p2, axis = 1)
s_I2p2 = np.sqrt(1/len(I2p2[:, 0]) + np.std(I2p2, axis = 1)**2)
pa_2p2, _ = opt.curve_fit(fit_Rp, 90-alpha2p2, I_2p2, p0 = [1000, 10, 1.5])

table2 = p.default_table(pd.DataFrame({
    '$\\alpha_1$ [°]': p.pad(p.readable(90-alpha1p2, [0.02]*len(alpha1p2)), len(I_2p2)),
    '$I_p$': p.pad(p.readable(I_1p2, s_I1p2), len(I_2p2)),
    '$\\alpha_1$ [°] ': p.pad(p.readable(90-alpha2p2, [0.02]*len(alpha2p2)), len(I_2p2)),
    '$I_p$ ': p.readable(I_2p2, s_I2p2),
    }), 'table2', 'neco', header=[(2, '1. vzorek'), (2, '2. vzorek')])

p.default_plot([90-alpha1p2, 90-alpha2p2], [I_1p2, I_2p2], '$\\alpha_1$ [°]', 'I [-]',
               legend = ['1. vzorek', '2. vzorek'],
               xerror = [[0.02]*len(alpha1p2), [0.02]*len(alpha2p2)],
               yerror = [s_I1p2, s_I2p2],
               fit = [[fit_Rp, *pa_1p2], [fit_Rp, *pa_2p2]])
#%%

def n_Br(theta_Br):
    return np.tan(theta_Br)

print(p.prenos_chyb_latex(n_Br))
#%%

pa_1s, er_1s = p.curve_fit(fit_Rs, 90-alpha, I_1s, p0 = [1000, 10, 1.5], 
                           err = [[0.02]*len(alpha), s_I1s],
                           imports = {'np': np},
                           global_functions = [r_s])
pa_2s, er_2s = p.curve_fit(fit_Rs, 90-alpha, I_2s, p0 = [1000, 10, 1.5], 
                           err = [[0.02]*len(alpha), s_I2s],
                           imports = {'np': np},
                           global_functions = [r_s])
pa_1p1, er_1p1 = p.curve_fit(fit_Rp, 90-alpha, I_1p1, p0 = [1000, 10, 1.5], 
                           err = [[0.02]*len(alpha), s_I1p1],
                           imports = {'np': np},
                           global_functions = [r_p])
pa_2p1, er_2p1 = p.curve_fit(fit_Rp, 90-alpha, I_2p1, p0 = [1000, 10, 1.5], 
                           err = [[0.02]*len(alpha), s_I2p1],
                           imports = {'np': np},
                           global_functions = [r_p])
pa_1p2, er_1p2 = p.curve_fit(fit_Rp, 90-alpha1p2, I_1p2, p0 = [2000, 10, 1.5], 
                           err = [[0.02]*len(alpha1p2), s_I1p2],
                           imports = {'np': np},
                           global_functions = [r_p])
pa_2p2, er_2p2 = p.curve_fit(fit_Rp, 90-alpha2p2, I_2p2, p0 = [1000, 10, 1.5], 
                           err = [[0.02]*len(alpha2p2), s_I2p2],
                           imports = {'np': np},
                           global_functions = [r_p])

#%%
pa = np.array([pa_1s, pa_1p1, pa_2s, pa_2p1, pa_1p2, pa_2p2])
er = np.array([er_1s, er_1p1, er_2s, er_2p1, er_1p2, er_2p2])

table3 = p.default_table(pd.DataFrame({
    ' ': [  '1. vzorek $s$ polarizácia', '1. vzorek $p$ polarizácia',
            '2. vzorek $s$ polarizácia', '2. vzorek $p$ polarizácia',
            '1. vzorek $p$ polarizácia detailne okolo Brewsterova uhlu', 
            '2. vzorek $p$ polarizácia detailne okolo Brewsterova uhlu' ],
    'A [-]': p.readable([*pa[:, 0]], [*er[:, 0]]),
    'B [-]': p.readable([*pa[:, 1]], [*er[:, 1]]),
    '$n_2$': p.readable([*pa[:, 2]], [*er[:, 2]])
    }), 'table3', 'nieco')
#%%

n = 1.8051
def fit(x, A, beta):
    theta = x*np.pi/180
    b = beta*np.pi/180
    return A*((np.cos(b)*r_p(theta, 1, n))**2 + (np.sin(b)*r_s(theta, 1, n))**2 )

n = 1.8051
pa_1s, er_1s = p.curve_fit(fit, 90-alpha, I_1s, p0 = [1000, 89], 
                           err = [[0.02]*len(alpha), s_I1s],
                           imports = {'np': np},
                           global_vars = {'n': n},
                           global_functions = [r_p, r_s], ignor_exception=True)
n = 1.516
pa_2s, er_2s = p.curve_fit(fit, 90-alpha, I_2s, p0 = [1000, 89], 
                           err = [[0.02]*len(alpha), s_I2s],
                           imports = {'np': np},
                           global_vars = {'n': n},
                           global_functions = [r_p, r_s], ignor_exception=True)
n = 1.8051
pa_1p1, er_1p1 = p.curve_fit(fit, 90-alpha, I_1p1, p0 = [1000, 1], 
                           err = [[0.02]*len(alpha), s_I1p1],
                           imports = {'np': np},
                           global_vars = {'n': n},
                           global_functions = [r_p, r_s], ignor_exception=True)
n = 1.516
pa_2p1, er_2p1 = p.curve_fit(fit, 90-alpha, I_2p1, p0 = [1000, 0.1], 
                           err = [[0.02]*len(alpha), s_I2p1],
                           imports = {'np': np},
                           global_vars = {'n': n},
                           global_functions = [r_p, r_s], ignor_exception=True)
n = 1.8051
pa_1p2, er_1p2 = p.curve_fit(fit, 90-alpha1p2, I_1p2, p0 = [2000, 0.1], 
                           err = [[0.02]*len(alpha1p2), s_I1p2],
                           imports = {'np': np},
                           global_vars = {'n': n},
                           global_functions = [r_p, r_s], ignor_exception=True)
n = 1.516
pa_2p2, er_2p2 = p.curve_fit(fit, 90-alpha2p2, I_2p2, p0 = [1000, 0.1], 
                           err = [[0.02]*len(alpha2p2), s_I2p2],
                           imports = {'np': np},
                           global_vars = {'n': n},
                           global_functions = [r_p, r_s], ignor_exception=True)



#%%
# parms_1p2 = []
# errs_1p2 = []

# for i in range(len(I1p2[0, :])):
#     pa, er = p.curve_fit(fit_Rp, 90-alpha1p2, I1p2[:, i], p0 = [1000, 10, 1.5],
#                             err=[[0]*len(alpha), [2]*len(alpha1p2)],#0.1*I1p2[:, i]],
#                             imports={'np': np}, global_functions=[r_p])
#     parms_1p2.append(pa)
#     errs_1p2.append(er)

# parms_1p2 = np.array(parms_1p2)
# errs_1p2 = np.array(errs_1p2)


# parms1p2 = np.mean(parms_1p2, axis=0)
# errs1p2 = np.sqrt(np.sum(errs_1p2**2, axis = 0)/errs_1p2.shape[0]**2 + np.std(errs_1p2, axis=0)**2)

#%% 1s
# parms_1s = []
# errs_1s = []

# for i in range(len(I1s[0, :])):
#     pa, er = p.curve_fit(fit_Rs, 90-alpha, I1s[:, i], p0 = [1000, 10, 1.5],
#                          err=[[0]*len(alpha), 0.02*I1s[:, i]],
#                          imports={'np': np}, global_functions=[r_s])
#     parms_1s.append(pa)
#     errs_1s.append(er)
# parms_1s = np.array(parms_1s)
# errs_1s = np.array(errs_1s)


# parms1s = np.mean(parms_1s, axis=0)
# errs1s = np.sqrt(np.sum(errs_1s**2, axis = 0)/errs_1s.shape[0]**2 + np.std(errs_1s, axis=0)**2)

# #%% 2s

# parms_2s = []
# errs_2s = []

# for i in range(len(I1s[0, :])):
#     pa, er = p.curve_fit(fit_Rs, 90-alpha, I2s[:, i], p0 = [1000, 10, 1.5],
#                          err=[[0]*len(alpha), 0.02*I2s[:, i]],
#                          imports={'np': np}, global_functions=[r_s])
#     parms_2s.append(pa)
#     errs_2s.append(er)
# parms_2s = np.array(parms_2s)
# errs_2s = np.array(errs_2s)


# parms2s = np.mean(parms_2s, axis=0)
# errs2s = np.sqrt(np.sum(errs_2s**2, axis = 0)/errs_2s.shape[0]**2 + np.std(errs_2s, axis=0)**2)


# #%% 1p1

# parms_1p1 = []
# errs_1p1 = []

# for i in range(len(I1p1[0, :])):
#     pa, er = p.curve_fit(fit_Rp, 90-alpha, I1p1[:, i], p0 = [1000, 10, 1.5],
#                          err=[[0]*len(alpha), 0.02*I1p1[:, i]],
#                          imports={'np': np}, global_functions=[r_p])
#     parms_1p1.append(pa)
#     errs_1p1.append(er)
# parms_1p1 = np.array(parms_1p1)
# errs_1p1 = np.array(errs_1p1)


# parms1p1 = np.mean(parms_1p1, axis=0)
# errs1p1 = np.sqrt(np.sum(errs_1p1**2, axis = 0)/errs_1p1.shape[0]**2 + np.std(errs_1p1, axis=0)**2)

# #%% 2p1

# parms_2p1 = []
# errs_2p1 = []

# for i in range(len(I2p1[0, :])):
#     pa, er = p.curve_fit(fit_Rp, 90-alpha, I2p1[:, i], p0 = [1000, 10, 1.5],
#                          err=[[0]*len(alpha), 0.02*I2p1[:, i]],
#                          imports={'np': np}, global_functions=[r_p])
#     parms_2p1.append(pa)
#     errs_2p1.append(er)
# parms_2p1 = np.array(parms_2p1)
# errs_2p1 = np.array(errs_2p1)


# parms2p1 = np.mean(parms_2p1, axis=0)
# errs2p1 = np.sqrt(np.sum(errs_2p1**2, axis = 0)/errs_2p1.shape[0]**2 + np.std(errs_2p1, axis=0)**2)