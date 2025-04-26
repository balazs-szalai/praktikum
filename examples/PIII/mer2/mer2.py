# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:06:32 2024

@author: balazs
"""

import numpy as np
import matplotlib.pyplot as plt
import praktikum as p
import pandas as pd
from funcs import n, lamb, n2

#%% data
data = np.array(pd.read_excel('mer2.xlsx'))

t = 24
s_t = np.sqrt(0.5**2+np.std([23, 25])**2)

#%% 1 - disperze meriaceho hranolu

n_m = n(lamb, t)
s_nm = p.prenos_chyb_multi(n, [[0]*len(lamb), [s_t]*len(lamb)], [lamb, [t]*len(lamb)])
s_nm = [np.sqrt(1e-10 + i**2) for i in s_nm]

table1 = p.default_table(pd.DataFrame({
    'spektrálne čiary': ['C', 'd', 'e', 'F', 'g',  'h'],
    '$N_1$': p.readable(n_m, s_nm),
    }), 'table1', 'pozdeji')

def fit(lamb, n0, a, lamb0):
    return n0 + a/(lamb+lamb0)

p1, e1 = p.curve_fit(fit, lamb, n_m, p0 = [1.6918, 17.6, -227], err = [[0]*len(lamb), s_nm])

p.default_plot(lamb, n_m, '$\lambda$ [nm]', '$N_1$',
               xerror=[0]*len(lamb), yerror=s_nm, spline = 'spline', legend = ['n($\lambda$)'], fit = [[fit, *p1]])

#%% 2 - disperze meraneho skla

lamb1 = [690.7, 579.1, 577.0, 546.1, 435.8, 407.8]
gamma1 = [32.785833333333336,
         31.374166666666667,
         31.321666666666665,
         30.516666666666666,
         27.463333333333335,
         25.928333333333335]#deg
gamma1 = [i*np.pi/180 for i in gamma1]
s_gamma = 2.908882086657216e-05

def n1(n, g):
    return np.sqrt(n**2 - np.cos(g)**2)
print(p.prenos_chyb_latex(n1))
#%%
n_m = fit(lamb1, *p1)
s_nm = p.prenos_chyb_multi(fit, [[0]*len(lamb1), [e1[0]]*len(lamb1), [e1[1]]*len(lamb1), [e1[2]]*len(lamb1)], [lamb1, [p1[0]]*len(lamb1), [p1[1]]*len(lamb1), [p1[2]]*len(lamb1)])

N1 = n1(n_m, gamma1)
s_N1 = p.prenos_chyb_multi(n1, [s_nm, [s_gamma]*len(lamb1)], [n_m, gamma1])

p2, e2 = p.curve_fit(fit, lamb1, N1, p0 = [1.6918, 17.6, -227], err = [[0]*len(lamb1), s_N1])

s_N1 = p.prenos_chyb_multi(fit, [[0]*len(lamb1), [e2[0]]*len(lamb1), [e2[1]]*len(lamb1), [e2[2]]*len(lamb1)], [lamb1, [p2[0]]*len(lamb1), [p2[1]]*len(lamb1), [p2[2]]*len(lamb1)])
s_N1 = [float(i) for i in s_N1]
#%%
p.default_plot(np.array(lamb1), np.array(N1), '$\lambda$ [nm]', '$N_2$',
               xerror=[0]*len(lamb), yerror=np.array(s_N1), legend = ['n($\lambda$)'], fit = [[fit, *p2]])

table2 = p.default_table(pd.DataFrame({
    '$\lambda$ [nm]': p.readable(list(lamb1), [0]*len(lamb1)),
    '$\gamma$ [°]': p.readable(gamma1, [s_gamma]*len(gamma1)),
    'N_2': p.readable(list(N1), [float(i) for i in s_N1])
    }), 'table2', 'pozdeji')

#%% stredni disp., abbe
def frac(a, b):
    return a/b

#C       d        e       F       g      h
#656.3   587.6    546.1   486.1   435.8  404.7

deltaN_FC = fit(486.1, *p2) - fit(656.3, *p2)
s_deltaN_FC = p.prenos_chyb(fit, [0, *e2], [486.1, *p2]) + p.prenos_chyb(fit, [0, *e2], [656.3, *p2])

deltaN_Fd = fit(486.1, *p2) - fit(587.6, *p2)
s_deltaN_Fd = p.prenos_chyb(fit, [0, *e2], [486.1, *p2]) + p.prenos_chyb(fit, [0, *e2], [587.6, *p2])

deltaN_Fe = fit(486.1, *p2) - N1[3]
s_deltaN_Fe = p.prenos_chyb(fit, [0, *e2], [486.1, *p2]) + s_N1[3]

deltaN_gF = N1[-2] - fit(486.1, *p2)
s_deltaN_gF = s_N1[-2] + p.prenos_chyb(fit, [0, *e2], [486.1, *p2])

deltaN_dC = fit(587.6, *p2) - fit(656.3, *p2)
s_deltaN_dC = p.prenos_chyb(fit, [0, *e2], [587.6, *p2]) + p.prenos_chyb(fit, [0, *e2], [656.3, *p2])

V_d = (fit(587.6, *p2) - 1)/deltaN_FC
s_Vd = p.prenos_chyb(frac, [p.prenos_chyb(fit, [0, *e2], [587.6, *p2]), s_deltaN_FC], [(fit(587.6, *p2) - 1), deltaN_FC])

n_d = fit(587.6, *p2)
s_nd = p.prenos_chyb(fit, [0, *e2], [587.6, *p2])

delta_Fd_FC = deltaN_Fd/deltaN_FC
s_deltaFd_FC = p.prenos_chyb(frac, [s_deltaN_Fd, s_deltaN_FC], [deltaN_Fd, deltaN_FC]) 

#%% 3 - n(t)

ts = np.array([18.0, 24.0, 28.0, 32.0, 36.0, 40.0, 43.0, 46.0, 49.0, 52.0])

gamma_y = np.array( [41.59,
                     41.571666666666665,
                     41.413333333333334,
                     41.1875,
                     40.94916666666667,
                     40.65416666666667,
                     40.3775,
                     40.13333333333333,
                     39.87916666666667,
                     39.61416666666667])*np.pi/180

gamma_g = np.array(  [40.84583333333333,
                     40.788333333333334,
                     40.60583333333334,
                     40.36666666666667,
                     40.100833333333334,
                     39.821666666666665,
                     39.53916666666667,
                     39.270833333333336,
                     39.01,
                     38.73416666666667])*np.pi/180

gamma_b = np.array(  [36.1575,
                     36.04416666666667,
                     35.78333333333333,
                     35.48916666666667,
                     35.181666666666665,
                     34.821666666666665,
                     34.5025,
                     34.178333333333335,
                     33.84166666666667,
                     33.48416666666667])*np.pi/180

n_y = np.array([n2(579.1, t) for t in ts])
s_ny = p.prenos_chyb_multi(n, [[0]*len(ts), [2.0]*len(ts)], [[579.1]*len(ts), ts])
s_ny = [np.sqrt(1e-10 + i**2) for i in s_ny]

n_g = np.array([n2(546.1, t) for t in ts])
s_ng = p.prenos_chyb_multi(n, [[0]*len(ts), [2.0]*len(ts)], [[546.1]*len(ts), ts])
s_ng = [np.sqrt(1e-10 + i**2) for i in s_ng]

n_b = np.array([n2(435.8, t) for t in ts])
s_nb = p.prenos_chyb_multi(n, [[0]*len(ts), [2.0]*len(ts)], [[435.8]*len(ts), ts])
s_nb = [np.sqrt(1e-10 + i**2) for i in s_nb]

def N2(N1, gamma):
    return np.sqrt(N1**2 - np.cos(gamma)*np.sqrt(N1**2 - np.cos(gamma)**2))

N2_y = N2(n_y, gamma_y)
s_N2y = p.prenos_chyb_multi(N2, [s_ny, [s_gamma]*len(ts)], [n_y, gamma_y])

N2_g = N2(n_g, gamma_g)
s_N2g = p.prenos_chyb_multi(N2, [s_ng, [s_gamma]*len(ts)], [n_g, gamma_g])

N2_b = N2(n_b, gamma_b)
s_N2b = p.prenos_chyb_multi(N2, [s_nb, [s_gamma]*len(ts)], [n_b, gamma_b])

def quad(x, a, b, c):
    return a*x**2 + b*x + c

abc_y, s_abcy = p.curve_fit(quad, ts, N2_y, err=[[2.0]*len(ts), s_N2y])
abc_g, s_abcg = p.curve_fit(quad, ts, N2_g, err=[[2.0]*len(ts), s_N2g])
abc_b, s_abcb = p.curve_fit(quad, ts, N2_b, err=[[2.0]*len(ts), s_N2b])

p.default_plot([ts]*3, [N2_y, N2_g, N2_b], 't [°C]', 'n', 
               legend = ['$\lambda$ = 579.1 nm', '$\lambda$ = 546.1 nm', '$\lambda$ = 435.8 nm'],
               xerror=[[2]*len(ts)]*3, yerror=[s_N2y, s_N2g, s_N2b], spline='spline', fit = [[quad, *abc_y], [quad, *abc_g], [quad, *abc_b]])

table3 = p.default_table(pd.DataFrame({
    't [°C]': p.readable(ts, [2.0]*len(ts)),
    '$\gamma$ [rad]': p.readable(gamma_y, [s_gamma]*len(ts)),
    '$N_1$': p.readable(n_y, s_ny),
    '$N_2$': p.readable(N2_y, s_N2y),
    '$\gamma$ [rad] ': p.readable(gamma_y, [s_gamma]*len(ts)),
    '$N_1$ ' : p.readable(n_y, s_ny),
    '$N_2$ ' : p.readable(N2_y, s_N2y),
    '$\gamma$ [rad]  ': p.readable(gamma_y, [s_gamma]*len(ts)),
    '$N_1$  ': p.readable(n_y, s_ny),
    '$N_2$  ': p.readable(N2_y, s_N2y),
    }), 'table3', 'pozdeji', header=[(1, ' '), (3, '$\lambda=579.1$nm'), (3, '$\lambda=546.1$nm'), (3, '$\lambda=435.8$nm')])


table4 = p.default_table(pd.DataFrame({
    'vlnové dĺžky': ['$\lambda$ = 579.1 nm', '$\lambda$ = 546.1 nm', '$\lambda$ = 435.8 nm'],
    'a $\times 10^{6}$': p.readable([abc_y[0]*10**6, abc_g[0]*10**6, abc_b[0]*10**6], [s_abcy[0]*10**6, s_abcg[0]*10**6, s_abcb[0]*10**6]),
    'b $\times 10^{4}$': p.readable([abc_y[1]*10**4, abc_g[1]*10**4, abc_b[1]*10**4], [s_abcy[1]*10**4, s_abcg[1]*10**4, s_abcb[1]*10**4]),
    'c': p.readable([abc_y[2], abc_g[2], abc_b[2]], [s_abcy[2], s_abcg[2], s_abcb[2]])
    }), 'table4', "fitovacie parametre ku kvadratickému fitu teplotnej závislosti indexu lomu zadanej kvapaliny")

#%%

def Abbe(n_d, n_F, n_C):
    return (n_d-1)/(n_F-n_C)

print(p.prenos_chyb_latex(Abbe))

def delta(n_d, n_F, n_C):
    return (n_F-n_C)/(n_d-1)

print(p.prenos_chyb_latex(delta))
#%%
def n_2(n_1, g):
    return np.sqrt(n_1**2 - np.cos(g)*np.sqrt(n_1**2 - np.cos(g)**2))

print(p.prenos_chyb_latex(n_2))