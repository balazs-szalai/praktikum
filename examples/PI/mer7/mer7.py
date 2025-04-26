# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 17:22:41 2023

@author: Balázs local
"""

import numpy as np
import scipy.optimize as opt
import pandas as pd
from praktikum.format_latex import format_to_latex, prenos_chyb_latex
from praktikum.main_funcs import prenos_chyb, round_to, default_plot, reset_counter, rand_plot_colors
import matplotlib.pyplot as plt

def T(l, g):
    return 2*np.pi*np.sqrt(l/g)
#print(format_to_latex(T))

def T_f(J, m, g, d, alpha):
    return 2*np.pi*np.sqrt(J/(m*g*d))*(1+1/4*(np.sin(alpha/2))**2)
#print(format_to_latex(T_f), "\n")

def g(l, T):
    return 4*np.pi**2*l/T**2
#print(format_to_latex(g), "\n")
#print(prenos_chyb_latex(g), "\n")

def g_c(l, T, alpha):
    return 4*np.pi**2*l*(1+1/4*(np.sin(alpha/2))**2)**2/T**2
#print(prenos_chyb_latex(g_c), "\n")

def J(J_0, M, a):
    return J_0 + M*a**2
#print(format_to_latex(J), "\n")

def J_gul(M, r):
    return 2/5*M*r**2
#print(format_to_latex(J_gul), "\n")

def J_kyv(M, r, l):
    return 2/5*M*r**2+M*l**2

def J_add(M, l):
    return M*l**2

def div(a, b):
    return a/b

def J_tyc(M, r):
    return 1/3*M*r**2
#print(format_to_latex(J_tyc), "\n")

def g_fyz(l, T, alpha, r):
    return 4*np.pi**2*(1+1/4*(np.sin(alpha/2))**2)**2*(2/5*r**2/l+l)/T**2
#print(format_to_latex(g_fyz))
print(prenos_chyb_latex(g_fyz))

def lin(x, a, b):
    return a*x+b

#%%
plt.close('all')
reset_counter()

data = pd.read_excel('mer7.xlsx', 'List5')
m = 62, 1 #g
l_m = 103.5/100, 0.01 #m
D = 99.5/100, 0.001/np.sqrt(3) #m
r = 1.3/100, 0.1/100 #m

T20_m = np.array(data.iloc[3:8,2], dtype = np.float64)
x_l = np.array(data.iloc[3:10, 5], dtype = np.float64)
T20_d = np.array(data.iloc[3:10, 6], dtype = np.float64)
T20_u = np.array(data.iloc[3:10, 7], dtype = np.float64)
T20 = np.array(data.iloc[13:24,5][pd.notna(data.iloc[13:24, 5])], dtype = np.float64)

sigma_20T = 0.001
sigma_T = sigma_20T/20

T_m = np.mean(T20_m/20), np.sqrt((np.std(T20_m)**2+sigma_20T**2)/len(T20_m))/20
T_r = np.mean(T20/20), np.sqrt((np.std(T20)**2+sigma_20T**2)/len(T20))/20
print(round_to(*T_m), "\n")
print(round_to(*T_r), "\n")

#%%
parms_d, _ = opt.curve_fit(lin, x_l, T20_d)
parms_u, _ = opt.curve_fit(lin, x_l, T20_u)

colors = rand_plot_colors(2)
default_plot([x_l, x_l], [T20_d, T20_u], 'relatívna poloha čočky [cm]', "20 perióda", legend=['čočka dole', "čočka hore"], fit = [[lin, *parms_d],[lin, *parms_u]], colors=colors, save=True)
#%%

g_m = g(l_m[0], T_m[0]), prenos_chyb(g, np.diag([l_m[1]**2, T_m[1]**2]), [l_m[0], T_m[0]])
g_r = g(D[0], T_r[0]), prenos_chyb(g, np.diag([D[1]**2, T_r[1]**2]), [D[0], T_r[0]])
g_rc = g_c(D[0], T_r[0], np.pi/180*3), prenos_chyb(g_c, np.diag([D[1]**2, T_r[1]**2, (np.pi/180*3)**2]), [D[0], T_r[0], np.pi/180*3])
g_mc = g_fyz(l_m[0], T_m[0], np.pi/180*3, r[0]), prenos_chyb(g_fyz, np.diag([l_m[1]**2, T_m[1]**2, (np.pi/180*3)**2, r[1]**2]), [l_m[0], T_m[0], np.pi/180*3, r[0]])
print(round_to(*g_m), "\n")
print(round_to(*g_r), "\n")
print(round_to(*g_rc), "\n")
print(round_to(*g_mc), "\n")
#%%

J_m = J_kyv(m[0], r[0], l_m[0]), prenos_chyb(J_kyv, np.diag([m[1], r[1]**2, l_m[1]**2]), (m[0], r[0], l_m[0]))
print(round_to(*J_m), "\n")
J_a = J_add(m[0], l_m[0]), prenos_chyb(J_add, np.diag([m[1], l_m[1]**2]), (m[0], l_m[0]))
rel_j = div(J_m[0], J_a[0]), prenos_chyb(div, np.diag([J_m[1]**2, J_a[1]**2]), (J_m[0], J_a[0]))
print(round_to(*rel_j), "\n")