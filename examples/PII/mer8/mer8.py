# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:06:29 2023

@author: balaz
"""

import numpy as np
import matplotlib.pyplot as plt
import praktikum as p
import pandas as pd
from math import sqrt
import cv2
from scipy.ndimage import gaussian_filter

data = np.array(pd.read_excel('mer8.xlsx'))

C = data[1:, 3].astype(float)
s_C = C*0.01

U_s = data[1:, 4].astype(float)
s_Us = U_s*0.15/100+2*0.01

U_min = data[1:, 5].astype(float)
s_Umin = np.ones_like(U_min)*0.4/sqrt(3)

U_0 = data[1:, 6].astype(float)
s_U0 = np.ones_like(U_0)*0.4/sqrt(3)

R_z = data[1:13, 7].astype(float)
s_Rz = R_z*0.1/100

I = data[1:13, 8].astype(float)
s_I = I*1/100+3*0.01

spec = data[1:13, 9].astype(float)
U_smin = data[1:13, 10].astype(float)
s_Usmin = spec/5/sqrt(3)

U_smax = data[1:13, 11].astype(float)
s_Usmax = spec/5/sqrt(3)

#%% 1 a
table1 = p.default_table(pd.DataFrame({
    'C [$\mu F$]' : p.readable(C, s_C),
    '$U_s$ [V]' : p.readable(U_s, s_Us),
    '$U_{min}$ [V]' : p.readable(U_min, s_Umin),
    '$U_{max}$ [V]' : p.readable(U_0, s_U0)
    }), 'table1',
    'Namerané hodnoty jednosmerného napätia $U_s$, maximálného $U_{max}$ a minimálneho napätia $U_{min}$ v závislostti na kapacitu $C$')

def f(x, a, b, c):
    return a*(1-b/(x+c))
def const(x, a):
    return np.zeros_like(x) + a

parms1, _ = p.curve_fit(f, C, U_s, p0 = [8.8, 1, 1])
parms2, _ = p.curve_fit(f, C, U_min, p0 = [8.8, 1, 1])
parms3, _ = p.curve_fit(const, C, U_0, p0 = [8.8], imports={'np': np})

p.default_plot([C, C, C], [U_s, U_min, U_0], 'C [$\mu F$]', 'U [V]', legend=['$U_s$', '$U_{min}$', '$U_{max}$'], spline='spline', fit = [[f, *parms1],
                                                                                                                                          [f, *parms2],
                                                                                                                                          [const, 8.8]],
                xerror = [s_C, s_C, s_C], yerror = [s_Us, s_U0, s_U0])

#%% 1 b
table2 = p.default_table(pd.DataFrame({
    '$I$ [mA]' : p.readable(I, s_I),
    '$U_{min}$ [V]' : p.readable(U_smin, s_Usmin),
    '$U_{max}$ [V]' : p.readable(U_smax, s_Usmax)
    }), 'table2', 'namerané hodnoty napätia $U_{min}$ a $U_{max}$ v závislosti na prúdu $I$')

a, b = p.lin_fit(I, U_smax-U_smin)
a, s_a = a
b, s_b = b

p.default_plot(I, U_smax-U_smin, 'I [mA]', '$\Delta U$ [V]', xerror=s_I, yerror=s_Usmin*2, 
                fit = [[lambda x, a, b: x*a +b, a, b]])

#%%
def kf(U_max, U_min):
    return U_max/(U_max-U_min)

def lin(x, a, b):
    return a*x+b

s_kf = p.prenos_chyb_multi(kf, [s_U0, s_Umin], [U_0,  U_min])

parms, errs = p.curve_fit(lin, C[:-3], U_0[:-3]/(U_0[:-3]-U_min[:-3]), err = [s_C[:-3], s_kf[:-3]], p0 = [1, 0])
a,b = parms
s_a,s_b = errs

p.default_plot(C, U_0/(U_0-U_min), 'C [$\mu F$]', '$k_f$', xerror=[s_C], yerror=[s_kf], marker='with lines',
                fit=[[lin, a, b]])

#%%

def kf(U_0, U_max, U_min):
    return U_0/(U_max-U_min)

s_kf = p.prenos_chyb_multi(kf, [s_U0[:len(s_Usmax)], s_Usmax, s_Usmin], [U_0[:len(U_smax)], U_smax,  U_smin])

parms, _ = p.curve_fit(f, I, kf(U_0[:len(U_smax)], U_smax, U_smin), p0 = [10, 1, 0])

p.default_plot(I, kf(U_0[:len(U_smax)], U_smax, U_smin), 'I [mA]', '$k_f$', xerror=[s_I], yerror=[s_kf],
                fit = [[f, *parms]])