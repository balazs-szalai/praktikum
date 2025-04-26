# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:24:59 2024

@author: balazs
"""

import numpy as np
import matplotlib.pyplot as plt
import praktikum as p
import pandas as pd
from math import sqrt

k = 2*np.pi/632.8e-9
n = 22

def I(x, n, a, b, A, B, C):
    return A*(np.sin(a*k*np.sin(x-C)/2)/(a*k*np.sin(x-C)/2))**2*np.sin(n*b*k*np.sin(x-C)/2)**2/np.sin(b*k*np.sin(x-C)/2)**2 + B

data = np.loadtxt('temp/mr1', np.float64, skiprows=1)

# #%% 1 mr

# n = np.arange(-10, 11)
# x = np.array([1, 13, 26, 39, 51, 64, 76, 89, 101, 114, 126, 138, 151, 163, 176, 188, 200, 213, 226, 239, 251])
# s_x = 1

# plt.plot(n, x)

# a, b = p.lin_fit(n, x, [[0]*len(n), [s_x]*len(x)])
# a, s_a = a
# b, s_b = b

# plt.plot(n, a*n + b)

# #%%

# def b(L, lambd, x_1):
#     return L*lambd/x_1

# print(p.round_to(b(1000, 632.8e-6, a), p.prenos_chyb(b, [3, 0, s_a], [1000, 632.8e-6, a])))
# #%%

# table1 = p.default_table(pd.DataFrame({
#     'n': np.arange(-10, 11),
#     'x [mm]': p.readable(x, [s_x]*len(x))
#     }), 'table1', 'nci')

#%%
L = 1000
s_L = 3
x = data[:, 0]
y = data[:, 1]

def f(a, b):
    return np.arctan(a/b)

phi = f(x, L)
s_phi = p.prenos_chyb_multi(f, [[0.003]*len(x), [3]*len(x)], [x, [L]*len(x)])

parms1, errs1 = p.curve_fit(I, phi, y, err = [s_phi, [1/256]*len(y)], p0 = [22, 1e-5, 5.2e-5, 0.4, 8, 3.4e-5], global_vars={'k': k, 'n': n}, imports={'np': np}, ignor_exception=True, n = 300)

#%%
# parms1 = np.array([0]+list(parms1))
# errs1 = np.array([0]+list(errs1))
table2 = p.default_table(pd.DataFrame({
    'n [-]': [p.round_to(parms1[0], errs1[0])],
    'a [$\mu m$]': [p.round_to(parms1[1]*10**6, errs1[1]*10**6)],
    'b [$\mu m$]': [p.round_to(parms1[2]*10**6, errs1[2]*10**6)],
    'A': [p.round_to(parms1[3], errs1[3])],
    'B': [p.round_to(parms1[4], errs1[4])],
    'C$\times 10^{6}$': [p.round_to(parms1[5]*10**6, errs1[5]*10**6)],
    }), 'table2', 'caption')

#%%
p.default_plot([phi], [y], '$\\varphi$ [rad]', 'I', legend = ['mriezka'], fit  = [[I, *parms1]], marker='-')

#%%

def I(x, b, A, B, C):
    return A*(np.sin(b*k*(x-C)/2)/(b*k*(x-C)/2))**2 + B

data = np.loadtxt('temp/st1', np.float64, skiprows=1)

L = 1000
s_L = 3
x = data[:, 0]
y = data[:, 1]

def f(a, b):
    return np.arctan(a/b)

phi = np.arctan(x/L)
s_phi = np.array(p.prenos_chyb_multi(f, [[0.003]*len(x), [3]*len(x)], [x, [L]*len(x)]))
#%%
parms2, errs2 = p.curve_fit(I, phi[np.abs(x) > 0.005], y[np.abs(x) > 0.005], err = [s_phi[np.abs(x) > 0.005], [1/256]*len(y[np.abs(x) > 0.005])], p0 = [0.0002, 250, 5, 0], global_vars={'k': k}, imports={'np': np}, ignor_exception=True, n=1000)

#%%

table3 = p.default_table(pd.DataFrame({
    'b [$\mu m$]': [p.round_to(parms2[0]*10**6, errs2[0]*10**6)],
    'A': [p.round_to(parms2[1], errs2[1])],
    'B': [p.round_to(parms2[2], errs2[2])],
    'C$\times 10^{6}$': [p.round_to(parms2[3]*10**6, errs2[3]*10**6)],
    }), 'table3', 'caption')

#%%
p.default_plot([phi], [y], '$\\varphi$ [rad]', 'I', legend = ['štrbina'], fit  = [[I, *parms2]], marker='-')

#%%

def I(x, a, b, A, B, C, D):
    return A*(np.sin(b*k*(x-C)/2)/(b*k*(x-C)/2))**2*(np.cos(a*k*(x-C)/2)**2+D) + B

print(p.format_to_latex(I))

data = np.loadtxt('temp/dst1', np.float64, skiprows=1)

L = 1000
s_L = 3
x = data[:, 0]
y = data[:, 1]

def f(a, b):
    return np.arctan(a/b)
#%%
phi = np.arctan(x/L)
s_phi = np.array(p.prenos_chyb_multi(f, [[0.003]*len(x), [3]*len(x)], [x, [L]*len(x)]))
#%%
parms3, errs3 = p.curve_fit(I, phi[np.abs(x) > 0.005], y[np.abs(x) > 0.005], err = [s_phi[np.abs(x) > 0.005], [1/256]*len(y[np.abs(x) > 0.005])], p0 = [0.0003, 0.0002, 250, 5, 0.00027, 0.1], global_vars={'k': k}, imports={'np': np}, ignor_exception=True, n = 1000)
#%%

table4 = p.default_table(pd.DataFrame({
    'a [$\mu m$]': [p.round_to(parms3[0]*10**6, errs3[0]*10**6)],
    'b [$\mu m$]': [p.round_to(parms3[1]*10**6, errs3[1]*10**6)],
    'A': [p.round_to(parms3[2], errs3[2])],
    'B': [p.round_to(parms3[3], errs3[3])],
    'C$\times 10^{3}$': [p.round_to(parms3[4]*10**3, errs3[4]*10**3)],
    'D$\\times$': [p.round_to(parms3[5], errs3[5])]
    }), 'table4', 'caption')

#%%
p.default_plot([phi], [y], '$\\varphi$ [rad]', 'I', legend = ['dvojštrbina'], fit  = [[I, *parms3]], marker='-')

#%%

scale = [-0.18, 0.43, 1.05, 1.66, 2.26, 2.88, 3.47, 4.07, 4.7, 5.3, 5.97, 6.56, 7.19, 7.75, 8.35]
s_sc = 0.05

a, b = p.lin_fit(np.arange(len(scale)), scale, err = [[0]*len(scale), [0.05]*len(scale)])
a, s_a = a

def inv(x):
    return 1/x
#%%
st = inv(a*10)*1000
s_st = p.prenos_chyb(inv, [s_a*10], [a*10])*1000
#%%

table5 = p.default_table(pd.DataFrame({
    'x [dielka]': p.readable(scale, [s_sc]*len(scale))
    }), 'table5', 'caption')
#%%
mr = [0.03, 0.37, 0.67, 0.99, 1.31, 1.62, 1.95, 2.24, 2.55, 2.86, 3.18, 3.48, 3.79, 4.11, 4.43, 4.74, 5.0, 5.31, 5.65, 5.92, 6.28, 6.59, 6.92, 7.23, 7.53, 7.84, 8.15, 8.47]
s_mr = 0.05

a, b = p.lin_fit(np.arange(len(mr)), mr, err = [[0]*len(mr), [0.05]*len(mr)])
a, s_a = a
#%%
def mul(a, b):
    return a*b


b = mul(a, st)
s_b = p.prenos_chyb(mul, [s_a, s_st], [a, st])
#%%
table6 = p.default_table(pd.DataFrame({
    'x [dielka stupnici mikroskopu]': p.readable(mr, [s_mr]*len(mr))
    }), 'table6', 'caption')
#%%
x1 = np.array([3.88, 4.7, 3.81, 3.81])
x2 = np.array([5.19, 5.97, 5.15, 5.11])

x = x2-x1
s_x = np.sqrt(np.std(x)**2 + 0.1**2/len(x))
x = np.mean(x)

s_x = p.prenos_chyb(mul, [s_x, s_st], [x, st])
x = mul(x, st)
#%%
table7 = p.default_table(pd.DataFrame({
    '$x_1$ [dielik stupnice mikorskopu]': p.readable(x1, [0.05]*len(x1)),
    '$x_2$ [dielik stupnice mikorskopu]': p.readable(x2, [0.05]*len(x2)),
    }), 'table7', 'caption')
#%%

x1 = [1.42, 1.13, 0.17]
x2 = [2.76, 2.37, 1.41]
x3 = [5.13, 4.72, 3.88]
x4 = [6.4, 6.01, 5.02]

x1, x2, x3, x4 = np.array(x1), np.array(x2), np.array(x3), np.array(x4)

d1 = x2-x1
d2 = x4-x3
d3 = np.mean(np.array([x4, x3]), axis = 0) - np.mean(np.array([x1, x2]), axis = 0)

s_d1 = np.sqrt(np.std(d1)**2 + 0.1**2/len(d1))
s_d2 = np.sqrt(np.std(d2)**2 + 0.1**2/len(d2))
s_d3 = np.sqrt(np.std(d3)**2 + 0.1**2/len(d3))

d = np.mean([np.mean(d1), np.mean(d2)])
s_d = np.sqrt(np.std(d2)**2 + np.mean([s_d1, s_d2])**2)

d3 = np.mean(d3)

s_d = p.prenos_chyb(mul, [s_d, s_st], [d, st])
d = mul(d, st)

s_d3 = p.prenos_chyb(mul, [s_d3, s_st], [d3, st])
d3 = mul(d3, st)

#%%

table8 = p.default_table(pd.DataFrame({
    '$x_1$ [dielik stupnice mikorskopu]': p.readable(x1, [0.05]*len(x1)),
    '$x_2$ [dielik stupnice mikorskopu]': p.readable(x2, [0.05]*len(x2)),
    '$x_3$ [dielik stupnice mikorskopu]': p.readable(x3, [0.05]*len(x3)),
    '$x_4$ [dielik stupnice mikorskopu]': p.readable(x4, [0.05]*len(x4)),    
    }), 'table8', 'caption')