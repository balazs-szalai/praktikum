# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:03:25 2024

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

data = pd.read_excel('mer7.xlsx')
lamb = 632.8e-6

#%% 1 - geom

l = np.array(data.iloc[1:4, 0]).astype(np.float64)
d = np.array(data.iloc[1:4, 1]).astype(np.float64)

table1 = p.default_table(pd.DataFrame({
    'l [cm]': p.readable(l, [1]*len(l)),
    'd [mm]': p.readable(d, [3]*len(d))
    }), 'table1', 'neco')

s_l = np.sqrt(np.std(l)**2 + 1/len(l))
l = np.mean(l)

s_d = np.sqrt(np.std(d)**2 + 3/len(d))/10
d = np.mean(d)/10

def vartheta(d, l):
    return 2*np.arctan(d/(2*l))

# print(p.prenos_chyb_latex(vartheta))

theta = 2*np.arctan(d/(2*l))*180/np.pi
s_theta = p.prenos_chyb(vartheta, [s_d, s_l], [d, l])*180/np.pi
d_F_geom = lamb/(2*np.sin(theta/2))

#%% 1 - interf

phi = np.array(data.iloc[6:11, 0]).astype(np.float64)
N_F = np.array(data.iloc[6:11, 1]).astype(np.float64)

def vartheta(lamb, N_F, Phi):
    return 2*np.arcsin((lamb*(N_F-1))/(2*Phi))

thet = np.array(vartheta(lamb, N_F, phi))*180/np.pi
s_thet = np.array(p.prenos_chyb_multi(vartheta, [[0]*len(N_F), [0]*len(N_F), [0.02]*len(N_F)], [[lamb]*len(N_F), N_F, phi]))*180/np.pi

table2 = p.default_table(pd.DataFrame({
    '$\Phi$ [mm]': p.readable(phi, [0.02]*len(N_F)),
    '$N_F$ [-]': N_F.astype(int),
    '$\vartheta$ [$\degree$]': p.readable(thet, s_thet)
    }), 'table2', 'naco')

s_thet = np.sqrt(np.std(thet)**2 + np.sum(s_thet**2)/len(thet)**2)*np.pi/180
thet = np.mean(thet)*np.pi/180

d_F_int = phi/(N_F-1)

d_F = np.mean(d_F_int)

def d_F(theta):
    return lamb/(2*np.sin(theta/2))

d_F_geom = d_F(theta*np.pi/180)
s_d_F_geom = p.prenos_chyb(d_F, [s_theta*np.pi/180], [theta*np.pi/180])

d_F_proj = d_F(thet)
s_d_F_proj = p.prenos_chyb(d_F, [s_thet], [thet])

#%% 3

t = np.array(data.iloc[1:, 4]).astype(np.float64)
N = np.array(data.iloc[1:, 5]).astype(np.float64)

def v_x(n, lamb, l, thet):
    return (n*lamb)/(2*l*np.sin((thet/2)))

v = v_x(N, lamb, t, thet)
s_v = p.prenos_chyb_multi(v_x, [[0]*len(N), [0]*len(N), [0.1]*len(t), [s_thet]*len(N)], 
                          [N, [lamb]*len(N), t, [thet]*len(N)])

table3 = p.default_table(pd.DataFrame({
    'l [ms]': p.readable(t, [0.1]*len(t)),
    'N [-]': N.astype(int),
    '$v_x$ [$\\frac{cm}{s}$]': p.readable(np.array(v)*100, np.array(s_v)*100)
    }), 'table3', 'caption')

delta_fD = N/t

vx = d_F*delta_fD

#%% hist

hist, bins, fig = plt.hist(v*100, 10, alpha = 0.8)
centers = np.convolve(bins, np.array([0.5, 0.5]), mode = 'valid')

def f(x, a, b, c):
    return a*np.exp(-(x-b)**2/(2*c**2))

parms, errs = p.curve_fit(f, centers, hist,
                          imports={'np': np})

xs = np.linspace(bins[0], bins[-1], 1000)

plt.plot(xs, f(xs, *parms), color = 'red', linewidth = 3)

plt.xlabel('$v$ [$\\frac{cm}{s}$]')
plt.ylabel('n (počet častíc)')

a, b, c = parms
# plt.plot([b-c, b-c], [0, a], color = 'red')
# plt.plot([b+c, b+c], [0, a], color = 'red')


#%%



# print(p.prenos_chyb_latex(vartheta, parms = ['lambda_0', 'N_F', 'Phi']))



# def f(x, l):
#     return (1)/(2*l*np.sin(x))

# print(p.prenos_chyb_latex(v_x, parms=['n', 'lambda_0', 'l', 'vartheta']))



# print(p.format_to_latex(f, parms = ["x", "A", "B", "C"]))

# print(p.format_to_latex(v_x))#, parms=['N', 'lambda_0', 'l', 'varthteta']))

# def f(x):
#     return np.asin(x)

# print(p.format_to_latex(f))
# print(p.prenos_chyb_latex(f))