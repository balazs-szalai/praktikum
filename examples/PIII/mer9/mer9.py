# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:29:36 2024

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

data = pd.read_excel('mer9.xlsx')

#%% 1

h = np.array([40, 41])
d = np.array([18, 17])

table1 = p.default_table(pd.DataFrame({
    'h [cm]': p.readable(h, [1]*2),
    'd [cm]': p.readable(d, [1]*2)
    }), 'table1', 'caption')

def theta_B(h, d):
    return np.pi/2 - np.arctan(h/d)/2

s_h = np.sqrt(np.std(h)**2 + 1/2)
s_d = np.sqrt(np.std(d)**2 + 1/2)

h = np.mean(h)
d = np.mean(d)

thetaB = theta_B(h, d)
s_thetaB = p.prenos_chyb(theta_B, [s_h, s_d], [h, d])

#%% 2

I = np.array(data.iloc[1:, 0], float)
s_I = 0.01

P = np.array(data.iloc[1:, 2], float)
s_P = np.array(data.iloc[1:, 3], float)

s_P = np.sqrt(s_P**2 + 0.1**2)

table2 = p.default_table(pd.DataFrame({
    'I [mA]': p.readable(I, [s_I]*len(I)),
    'P [$\mu W$]': p.readable(P, s_P)
    }), 'table2', 'caption')

p.default_plot(I, P, 'I [mA]', 'P [$\mu W$]',
               xerror=[[s_I]*len(I)],
               yerror=[s_P], spline = None)

#%% 3

z = np.array(data.iloc[1:6, 8], float)
s_z = 1

x = np.array(data.iloc[1:6, 9], float)
s_x = np.array(data.iloc[1:6, 10], float)

y = np.array(data.iloc[1:6, 11], float)
s_y = np.array(data.iloc[1:6, 12], float)

table3 = p.default_table(pd.DataFrame({
    'z [cm]': p.readable(z, [s_z]*len(z)),
    'x [$\mu m$]': p.readable(x, s_x),
    'y [$\mu m$]': p.readable(y, s_y)
    }), 'table3', 'caption')

ax, b = p.lin_fit(z, x, err = [[s_z]*len(z), s_x])
ay, b = p.lin_fit(z, y, err = [[s_z]*len(z), s_y])

ax, s_ax = ax
ay, s_ay = ay

#%% 4

d = np.array(data.iloc[5:8, 14], float)
s_d = 1

x_1 = np.array(data.iloc[5:8, 15], float)
x_2 = np.array(data.iloc[5:8, 16], float)

s_x = 1

h = 1000

def lamb(d, x, h):
    return h * np.sin(np.arctan(x/d))

table4 = p.default_table(pd.DataFrame({
    'd [cm]': p.readable(d, [s_d]*len(d)),
    '$x_1$ [cm]': p.readable(x_1, [s_x]*len(x_1)),
    '$x_2$ [cm]': p.readable(x_2, [s_x]*len(x_2))
    }), 'table4', 'caption')

lmb = lamb(np.append(d, d), np.append(x_1, x_2), h)
s_lmb = np.array(p.prenos_chyb_multi(lamb, [[s_d]*2*len(d), [s_x]*2*len(x_1), [0]*len(d)*2], [np.append(d, d), np.append(x_1, x_2), [h]*len(d)*2]))

s_lmb = np.sqrt(np.std(lmb)**2 + np.sum(s_lmb**2)/len(lmb)**2)
lmb = np.mean(lmb)
#%% 5

# spektr = np.array(pd.read_csv('temp/spektrometr1.csv', skiprows=32, delimiter='\t', ))

spektr = np.loadtxt('temp/spektrometr1.csv', skiprows=33, dtype = np.object_)
spektr[:, 0] = [i[1:] for i in spektr[:, 0]]
spektr[:, 1] = [i[:-1] for i in spektr[:, 1]]
spektr = spektr.astype(np.float64)

p.default_plot(spektr[:, 0], spektr[:, 1], '$\lambda$ [nm]', 'I [-]', marker='lines')

#%% 6
table5 = p.default_table(data.iloc[10:15, 5:7], 'table5', 'caption')
