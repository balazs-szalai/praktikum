# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 22:39:52 2024

@author: balazs
"""

import numpy as np
import matplotlib.pyplot as plt
import praktikum as p
import pandas as pd
from math import sqrt
import cv2
from scipy.optimize import curve_fit
#%%
x11 = [0, 149, 298, 439]
# x12 = [13, 319, 622]
x13 = [14, 125, 257, 386, 496, 622]
s_x1 = 10

x21 = [259, 407, 560, 700]
# x22 = [232, 552, 842]
x23 = [244, 367, 489, 609, 727, 848]
s_x2 = 10

table1 = p.default_table(pd.DataFrame({
    '$y_{vrypy}$':  p.pad(p.readable(x11, [s_x1]*len(x11)), 6),
    '$y_{okolie}$': p.pad(p.readable(x21, [s_x2]*len(x21)), 6),
    # '$y_{vrypy}$ ':  p.pad(p.readable(x12, [s_x1]*len(x12)), 6),
    # '$y_{okolie}$ ': p.pad(p.readable(x22, [s_x2]*len(x22)), 6),
    '$y_{vrypy}$  ':  p.pad(p.readable(x13, [s_x1]*len(x13)), 6),
    '$y_{okolie}$  ': p.pad(p.readable(x23, [s_x2]*len(x23)), 6)
    }), 'table1', 'pozdeji', header=[(2, '1. miesto'), (2, '2. miesto')])

# plt.plot(x_scale, x1)
# plt.plot(x_scale, x2)

x_scale = np.arange(len(x21))
a11, b11 = p.lin_fit(x_scale, x21, err = [[0]*len(x21), [s_x1]*len(x21)])

x_11 = np.mean(np.abs(np.array(x21)-np.array(x11)))
s_x11 = s_x1*np.sqrt(2/len(x11))
x_21, s_x21 = a11
# s_x21 = np.sqrt(s_x21**2 + s_x11**2/2)

# x_scale = np.arange(len(x12))
# a12, b12 = p.lin_fit(x_scale, x22, err = [[0]*len(x22), [s_x1]*len(x22)])

# x_12 = np.mean(np.abs(np.array(x22)-np.array(x12)))
# s_x12 = s_x1*np.sqrt(2/len(x12))
# x_22, s_x22 = a12
# s_x22 = np.sqrt(s_x22**2 + s_x12**2/2)

x_scale = np.arange(len(x13))
a13, b13 = p.lin_fit(x_scale, x23, err = [[0]*len(x23), [s_x1]*len(x23)])

x_13 = np.mean(np.abs(np.array(x23)-np.array(x13)))
s_x13 = s_x1*np.sqrt(2/len(x13))
x_23, s_x23 = a13
# s_x23 = np.sqrt(s_x23**2 + s_x13**2/2)

table2 = p.default_table(pd.DataFrame({
    ' ': ['1. miesto', '2. miesto'],
    '$x_1$': p.readable([x_11, x_13], [s_x11, s_x13]),
    '$x_2$': p.readable([x_21, x_23], [s_x21, s_x23])
    }), 'table2', 'nic')
#%%
lamb = 589.3e-9
s_lamb = 0#0.3e-9
def t(x_1, x_2, lamb):
    return x_1/x_2*lamb/2

print(p.prenos_chyb_latex(t))

t1 = t(x_11, x_21, lamb)
s_t1 = p.prenos_chyb(t, [s_x11, s_x21, s_lamb], [x_11, x_21, lamb])

t3 = t(x_13, x_23, lamb)
s_t3 = p.prenos_chyb(t, [s_x13, s_x23, s_lamb], [x_13, x_23, lamb])

#%% getting the scale
img = cv2.imread('temp_img\\scale.jpg')
img = np.sum(img, axis = 2)
scale = img[440, :]

scale_fft = np.fft.rfft(scale)
freq = np.fft.rfftfreq(1280)

#%%
pow_spec_dens = np.abs(scale_fft)**2 
f1, f2 = freq[np.argsort(pow_spec_dens[1:])[-2:]+1]

ppm = 1/f2*50_000

def ppm0(f):
    return 50_000/f 

print(p.round_to(ppm, p.prenos_chyb(ppm0, [1/1280], [f2])))

#%%
def get_x(x, ppm):
    return x/ppm

def get_data(file, center):
    img = cv2.imread(file)
    img = img[:, :, 1]
    
    data = img[center[0],:]
    x = (np.arange(img.shape[1])-center[1]).astype(np.float64)
    x /= ppm
    s_x = p.prenos_chyb_multi(get_x, [[1]*len(x), [p.prenos_chyb(ppm0, [1/1280], [f2])]*len(x)], [x, [ppm]*len(x)])
    
    return x, s_x, data

def get_vert_data(file, center):
    n = 20
    img = cv2.imread(file)
    img = img[:, :, 1]
    data = np.sum([img[:, center[1]-n//2+i] for i in range(n)], axis=0)/n
    return data

lamb = 589.3*1e-9
def I(x, A0, A1, A2, d, R, B0, B1, B2, c):
    return (A0 + A1*x + A2*x**2)*np.sin(2*np.pi/lamb*(d+(x-c)**2/(2*R)))**2 + (B0 + B1*x + B2*x**2) 

def fit_center(data, R0, center):
    x = np.arange(1024)
    R0 *= ppm**2
    B0 = np.mean(data)
    B1 = 0
    B2 = 0
    A0 = np.max(data)-np.min(data)
    A1 = 0 
    A2 = 0
    d0 = np.arcsin(np.sqrt(np.abs((data[np.argsort(np.abs(x))[0]] - B0)/A0)))*lamb/(2*np.pi)
    c = center[0]
    
    parms, _ = curve_fit(I, x, data, p0 = [A0, A1, A2, d0, R0, B0, B1, B2, c])
    return parms, _

# parms1, errs1 = fit(data1, 0.17)
# parms2, errs2 = fit(data2, 0.03)
# parms3, errs3 = fit(data3, 0.06)
# parms4, errs4 = fit(data4, 0.32)

center1 = [565, 620]
file1 = 'temp_img\\F50_1.jpg'
data1 = get_vert_data(file1, center1)

center2 = [500, 680]
file2 = 'temp_img\\F50_2.jpg'
data2 = get_vert_data(file2, center2)

center3 = [540, 670]
file3 = 'temp_img\\F100_1.jpg'
data3 = get_vert_data(file3, center3)

center4 = [575, 650]
file4 = 'temp_img\\F100_2.jpg'
data4 = get_vert_data(file4, center4)

parms1, errs1 = fit_center(data1, 0.17, center1)
parms2, errs2 = fit_center(data2, 0.03, center2)
parms3, errs3 = fit_center(data3, 0.06, center3)
parms4, errs4 = fit_center(data4, 0.32, center4)

#%%

center1 = [int(parms1[-1]), 635]
file1 = 'temp_img\\F50_1.jpg'
x1, s_x1, data1 = get_data(file1, center1)

center2 = [int(parms1[-1]), 680]
file2 = 'temp_img\\F50_2.jpg'
x2, s_x2, data2 = get_data(file2, center2)

center3 = [int(parms1[-1]), 670]
file3 = 'temp_img\\F100_1.jpg'
x3, s_x3, data3 = get_data(file3, center3)


center4 = [int(parms1[-1]), 650]
file4 = 'temp_img\\F100_2.jpg'
x4, s_x4, data4 = get_data(file4, center4)


#%%
# A0 = np.max(data)-np.min(data)-10
# def I(A0, x, d, R, B0):
#     return (A0)*np.sin(2*np.pi/lamb*(d+x**2/(2*R)))**2 + B0 
#%%
def fit(data, x, s_x, R0):
    B0 = np.mean(data)
    B1 = 0
    B2 = 0
    A0 = np.max(data)-np.min(data)
    A1 = 0 
    A2 = 0
    d0 = np.arcsin(np.sqrt(np.abs((data[np.argsort(np.abs(x))[0]] - B0)/A0)))*lamb/(2*np.pi)
    c = 0
    
    parms, errs = p.curve_fit(I, x, data, p0 = [A0, A1, A2, d0, R0, B0, B1, B2, c], err = [s_x, [1/255]*len(data)], imports={'np': np}, global_vars={'lamb': lamb}, ignor_exception=True)
    return parms, errs

parms1, errs1 = fit(data1, x1, s_x1, 0.17)
parms2, errs2 = fit(data2, x2, s_x2, 0.03)
parms3, errs3 = fit(data3, x3, s_x3, 0.06)
parms4, errs4 = fit(data4, x4, s_x4, 0.32)

#%%
p.default_plot([x1], [data1], 'x [m]', 'I [-]',
                legend = ['F 50 - 1. strana'],
                fit=[[I, *parms1]], marker='lines')
p.default_plot([x2], [data2], 'x [m]', 'I [-]',
                legend = ['F 50 - 2. strana'],
                fit=[[I, *parms2]], marker='lines')
p.default_plot([x3], [data3], 'x [m]', 'I [-]',
                legend = ['F 100 - 1. strana'],
                fit=[[I, *parms3]], marker='lines')
p.default_plot([x4], [data4], 'x [m]', 'I [-]',
                legend = ['F 100 - 2. strana'],
                fit=[[I, *parms4]], marker='lines')

#%%
parms = np.array([parms1, parms2, parms3, parms4])
errs = np.array([errs1, errs2, errs3, errs4])

table3 = p.default_table(pd.DataFrame({
    ' ': ['F 50 - 1. strana', 'F 50 - 2. strana', 'F 100 - 1. strana', 'F 100 - 2. strana'],
    '$A_0$': p.readable(parms[:, 0], errs[:, 0]),
    '$A_1\\times 10^{-2}$': p.readable(parms[:, 1]/100, errs[:, 1]/100),
    '$A_2\\times 10^{-6}$': p.readable(parms[:, 2]/10**6, errs[:, 2]/10**6),
    '$d \\times 10^{8}$': p.readable(parms[:, 3]*10**8, errs[:, 3]*10**8),
    '$R$': p.readable(parms[:, 4], errs[:, 4]),
    '$B_0$': p.readable(parms[:, 5], errs[:, 5]),
    '$B_1\\times 10^{-2}$': p.readable(parms[:, 6]/100, errs[:, 6]/100),
    '$B_2\\times 10^{-6}$': p.readable(parms[:, 7]/10**6, errs[:, 7]/10**6),
    }), 'table3', 'nic')

#%%
n = 1.51680
def f(R_1, R_2, n):
    return 1/((1/R_1-1/R_2)*(1-n))

def f_t(R_1, R_2, n, t):
    return (n*R_1*R_2)/((n-1)*(t*(n-1) + n*(R_2+R_1)))

t1 = 6.5 
t2 = 4.0 

R_11 = parms1[4]*1000
s_R11 = errs1[4]*1000

R_12 = parms2[4]*1000
s_R12 = errs2[4]*1000

R_21 = parms3[4]*1000
s_R21 = errs3[4]*1000

R_22 = parms4[4]*1000
s_R22 = errs4[4]*1000

f_1 = f(R_11, -R_12, n)
s_f1 = p.prenos_chyb_eval(f, [s_R11, s_R12, 0], [R_11, R_12, n])

f_2 = f(R_21, -R_22, n)
s_f2 = p.prenos_chyb_eval(f, [s_R21, s_R22, 0], [R_21, R_22, n])

f_t1 = f_t(R_11, R_12, n, t1)
s_ft1 = p.prenos_chyb_eval(f, [s_R11, s_R12, 0], [R_11, R_12, n])

f_t2 = f_t(R_21, R_22, n, t2)
s_ft2 = p.prenos_chyb_eval(f, [s_R21, s_R22, 0], [R_21, R_22, n])

print(p.format_to_latex(f_t))

