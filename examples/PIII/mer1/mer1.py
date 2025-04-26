# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:37:32 2024

@author: balazs
"""

import numpy as np
import matplotlib.pyplot as plt
import praktikum as p
import pandas as pd
from math import sqrt

#%% data
data = np.array(pd.read_excel('mer1.xlsx'))

z_p = 10 #cm
s_zp = 0.1 #cm

y_0 = 9.7 #10 #mm
s_y0 = 0.1 #mm

z_o1 = data[2:14, 2].astype(np.float64) #cm
s_zo1 = 0.1 #cm

z_1 = data[2:14, 4].astype(np.float64) #cm
s_z1 = 0.1

z_2 = data[2:14, 6].astype(np.float64) #cm
s_z2 = 0.1 

y_1 = data[2:14, 10].astype(np.float64) #mm
s_y1 = 0.2 

y_2 = data[2:14, 12].astype(np.float64) #mm
s_y2 = 0.2

a_30 = 30 #cm #data[16:32].astype(np.float64) #
s_a30 = 0.2 #cm

a_60 = 60 #cm
s_a60 = 0.2

d_111 = data[17:34, 2].astype(np.float64) #mm
s_d111 = 0.1

d_112 = data[17:34, 3].astype(np.float64) #mm
s_d112 = 0.1 

z_o211 = data[17:34, 4].astype(np.float64) #mm
s_zo211 = 0.1

d_121 = data[34:49, 2].astype(np.float64) #mm
s_d121 = 0.1

d_122 = data[34:49, 3].astype(np.float64) #mm
s_d122 = 0.1 

z_o212 = data[34:49, 4].astype(np.float64) #mm
s_zo212 = 0.1

d_211 = data[51:66, 2].astype(np.float64) #mm
s_d211 = 0.1

d_212 = data[51:66, 3].astype(np.float64) #mm
s_d212 = 0.1 

z_o221 = data[51:66, 4].astype(np.float64) #mm
s_zo221 = 0.1

d_221 = data[66:81, 2].astype(np.float64) #mm
s_d221 = 0.1

d_222 = data[66:81, 3].astype(np.float64) #mm
s_d222 = 0.1 

z_o222 = data[66:81, 4].astype(np.float64) #mm
s_zo222 = 0.1

delta_1 = 1.5 #mm
s_delta1 = 2

delta_2 = 7 #mm
s_delta2 = 0.2

d = 38 #mm
s_d = 1

phi = 5 #1/m

#%% 1 Bessel

z_o = np.array([np.mean(z_o1[i:i+3]) for i in range(0, len(z_o1), 3)])
s_zo = np.array([np.sqrt(s_zo1**2+np.std(z_o1[i:i+3])**2) for i in range(0, len(z_o1), 3)])

z1 = z_1[:]
z_1 = np.array([np.mean(z1[i:i+3]) for i in range(0, len(z_o1), 3)])
s_z1 = np.array([np.sqrt(s_z1**2+np.std(z1[i:i+3])**2) for i in range(0, len(z_o1), 3)])

z2 = z_2[:]
z_2 = np.array([np.mean(z2[i:i+3]) for i in range(0, len(z_o1), 3)])
s_z2 = np.array([np.sqrt(s_z2**2+np.std(z2[i:i+3])**2) for i in range(0, len(z_o1), 3)])

D = z_o-z_p
s_D = s_zo+s_zp

Delta = np.abs(z_2-z_1)
s_Delta = s_z1+s_z2

def f_bessel_f(D, Delta):
    return (D**2-Delta**2)/(4*D)

f_bessel = np.mean(f_bessel_f(D, Delta))
s_fbessel = np.array(p.prenos_chyb_multi(f_bessel_f, [s_D, s_Delta], [D, Delta]))
s_fbessel = np.sqrt(np.sum(s_fbessel**2/len(D)**2))

print(p.round_to(f_bessel, s_fbessel))

#%% 1 dvoji zvets

y1 = y_1[:]
y_1 = np.array([np.mean(y1[i:i+3]) for i in range(0, len(z_o1), 3)])
s_y1 = np.array([np.sqrt(s_y1**2+np.std(y1[i:i+3])**2) for i in range(0, len(z_o1), 3)])

y2 = y_2[:]
y_2 = np.array([np.mean(y2[i:i+3]) for i in range(0, len(z_o1), 3)])
s_y2 = np.array([np.sqrt(s_y2**2+np.std(y2[i:i+3])**2) for i in range(0, len(z_o1), 3)])

def f_dvoj_f1(y_1, y_2, y, z_1, z_2):
    return (y_1/y)*(y_2/y)*np.abs(z_1-z_2)/np.abs((y_1/y)-(y_2/y))

def f_dvoj_f2(y_1, y_2, y, z_1, z_2):
    return np.abs(z_1-z_2)/np.abs((y_1/y)-(y_2/y))

def f_dvoj_f(y_1, y_2, y, Delta):
    return (y_1/y)*(y_2/y)*Delta/np.abs((y_1/y)-(y_2/y))

f_dvoj_1 = np.mean(f_dvoj_f1(y_1, y_2, y_0, z_1, z_2))
f_dvoj_2 = np.mean(f_dvoj_f2(y_1, y_2, y_0, z_1, z_2))
f_dvoj = np.mean(f_dvoj_f(y_1, y_2, y_0, Delta))

s_fdvoj1 = np.array(p.prenos_chyb_multi(f_dvoj_f1, [s_y1, s_y2, [s_y0]*len(y_1), s_z1, s_z2], [y_1, y_2, [y_0]*len(y_1), z_1, z_2]))
s_fdvoj2 = np.array(p.prenos_chyb_multi(f_dvoj_f2, [s_y1, s_y2, [s_y0]*len(y_1), s_z1, s_z2], [y_1, y_2, [y_0]*len(y_1), z_1, z_2]))
s_fdvoj = np.array(p.prenos_chyb_multi(f_dvoj_f, [s_y1, s_y2, [s_y0]*len(y_1), s_Delta], [y_1, y_2, [y_0]*len(y_1), Delta]))

s_fdvoj1 = np.sqrt(np.sum((s_fdvoj1**2)/len(s_fdvoj1)**2))
s_fdvoj2 = np.sqrt(np.sum((s_fdvoj2**2)/len(s_fdvoj2)**2))
s_fdvoj = np.sqrt(np.sum((s_fdvoj**2)/len(s_fdvoj)**2))

print(p.round_to(f_dvoj_1, s_fdvoj1))
print(p.round_to(f_dvoj_2, s_fdvoj2))
print(p.round_to(f_dvoj, s_fdvoj))

#%% 2 - 1

d_1 = sorted(list(set(d_111)))

z1 = z_o211[:]
z_o211 = np.array([np.mean(z1[d_111 == i]) for i in d_1])
s_zo211 = np.array([np.sqrt(s_zo211**2+np.std(z1[d_111 == i])**2) for i in d_1])

z1 = d_112[:]
d_112 = np.array([np.mean(z1[d_111 == i]) for i in d_1])
s_d112 = np.array([np.sqrt(s_d112**2+np.std(z1[d_111 == i])**2) for i in d_1])

z1 = d_111[:]
s_d111 = np.array([np.sqrt(s_d111**2+np.std(z1[d_111 == i])**2) for i in d_1])
d_111 = np.array([np.mean(z1[d_111 == i]) for i in d_1])

s_11 = (d_111+d_112)/4
s_s11 = (s_d111 + s_d112)/4

z1 = z_o212[:]
z_o212 = np.array([np.mean(z1[d_121 == i]) for i in d_1])
s_zo212 = np.array([np.sqrt(s_zo212**2+np.std(z1[d_121 == i])**2) for i in d_1])

z1 = d_122[:]
d_122 = np.array([np.mean(z1[d_121 == i]) for i in d_1])
s_d122 = np.array([np.sqrt(s_d122**2+np.std(z1[d_121 == i])**2) for i in d_1])

z1 = d_121[:]
s_d121 = np.array([np.sqrt(s_d121**2+np.std(z1[d_121 == i])**2) for i in d_1])
d_121 = np.array([np.mean(z1[d_121 == i]) for i in d_1])

s_12 = (d_121+d_122)/4
s_s12 = (s_d121 + s_d122)/4

delta_a11 = z_o211-z_o211[0]
delta_a12 = z_o212-z_o212[0]

def sqr(x, a):
    return a*x**2

p11, e11 = p.curve_fit(sqr, s_11, delta_a11, [s_s11, s_zo211+s_zo211[0]])
p12, e12 = p.curve_fit(sqr, s_12, delta_a12, [s_s12, s_zo212+s_zo212[0]])

#%% 2 - 2

z1 = z_o221[:]
z_o221 = np.array([np.mean(z1[d_211 == i]) for i in d_1])
s_zo221 = np.array([np.sqrt(s_zo221**2+np.std(z1[d_211 == i])**2) for i in d_1])

z1 = d_212[:]
d_212 = np.array([np.mean(z1[d_211 == i]) for i in d_1])
s_d212 = np.array([np.sqrt(s_d212**2+np.std(z1[d_211 == i])**2) for i in d_1])

z1 = d_211[:]
s_d211 = np.array([np.sqrt(s_d211**2+np.std(z1[d_211 == i])**2) for i in d_1])
d_211 = np.array([np.mean(z1[d_211 == i]) for i in d_1])

s_21 = (d_211+d_212)/4
s_s21 = (s_d211 + s_d212)/4

z1 = z_o222[:]
z_o222 = np.array([np.mean(z1[d_221 == i]) for i in d_1])
s_zo222 = np.array([np.sqrt(s_zo222**2+np.std(z1[d_221 == i])**2) for i in d_1])

z1 = d_222[:]
d_222 = np.array([np.mean(z1[d_221 == i]) for i in d_1])
s_d222 = np.array([np.sqrt(s_d222**2+np.std(z1[d_221 == i])**2) for i in d_1])

z1 = d_221[:]
s_d221 = np.array([np.sqrt(s_d221**2+np.std(z1[d_221 == i])**2) for i in d_1])
d_221 = np.array([np.mean(z1[d_221 == i]) for i in d_1])

s_22 = (d_221+d_222)/4
s_s22 = (s_d221 + s_d222)/4

delta_a21 = z_o221-z_o221[0]
delta_a22 = z_o222-z_o222[0]

def sqr(x, a):
    return a*x**2

p21, e21 = p.curve_fit(sqr, s_21, delta_a21, [s_s21, s_zo221+s_zo221[0]])
p22, e22 = p.curve_fit(sqr, s_22, delta_a22, [s_s22, s_zo222+s_zo222[0]])

#%%
def lin(x, a):
    return a*x

p.default_plot([s_11, s_12, s_21,  s_22], [delta_a11, delta_a12, delta_a21, delta_a22], 'r [mm]', '$\Delta$ a [cm]',
               legend=['a = 30 cm, pozice 1', 'a = 60 cm, pozice 1', 'a = 30 cm, pozice 2', 'a = 60 cm, pozice 2'],
               xerror=[s_s11, s_s12, s_s21, s_s22], 
               yerror=[s_zo211+s_zo211[0], s_zo212+s_zo212[0], s_zo221+s_zo221[0], s_zo222+s_zo222[0]], spline = 'spline', marker='default',
               fit = [[sqr, p11],  [sqr, p12], [sqr, p21],  [sqr, p22]])
p.default_plot([s_11**2, s_12**2, s_21**2,  s_22**2], [delta_a11, delta_a12, delta_a21, delta_a22], '$r^2$ [mm]', '$\Delta$ a [cm]',
               legend=['a = 30 cm, pozice 1', 'a = 60 cm, pozice 1', 'a = 30 cm, pozice 2', 'a = 60 cm, pozice 2'],
               xerror=[2*s_s11*s_11, 2*s_s12*s_12, 2*s_s21*s_21, 2*s_s22*s_22], 
               yerror=[s_zo211+s_zo211[0], s_zo212+s_zo212[0], s_zo221+s_zo221[0], s_zo222+s_zo222[0]], spline = 'spline', marker='default',
               fit = [[lin, p11],  [lin, p12], [lin, p21],  [lin, p22]])
#%% 3

def f_dvoj_f1(y_1, y_2, y, z_1, z_2):
    return (y_1/y)*(y_2/y)*(z_1-z_2)/((y_1/y)-(y_2/y))

def f_dvoj_f(y_1, y_2, y, Delta):
    return (y_1/y)*(y_2/y)*Delta/((y_1/y)-(y_2/y))

def f_dvoj_f2(y_1, y_2, y, z_1, z_2):
    return (z_1-z_2)/((y_1/y)-(y_2/y))

def n(delta, d):
    return d/(d-delta)

print(p.round_to(n(delta_2, d), p.prenos_chyb(n, [s_delta2, s_d], [delta_2, d])))

#%% vzorceky chyb

print(p.prenos_chyb_latex(f_bessel_f))

print(p.prenos_chyb_latex(f_dvoj_f1))

print(p.prenos_chyb_latex(f_dvoj_f2))

print(p.prenos_chyb_latex(n))

#%% tabulky

table1 = p.default_table(pd.DataFrame({
    'D [cm]': p.readable(D, s_D),
    '$\Delta$ [cm]': p.readable(Delta, s_Delta),
    "$y_1'$ [mm]": p.readable(y_1, s_y1),
    "$y_2'$ [mm]": p.readable(y_2, s_y2),
    }), 'table1', 'Namerané hodnoty D, $\Delta$ a veľkosti obrazov pre obidve polohy šošovky včetne ich chýb')


table2 = p.default_table(pd.DataFrame({
    '$r$ [mm]': p.readable(s_11, s_s11),
    '$\Delta a$ [cm]': p.readable(delta_a11, s_zo211+s_zo211[0]),
    '$\Delta a $ [cm]': p.readable(delta_a12, s_zo212+s_zo212[0]),
    '$\Delta a  $ [cm]': p.readable(delta_a21, s_zo221+s_zo221[0]),
    '$\Delta a   $ [cm]': p.readable(delta_a22, s_zo222+s_zo222[0])
    }), 'table2', 'Namerané hodnoty ',
    header=[(1, '   '), (1, '$a = (30\\pm0.2)$ cm'), (1, '$a = (60\\pm0.2)$ cm'), (1, '$a = (30\\pm0.2)$ cm'), (1, '$a = (60\\pm0.2)$ cm')])

#%%
def inv(x):
    return 1/x

print(p.round_to(1/5*100, p.prenos_chyb(inv, [0.1], [5])*100))