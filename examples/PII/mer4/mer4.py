# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:32:16 2023

@author: balaz
"""

import numpy as np
import pandas as pd
import praktikum as p

data = pd.read_excel('mer4.xlsx', skiprows=1, skipfooter=20)

def sigma(U_34, I_12, l, t, d):
    return l/(t*d)*I_12/U_34

I_12, U_34 = np.array(data.iloc[:,:2]).T/1000 #A, V

s_I12 = I_12*0.01+3*0.01/1000
s_U34 = np.ones(U_34.shape)*0.1/1000

l = 7.1/1000
s_l = 0.005/1000

d = 2.71/1000
s_d = 0.005/1000

t = 1.950/1000
s_t = 0.005/1000
#%%
table1 = p.default_table(pd.DataFrame({'$I_{12}$ [mA]': p.readable(I_12, s_I12),
                       '$U_{34}$ [mV]': p.readable(U_34, s_U34)}), 'table1', 'namerané hodnoty prúdu a napätia na vzroku v nulovom magnetickom poli')
#%%
def lin(x, a):
    return a*x

a, s_a = p.curve_fit(lin, U_34, I_12, [s_U34, s_I12])
a, s_a = a[0], s_a[0]
#%%
def sig(k, l,d,t):
    return k*l/(t*d)
sigm = sig(a, l, d, t)
s_sigm = p.prenos_chyb(sig, [s_a, s_l, s_d, s_t], [a, l, d, t])
print(p.round_to(sig(a, l, d, t), p.prenos_chyb(sig, [s_a, s_l, s_d, s_t], [a, l, d, t])))
#%%
fig, ax = p.default_plot([U_34], [I_12], '$U_{34}$ [mV]', '$I_{12}$ [mA]', fit = [[lin, a]], colors = p.rand_plot_colors(1), xerror = [s_U34], yerror = [s_I12])#
#%%
I_1 = np.array(data.iloc[:-2, 3])
Up_1 = np.array(data.iloc[:-2, 4])
Un_1 = np.array(data.iloc[:-2, 6])

s_I = np.ones(I_1.shape)*0.5*6/100/np.sqrt(3)
s_U = np.ones(Up_1.shape)*0.1

I_3 = np.array(data.iloc[:-2, 10])
Up_3 = np.array(data.iloc[:-2, 11])
Un_3 = np.array(data.iloc[:-2, 13])

I_5 = np.array(data.iloc[:-2, 17])
Up_5 = np.array(data.iloc[:-2, 18])
Un_5 = np.array(data.iloc[:-2, 20])

#%%
table2 = pd.DataFrame({'$I$ [A]': p.readable(I_1, s_I),
                       '$U^+_{56}$ [mV]': p.readable(Up_1, s_U),
                       '$U^-_{56}$ [mV]': p.readable(Un_1, s_U),
                       '$I $ [A]': p.readable(I_3, s_I),
                       '$U^+_{56} $ [mV]': p.readable(Up_3, s_U),
                       '$U^-_{56} $ [mV]': p.readable(Un_3, s_U),
                       '$I  $ [A]': p.readable(I_5, s_I),
                       '$U^+_{56}  $ [mV]': p.readable(Up_5, s_U),
                       '$U^-_{56}  $ [mV]': p.readable(Un_5, s_U)})
table2 = p.default_table(table2, 'table2', 'namerané hodnoty prúdu cez elektromegnetu a napätia $U_{56}^+$ a $U_{56}^-$', [(3, '$I_{12} = 1$ mA'), (3, '$I_{12} = 5$ mA'), (3, '$I_{12} = 5$ mA')])
#%%
def times(a, b):
    return a*b
B = I_1*0.098
s_B = p.prenos_chyb_multi(times, [[s_I[i], 0.001] for i in range(len(s_I))], [[I_1[i], 0.098] for i in range(len(s_I))], 1)

U_H1 = (Un_1-Up_1)/2/1000
U_H3 = (Un_3-Up_3)/2/1000
U_H5 = (Un_5-Up_5)/2/1000
s_U = np.ones(Up_1.shape)*0.1/1000

a1, s_a1 = p.curve_fit(lin, B, U_H1, [s_B, s_U])
a1, s_a1 = a1[0], s_a1[0]

a3, s_a3 = p.curve_fit(lin, B, U_H3, [s_B, s_U])
a3, s_a3 = a3[0], s_a3[0]

a5, s_a5 = p.curve_fit(lin, B, U_H5, [s_B, s_U])
a5, s_a5 = a5[0], s_a5[0]
#%%
def R_H(k, t, I_12):
    return k*t/I_12

R_H1 = R_H(a1, t, 0.001)
s_RH1 = p.prenos_chyb(R_H, [s_a1, s_t, 0.001*0.01+3*0.01/1000], [a1, t, 0.001])

R_H3 = R_H(a3, t, 0.003)
s_RH3 = p.prenos_chyb(R_H, [s_a3, s_t, 0.003*0.01+3*0.01/1000], [a3, t, 0.003])

R_H5 = R_H(a5, t, 0.005)
s_RH5 = p.prenos_chyb(R_H, [s_a5, s_t, 0.005*0.01+3*0.01/1000], [a5, t, 0.005])
#%%
fig1, ax1 = p.default_plot([B, B, B], [U_H1*1000, U_H3*1000, U_H5*1000], 'B [T]', 'U_H [mV]', fit = [[lin, a1*1000], [lin, a3*1000], [lin, a5*1000]], legend = ['$I_{12} = 1$ mA', '$I_{12} = 3$ mA', '$I_{12} = 5$ mA'], colors=p.rand_plot_colors(3), xerror = [s_B, s_B, s_B], yerror = [s_U*1000, s_U*1000, s_U*1000])
#%%
def mu(R_H, sigma):
    return R_H*sigma
print(p.round_to(mu(R_H5, sigm), p.prenos_chyb(mu, [s_RH5, s_sigm], [R_H5, sigm])))

#%%
def n(R_H):
    return (3*np.pi/8)/(1.602_176_634*1e-19*R_H)
print(p.round_to(n(R_H5)/1e20, p.prenos_chyb(n, [s_RH5], [R_H5])/1e20))