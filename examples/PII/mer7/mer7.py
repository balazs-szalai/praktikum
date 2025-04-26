# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:00:48 2023

@author: balaz
"""

import numpy as np
import matplotlib.pyplot as plt
import praktikum as p
import pandas as pd
from math import sqrt


def Keithley_U(U, rd):
    U0 = np.copy(U)
    res = np.empty_like(U)
    U = U0[U0 < 100]
    res[U0 < 100] = np.sqrt((U*0.0055/100+100*0.004/100)**2+rd[U0 < 100]**2)
    U = U0[(U0 > 100)*(U0 < 1000)]
    res[(U0 > 100)*(U0 < 1000)] = np.sqrt((U*0.0045/100 +
                                           1000*0.0008/100)**2+rd[(U0 > 100)*(U0 < 1000)]**2)
    return res


def Keithley_I(I, rd):
    res = np.empty_like(I)
    I0 = np.copy(I)
    I = I0[I0 < 10]
    res = np.sqrt((I*0.0055/100+10*0.025/100)**2+rd[I0 < 10]**2)
    I = I0[(I0 > 10)*(I0 < 100)]
    res[(I0 > 10)*(I0 < 100)] = np.sqrt((I*0.0055/100 +
                                         1000*0.006/100)**2+rd[(I0 > 10)*(I0 < 100)]**2)
    return res


def Fluke_U(U, rd):
    U0 = np.copy(U)
    res = np.empty_like(U)
    U = U0[U0 < 0.2]
    res[U0 < 0.2] = np.sqrt((U*0.015/100+0.2*0.004/100)**2+rd[U0 < 0.2]**2)
    U = U0[(U0 > 0.2)*(U0 < 2)]
    res[(U0 > 0.2)*(U0 < 2)] = np.sqrt((U*0.015/100 +
                                        2*0.003/100)**2+rd[(U0 > 0.2)*(U0 < 2)]**2)
    U = U0[(U0 > 2)*(U0 < 20)]
    res[(U0 > 2)*(U0 < 20)] = np.sqrt((U*0.015/100 +
                                       2*0.004/100)**2+rd[(U0 > 2)*(U0 < 20)]**2)
    return res


def Fluke_I(I, rd):
    res = np.empty_like(I)
    I0 = np.copy(I)

    I = I0[I0 < 0.2]
    res[I0 < 0.2] = np.sqrt((I*0.03/100+0.2*0.005/100)**2+rd[I0 < 0.2]**2)

    I = I0[(I0 > 0.2)*(I0 < 2)]
    res[(I0 > 0.2)*(I0 < 2)] = np.sqrt((I*0.02/100 +
                                        2*0.005/100)**2+rd[(I0 > 0.2)*(I0 < 2)]**2)

    I = I0[(I0 > 2)*(I0 < 20)]
    res[(I0 > 2)*(I0 < 20)] = np.sqrt((I*0.04/100 +
                                       20*0.02/100)**2+rd[(I0 > 2)*(I0 < 20)]**2)

    I = I0[(I0 > 20)*(I0 < 200)]
    res[(I0 > 20)*(I0 < 200)] = np.sqrt((I*0.03/100 +
                                         200*0.008/100)**2+rd[(I0 > 20)*(I0 < 200)]**2)

    I = I0[(I0 > 200)*(I0 < 2000)]
    res[(I0 > 200)*(I0 < 2000)] = np.sqrt((I*0.08/100 +
                                           2000*0.02/100)**2+rd[(I0 > 200)*(I0 < 2000)]**2)

    return res


data = np.array(pd.read_excel('mer7.xlsx'))

U_BE = data[:, 0]
U_BE = U_BE[~np.isnan(U_BE)]
s_UBE = data[:, 1][~np.isnan(data[:, 1])]
s_UBE = Keithley_U(U_BE, s_UBE)


I_B = data[:, 2]
I_B = I_B[~np.isnan(I_B)]
s_IB = data[:, 3][~np.isnan(data[:, 3])]
s_IB = Keithley_I(I_B, s_IB)


U_CE2 = data[:, 6]
U_CE2 = U_CE2[~np.isnan(U_CE2)]
s_UCE2 = data[:, 7][~np.isnan(data[:, 7])]
s_UCE2 = Fluke_U(U_CE2, s_UCE2)


I_C2 = data[:, 8]
I_C2 = I_C2[~np.isnan(I_C2)]
s_IC2 = data[:, 9][~np.isnan(data[:, 9])]
s_IC2 = Fluke_I(I_C2, s_IC2)


U_CE02 = data[:, 11]
U_CE02 = U_CE02[~np.isnan(U_CE02)]
s_UCE02 = data[:, 12][~np.isnan(data[:, 12])]
s_UCE02 = Fluke_U(U_CE02, s_UCE02)


I_C02 = data[:, 13]
I_C02 = I_C02[~np.isnan(I_C02)]
s_IC02 = data[:, 14][~np.isnan(data[:, 14])]
s_IC02 = Fluke_I(I_C02, s_IC02)


U_CE03 = data[:, 16]
U_CE03 = U_CE03[~np.isnan(U_CE03)]
s_UCE03 = data[:, 17][~np.isnan(data[:, 17])]
s_UCE03 = Fluke_U(U_CE03, s_UCE03)


I_C03 = data[:, 18]
I_C03 = I_C03[~np.isnan(I_C03)]
s_IC03 = data[:, 19][~np.isnan(data[:, 19])]
s_IC03 = Fluke_I(I_C03, s_IC03)


I_B2 = data[:, 21]
I_B2 = I_B2[~np.isnan(I_B2)]
s_IB2 = data[:, 22][~np.isnan(data[:, 22])]
s_IB2 = Keithley_I(I_B2, s_IB2)


I_C2b = data[:, 23]
I_C2b = I_C2b[~np.isnan(I_C2b)]
s_IC2b = data[:, 24][~np.isnan(data[:, 24])]
s_IC2b = Fluke_I(I_C2b, s_IC2b)


I_B6 = data[:, 26]
I_B6 = I_B6[~np.isnan(I_B6)]
s_IB6 = data[:, 27][~np.isnan(data[:, 27])]
s_IB6 = Keithley_I(I_B6, s_IB6)


I_C6b = data[:, 28]
I_C6b = I_C6b[~np.isnan(I_C6b)]
s_IC6b = data[:, 29][~np.isnan(data[:, 29])]
s_IC6b = Fluke_I(I_C6b, s_IC6b)

# %% 1 -- I_B(U_BE)
def f(x, b, c):
    return c*(np.e**(b*x)-1)


table1 = p.default_table(pd.DataFrame({
    '$U_{BE}$ [mV]': p.readable(U_BE, s_UBE),
    '$I_B$ [mA]': p.readable(I_B, s_IB)
}), 'table1', 'namerané hodnoty prúdu $I_B$ v závislosti na $U_{BE}$ pri napäti 5 V medzi kolektororm a emittorom')


parms, errs = p.curve_fit(f, U_BE[((U_BE > 685) | (U_BE < 580))], I_B[((U_BE > 685) | (U_BE < 580))], p0=[
                          1/500, 0.1], err=[s_UBE[((U_BE > 685) | (U_BE < 580))], s_IB[((U_BE > 685) | (U_BE < 580))]], imports = {'np':np})
# a, b = p.lin_fit(U_BE[U_BE>685], np.log(I_B[U_BE>685]))

fig1, ax1 = p.default_plot(U_BE, I_B, '$U_{BE}$ [mV]', '$I_B$ [mA]',
                           xerror=[s_UBE], yerror=[s_IB], fit=[[f, *parms]],
                           legend=['vstupná charakteristika'])
# parms, errs = p.curve_fit(f, U_BE[((U_BE < 685) | (U_BE < 580))], I_B[((U_BE < 685) | (U_BE < 580))], p0=[
#                           1/500, 0.1], global_vars={'e': e}, err=[s_UBE[((U_BE < 685) | (U_BE < 580))], s_IB[((U_BE < 685) | (U_BE < 580))]])
ax1.plot(np.linspace(0, 720, 1000), f(np.linspace(0, 720, 1000), *parms), '--',
         color=p.rand_plot_colors(2)[1], alpha=0.5, label='fit pre nízky prúd')
ax1.legend()


# %% 2 -- I_C(U_CE)

def f(x, a, b):
    return a*(1-np.e**(b*x))


table2 = p.default_table(pd.DataFrame({
    '$U_{CE}$ [V]': p.readable(U_CE02, s_UCE02),
    '$I_{C}$ [mA]': p.readable(I_C02, s_IC02),
    '$U_{CE} $ [V]': p.readable(U_CE03, s_UCE03),
    '$I_{C} $ [mA]': p.readable(I_C03, s_IC03),
    '$U_{CE}  $ [V]': p.pad(p.readable(U_CE2, s_UCE2), len(U_CE02)),
    '$I_{C}  $ [mA]': p.pad(p.readable(I_C2, s_IC2), len(U_CE02)),
}), 'table2', 'namerané hodnoty prúdu $I_C$ v závislosti na $U_{CE}$ pre $I_B$ = 0.2, 0.3 a 2 mA',
    header=[(2, '$I_B$ = 0.2 mA'), (2, '$I_B$ = 0.3 mA'), (2, '$I_B$ = 2 mA')])
fig2, ax2 = p.default_plot([U_CE02, U_CE03, U_CE2], [I_C02, I_C03, I_C2], '$U_{BE}$ [V]', '$I_C$ [mA]',
                           legend=['$I_B$ = 0.2 mA',
                                   '$I_B$ = 0.3 mA', '$I_B$ = 2 mA'],
                           xerror=[s_UCE02, s_UCE03, s_UCE2],
                           yerror=[s_IC02, s_IC03, s_IC2])
fig2, ax2 = p.default_plot([U_CE02, U_CE03], [I_C02, I_C03], '$U_{BE}$ [V]', '$I_C$ [mA]',
                           legend=['$I_B$ = 0.2 mA', '$I_B$ = 0.3 mA'],
                           xerror=[s_UCE02, s_UCE03],
                           yerror=[s_IC02, s_IC03])

# %% 3 -- I_C(I_B)
table3 = p.default_table(pd.DataFrame({
    '$I_B$ [mA]': p.readable(I_B2, s_IB2),
    '$I_C$ [mA]': p.readable(I_C2b, s_IC2b),
    '$I_B $ [mA]': p.readable(I_B6, s_IB6),
    '$I_C $ [mA]': p.readable(I_C6b, s_IC6b),
}), 'table3', 'namerané hodnoty kollektorového prúdu $I_C$ v závislosti na bázového prúdu $I_B$, pre 2 rôzne napätí $U_{CE}$ = 2 a 6 V',
    header=[(2, '$U_{CE}$ = 2 V'), (2, '$U_{CE}$ = 6 V')])

# %% 4 -- lin fit


def lin(x, a):
    return a*x


a2, s_a2 = p.curve_fit(lin, I_B2, I_C2b, [s_IB2, s_IC2b])
a2, s_a2 = a2[0], s_a2[0]

a6, s_a6 = p.curve_fit(lin, I_B6, I_C6b, [s_IB6, s_IC6b], global_functions=[Keithley_I], imports={'np': np})
a6, s_a6 = a6[0], s_a6[0]
# %%
fig3, ax3 = p.default_plot([I_B2, I_B6], [I_C2b, I_C6b], '$I_B$ [mA]', '$I_C$ [mA]',
                           legend=['$U_{CE}$ = 2 V', '$U_{CE}$ = 6 V'],
                           xerror=[s_IB2, s_IB6], yerror=[s_IC2b, s_IC6b], fit=[[lin, a2], [lin, a6]])
