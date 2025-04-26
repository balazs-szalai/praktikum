# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:15:57 2023

@author: balaz
"""

import numpy as np
import matplotlib.pyplot as plt
import praktikum as p
import pandas as pd
from math import sqrt
def trida_pres(spec, trid=2):
    return spec*trid/sqrt(3)/100

data = np.array(pd.read_excel('mer6.xlsx'))

R = 3300
s_R = 1

U_B01 = data[:3, 1].astype(np.float64)
s_UB01 = np.sqrt(np.std(U_B01)**2 + p.prenos_chyb(lambda *x: np.mean(x), (U_B01*0.3/100+1*0.1), U_B01)**2)

U_B02 = data[:3, 2].astype(np.float64)
s_UB02 = np.sqrt(np.std(U_B02)**2 + p.prenos_chyb(lambda *x: np.mean(x), (U_B02*0.3/100+1*0.1), U_B02)**2)

U1 = data[:, 3].astype(np.float64)
s_U1 = (U1*0.3/100+1*0.1)

I1 = data[:, 4].astype(np.float64)
s_I1 = np.ones(I1.shape)*trida_pres(15)

U2 = data[:11, 5].astype(np.float64)
s_U2 = (U2*0.3/100+1*0.1)

I2 = data[:11, 6].astype(np.float64)
s_I2 = np.ones(I2.shape)*trida_pres(15)

U0 = 41.8
s_U0 = (U0*0.3/100+1*0.1)

c = data[:12, 9].astype(np.float64)+0.1
s_c = c*1/100

U_zh = data[:12, 10].astype(np.float64)
s_Uzh = np.array([0.3]*len(U_zh), np.float64)

U_B0 = data[:12, 11].astype(np.float64)
s_UB0 = np.array([0.3]*len(U_B0), np.float64)

f = data[:12, 12].astype(np.float64)
s_f = []
for val in f:
    if val < 2000:
        s_f.append(val*0.01/100+2000*0.003/100)
    else:
        s_f.append(val*0.01/100+20_000*0.003/100)
s_f = np.array(s_f, np.float64)
s_f = np.sqrt(s_f**2+(np.array([10, 10, 10, 10, 1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01], np.float64))**2)
#%% 1.
table1 = p.default_table(pd.DataFrame({
    '$U_{B01}$ [V]': p.readable(U_B01, (U_B01*0.3/100+1*0.1)),
    '$U_{B02}$ [V]': p.readable(U_B02, (U_B02*0.3/100+1*0.1))
    }), 'table1', 'namerané hodnoty spínacieho napätia $U_{B0}$ pre obidve polariry včetne chýb')

table2 = p.default_table(pd.DataFrame({
    '$U_1$ [V]': p.readable(U1, s_U1),
    '$I_1$ [mA]': p.readable(I1, s_I1),
    '$U_2$ [V]': p.pad(p.readable(-U2, s_U2), len(U1)),
    '$I_2$ [mA]': p.pad(p.readable(-I2, s_I2), len(U1)),
    }), 'table2', 'namerané hodnoty VA charakteristiky diaku, hodnoty $U_2$  a $I_2$ sú záporné, lebo sme merali s opačnou polaritou')
#%% 1 a.
print(f'$U_{{B01}} = ({p.round_to(np.mean(U_B01), s_UB01)[2:-1]}) \enspace V$')
print(f'$U_{{B02}} = ({p.round_to(np.mean(U_B02), s_UB02)[2:-1]}) \enspace V$')
#%% 1 b.
U_B01 = np.mean(U_B01)
U_B02 = np.mean(U_B02)

dU1 = U_B01-min(U1[U1>20])
s_dU1 = s_UB01+s_U1[U1>20][np.argmin(U1[U1>20])]

dU2 = U_B02-min(U2)
s_dU2 = s_UB01+s_U2[np.argmin(U2)]

print(f'$\Delta U_1 = ({p.round_to(dU1, s_dU1)[2:-1]}) \enspace V$')
print(f'$\Delta U_2 = ({p.round_to(dU2, s_dU2)[2:-1]}) \enspace V$')
#%% 1 c.
sym = abs(U_B01-U_B02)
s_sym = s_UB01+s_UB02
print(p.round_to(sym, s_sym))

#%% 2.
U = [*(-U2)]+[-U_B02]+[*U1[15:]]+[*reversed(U1[:15])]
s_U = [*(s_U2)]+[s_UB02]+[*s_U1[15:]]+[*reversed(s_U1[:15])]

I = [*(-I2)]+[0]+[*I1[15:]]+[*reversed(I1[:15])]
s_I = [*(s_I2)]+[0]+[*s_I1[15:]]+[*reversed(s_I1[:15])]

fig, ax = p.default_plot([U], [I], 'U [V]', 'I [mA]', xerror=[s_U], yerror=[s_I])
ax.plot([min(U1[U1>20]), min(U1[U1>20])], [-10, 10], '--', color = 'red')
ax.plot([-min(U2), -min(U2)], [-10, 10], '--', color = 'red')
ax.plot([U_B01, U_B01], [-10, 10], '--', color = 'red')
ax.plot([-U_B02, -U_B02], [-10, 10], '--', color = 'red')

ax.arrow(22, 9, dU1-2, 0, head_width = 0.5, color = 'k')
ax.arrow(22+dU1-1.5, 9, -dU1+2, 0, head_width = 0.5, color = 'k')
ax.text(22+dU1/2-0.5, 9.5, '$\Delta U_1$', horizontalalignment='center', fontsize = 15)

ax.arrow(-22, 9, -dU2+2, 0, head_width = 0.5, color = 'k')
ax.arrow(-22-dU2+1.5, 9, dU2-1.5, 0, head_width = 0.5, color = 'k')
ax.text(-22-dU2/2+0.5, 9.5, '$\Delta U_2$', horizontalalignment='center', fontsize = 15)

#%% 3.
table3 = p.default_table(pd.DataFrame({
    '$C$ [nF]': p.readable(c, s_c),
    '$U_{zh}$ [V]': p.readable(U_zh, s_Uzh),
    '$U_{B0}$ [V]': p.readable(U_B0, s_UB0),
    'f [Hz]': p.readable(f, s_f)
    }), 'table3', 'zamerané hodnoty pomocou oscilátoru a multimetru pre $U_{zh}$, $U_{B0}$ a $T$ ako závislosti $C$')

tau = R*c/1e6
s_tau = np.sqrt((s_R**2*(c/1e6)**2+(s_c/1e6)**2*R**2).astype(np.float64))

fig1, ax1 = p.default_plot([tau, tau], [U_zh, U_B0], '$\\tau$ [ms]', 'U [V]', legend = ['$U_{zh}$', '$U_{B0}$'])#, xerror = [s_tau, s_tau], yerror=[s_Uzh, s_UB0])

T = 1/f

def inv(x):
    return 1/x

s_T = [p.prenos_chyb_eval(inv, [s_f[i]], [f[i]]) for i in range(len(f))]
s_T = s_f/f**2

fig2, ax2 = p.default_plot([tau], [T*1000], '$\\tau$ [ms]', 'T [ms]')
#%% 4.

def t_1(R, C, U0, U_zh, U_B0):
    return R*C*np.log((U0-U_zh)/(U0-U_B0))
print(p.prenos_chyb_latex(t_1))
t1 = t_1(R, c/1e9, U0, U_zh/1.0, U_B0/1.0)
s_t1 = p.prenos_chyb_multi(t_1, [[s_R]*len(tau), s_c/1e9, [s_U0]*len(tau), s_Uzh, s_UB0], [[R]*len(tau), c/1e9, [U0]*len(tau), U_zh, U_B0])

table4 = p.default_table(pd.DataFrame({
    '$C$ [nF]': p.readable(c, s_c),
    '$T \enspace [\mu s]$': p.readable(T*1000_000, np.array(s_T)*1000_000),
    '$t_1 \enspace [\mu s]$': p.readable(t1*1000_000, np.array(np.array(s_t1)*1000_000))
    }), 'table5', 'namerané hodnoty periódy $T$ a vypočítané hodnoty $t_1$ spolu s kapacitou kondenzátoru $C$')

#%% 5. Lissajous
n = 10_000
color = p.rand_plot_colors(1)[0]
col = 'k'

t10 = 0.00357274
om = 2*np.pi/t10

x = np.linspace(0, t10*0.99, n)

def U(t, U0, U_zh, tau, t1):
    return (U_zh+(U0-U_zh)*(1-np.exp(-(t%t1)/tau)))
print(p.format_to_latex(U))
#%%

fig3, ax3 = plt.subplots(1,1)
ax3.plot(U0*np.cos(om*x), U(x, U0, 6.5, 0.0033, t10))
ax3.set_xlabel('$U_{sin}$ [V]')
ax3.set_ylabel('$U_{diak}$ [V]')
plt.title("1:1", fontdict = {'color': col,'size':20})

fig4, ax4 = plt.subplots(1,1)
ax4.plot(U0*np.cos(3*om*x[:n//2]), U(2*x[:n//2], U0, 6.5, 0.0033, t10), color = color)
ax4.plot(U0*np.cos(3*om*x[n//2+50:]), U(2*x[n//2+50:], U0, 6.5, 0.0033, t10), color = color)
ax4.set_xlabel('$U_{sin}$ [V]')
ax4.set_ylabel('$U_{diak}$ [V]')
plt.title("2:3", fontdict = {'color': col,'size':20})

fig5, ax5 = plt.subplots(1,1)
ax5.plot(U0*np.cos(2*om*x), U(x, U0, 6.5, 0.0033, t10))
ax5.set_xlabel('$U_{sin}$ [V]')
ax5.set_ylabel('$U_{diak}$ [V]')
plt.title("1:2", fontdict = {'color': col,'size':20})

fig6, ax6 = plt.subplots(1,1)
for i in range(4):
    ax6.plot(U0*np.cos(5*om*x[n//4*i+100:n//4*(i+1)]), U(4*x[n//4*i+100:n//4*(i+1)], U0, 6.5, 0.0033, t10), color = color)
ax6.set_xlabel('$U_{sin}$ [V]')
ax6.set_ylabel('$U_{diak}$ [V]')
plt.title("4:5", fontdict = {'color': col,'size':20})

fig7, ax7 = plt.subplots(1,1)
ax7.plot(U0*np.cos(3*om*x), U(x, U0, 6.5, 0.0033, t10))
ax7.set_xlabel('$U_{sin}$ [V]')
ax7.set_ylabel('$U_{diak}$ [V]')
plt.title("1:3", fontdict = {'color': col,'size':20})

fig8, ax8 = plt.subplots(1,1)
for i in range(3):
    ax8.plot(U0*np.cos(4*om*x[n//3*i+100:n//3*(i+1)]), U(3*x[n//3*i+100:n//3*(i+1)], U0, 6.5, 0.0033, t10), color = color)
ax8.set_xlabel('$U_{sin}$ [V]')
ax8.set_ylabel('$U_{diak}$ [V]')
plt.title("3:4", fontdict = {'color': col,'size':20})

a = 5
fig9, ax9 = plt.subplots(1,1)
for i in range(a):
    ax9.plot(U0*np.cos(6*om*x[n//a*i+100:n//a*(i+1)]), U(a*x[n//a*i+100:n//a*(i+1)], U0, 6.5, 0.0033, t10), color = color)
ax9.set_xlabel('$U_{sin}$ [V]')
ax9.set_ylabel('$U_{diak}$ [V]')
plt.title("5:6", fontdict = {'color': col,'size':20})

a = 7
fig10, ax10 = plt.subplots(1,1)
for i in range(a):
    ax10.plot(U0*np.cos(8*om*x[n//a*i+100:n//a*(i+1)]), U(a*x[n//a*i+100:n//a*(i+1)], U0, 6.5, 0.0033, t10), color = color)
ax10.set_xlabel('$U_{sin}$ [V]')
ax10.set_ylabel('$U_{diak}$ [V]')
plt.title("7:8", fontdict = {'color': col,'size':20})
#%%
li = data[:8, 17:]

k = [float(i[0])/float(i[-1]) for i in li[:, 0]]

f_gen1 = li[:, 1].astype(np.float64)
f_gen2 = li[:, 2].astype(np.float64)

f_1 = f_gen1*k
f_2 = f_gen2*k

table5 = p.default_table(pd.DataFrame({
    'pomer' : li[:, 0],
    '$f_{gen_1}$ [Hz]' : f_gen1.astype(np.str_),
    '$f_1$ [Hz]' : np.round(f_1, 2).astype(np.str_),
    '$f_{gen_2}$ [Hz]' : f_gen2.astype(np.str_),
    '$f_2$ [Hz]' : np.round(f_2, 2).astype(np.str_),
    }), 'table5', 'namerané frekvencie pomocou Lissajousových obrázkov')