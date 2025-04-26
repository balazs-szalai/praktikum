# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 16:29:23 2023

@author: Balázs local
"""

from praktikum.format_latex import format_to_latex
from praktikum.main_funcs import prenos_chyb, round_to
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

p_0 = 101_325  # Pa


def t_p(p):
    return 100 + 28.0216*(p/p_0-1) - 11.642*(p/p_0-1)**2 + 7.1*(p/p_0-1)**3


print(format_to_latex(t_p))


def R(t, R_0, A, B):
    return R_0*(1+A * t+B * t**2)


print(format_to_latex(R))

def epsilon(t_2, t_1, a, b, c):
    return a + b*(t_2-t_1) + c(t_2-t_1)**2


print(format_to_latex(epsilon))

R_M = 0.200 #Ohm
def sigma_R(R, R_M):
    return 0.5/100*R+0.1/100*R_M


print(format_to_latex(sigma_R))

epsilon_M = 0.100 #V
def sigma_epsilon(epsilon, epsilon_M):
    return 90e-6*epsilon + 35e-6*epsilon_M


print(format_to_latex(sigma_epsilon))

# %%
plt.close('all')
glob_counter = 0

u_l = pd.read_table('Lad.dat', decimal=',', encoding='ISO 8859-2',
                    sep='\s+', skiprows=1, names=['t', 'U'])
u_v = pd.read_table('Voda.dat', decimal=',', encoding='ISO 8859-2',
                    sep='\s+', skiprows=1, names=['t', 'U'])
u_c = pd.read_table('cín.dat', decimal=',', encoding='ISO 8859-2',
                    sep='\s+', skiprows=1, names=['t', 'U'])


def U_all(data, nul, k):
    global glob_counter
    glob_counter += 1
    fig1, ax1 = plt.subplots(1,1)
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('U [mV]')
    
    ax1.grid()


    ti = 0
    for i in range(nul+50, len(data.t), k):
        if abs(np.mean(data.U[nul:i])-np.mean(data.U[i:i+k])) < 3*np.std(data.U[nul:i]):
            continue
        else:
            ti = i
            break
    
    ti0 = 0
    for i in range(nul, 0, -k):
        if abs(np.mean(data.U[i:ti])-np.mean(data.U[i-k:i])) < 3*np.std(data.U[i:ti]):
            continue
        else:
            ti0 = i
            break
        
    ax1.plot(data['t'][:ti0+1], data['U'][:ti0+1]*1000, color = 'tab:blue')
    ax1.plot(data['t'][ti-1:], data['U'][ti-1:]*1000, color = 'tab:blue')
    ax1.plot(data['t'][ti0:ti], data['U'][ti0:ti]*1000, color = 'red')
    print('\n')
    print(data.t[ti0], data.t[ti])

    stds = sigma_epsilon(data.U, epsilon_M)
    ax1.fill_between(data['t'], (data.U+stds)*1000, (data.U-stds)*1000, alpha = 0.1)
    stds = stds[ti0:ti]
    
    plt.savefig(f'Figure_{glob_counter}.pdf', format = 'pdf')

    sigma_u_l =  np.sqrt(np.sum(stds**2+np.std(data.U[ti0:ti])**2)/(ti-ti0))
    U_l = np.mean(data.U[ti0:ti])*1000, sigma_u_l*1000
    print(round_to(*U_l))
    return U_l

#%%

R_l = 0.100 #Ohm
chyba_R_l = sigma_R(R_l, R_M)*1000
R_l = R_l*1000
print()
print(round_to(R_l, chyba_R_l))

R_v = pd.read_excel('kalibrace teplomeru.xlsx', 'R_voda')
R_c = pd.read_excel('kalibrace teplomeru.xlsx', 'R_cin')
R_c['t [s]'] = R_c['t [min]']*60

def R_all(data, nul, k):
    global glob_counter
    glob_counter += 1
    fig1, ax1 = plt.subplots(1,1)
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('R [Ohm]')
    
    ax1.grid()


    ti = len(data['R [kOhm]'])
    for i in range(nul+4, len(data['R [kOhm]']), k):
        if abs(np.mean(data['R [kOhm]'][nul:i])-np.mean(data['R [kOhm]'][i:i+k])) < 3*np.std(data['R [kOhm]'][nul:i]):
            continue
        else:
            ti = i
            break
    
    ti0 = 0
    for i in range(nul, 0, -k):
        if abs(np.mean(data['R [kOhm]'][i:ti])-np.mean(data['R [kOhm]'][i-k:i])) < 3*np.std(data['R [kOhm]'][i:ti]):
            continue
        else:
            ti0 = i
            break
        
    ax1.plot(data['t [s]'][:ti0+1], data['R [kOhm]'][:ti0+1]*1000, 'o-', color = 'tab:blue', markersize = 3)
    ax1.plot(data['t [s]'][ti-1:], data['R [kOhm]'][ti-1:]*1000, 'o-', color = 'tab:blue', markersize = 3)
    ax1.plot(data['t [s]'][ti0:ti], data['R [kOhm]'][ti0:ti]*1000, 'o-', color = 'red', markersize = 3)
    print('\n')
    print(data['t [s]'][ti0], data['t [s]'][ti-1])

    stds = sigma_R(data['R [kOhm]'], R_M)
    ax1.fill_between(data['t [s]'], (data['R [kOhm]']+stds)*1000, (data['R [kOhm]']-stds)*1000, alpha = 0.1)
    stds = stds[ti0:ti]
    
    plt.savefig(f'Figure_{glob_counter}.pdf', format = 'pdf')

    sigma_r =  np.sqrt(np.sum(stds**2+np.std(data['R [kOhm]'][ti0:ti])**2)/(ti-ti0))
    r = np.mean(data['R [kOhm]'][ti0:ti])*1000, sigma_r*1000
    print(round_to(*r))
    return r

#%%
plt.close('all')
glob_counter = 0
p = 101_200 #Pa

tp = t_p(p), prenos_chyb(t_p, np.diag([100**2]), [p])
print(round_to(*tp))

table = pd.DataFrame(columns = ['t [°C]', '$\varepsilon$ [mV]', 'R [$\Omega$]'], index = ['Ľad', "Voda", "Cín"])
table['t [°C]'] = [0, '$99.97 \pm 0.03$', 231.93]
table['$\varepsilon$ [mV]'] = ['$0.004 \pm 0.004$', '$4.311 \pm 0.004$', '$10.3 \pm 0.2$']
table['R [$\Omega$]'] = ['$100.0 \pm 0.7$', '$138.2 \pm 0.8$', '$156 \pm 1$']

table = table.style.to_latex(clines="all;data",
                          column_format = '|l|l|l|l|',
                          label = 'table1',
                          caption = 'hodnoty podla ktorych kalibrujeme')

def BR_0(R_1, R_2, R_3, t_1, t_2, t_3):
    return (R_3-R_1-(t_3-t_1)/(t_2-t_1)*(R_2-R_1))/((t_3-t_1)*(t_3-t_2))

def AR_0(R_1, R_2, R_3, t_1, t_2, t_3, BR_0):
    return (R_2-R_1-(t_2**2-t_1**2)*BR_0)/(t_2-t_1)

def R_0(R_1, R_2, R_3, t_1, t_2, t_3, BR_0, AR_0):
    return R_1-AR_0*t_1 - BR_0*t_1**2

def c(epsilon_1, epsilon_2, epsilon_3, t_1, t_2, t_3):
    return (epsilon_3-epsilon_1-(t_3-t_1)/(t_2-t_1)*(epsilon_2-epsilon_1))/((t_3-t_1)*(t_3-t_2))

def b(epsilon_1, epsilon_2, epsilon_3, t_1, t_2, t_3, c):
    return (epsilon_2-epsilon_1-c*(t_2**2-t_1**2))/(t_2-t_1)

def a(epsilon_1, epsilon_2, epsilon_3, t_1, t_2, t_3, c, b):
    return epsilon_1-c*t_1**2-b*t_1

# print(format_to_latex(BR_0), '\n')
# print(format_to_latex(AR_0), '\n')
# print(format_to_latex(R_0), '\n')
# print(format_to_latex(c), '\n')
# print(format_to_latex(b), '\n')
# print(format_to_latex(a), '\n')

t_1 = 0, 0
t_2 = tp
t_3 = 231.93, 0

eps_1 = U_all(u_l, 300, 2)
eps_2 = U_all(u_v, 1350*4, 2)
eps_3 = U_all(u_c, 950*4, 2)

R_1 = R_l, chyba_R_l
R_2 = R_all(R_v, 45, 2)
R_3 = R_all(R_c, 25, 2)

#%%
Br = BR_0(*np.array((R_1, R_2, R_3, t_1, t_2, t_3))[:,0]), prenos_chyb(BR_0, np.diag(np.array((R_1, R_2, R_3, t_1, t_2, t_3))[:,1]**2), np.array((R_1, R_2, R_3, t_1, t_2, t_3))[:,0])
Ar = AR_0(*np.array((R_1, R_2, R_3, t_1, t_2, t_3, Br))[:,0]), prenos_chyb(AR_0, np.diag(np.array((R_1, R_2, R_3, t_1, t_2, t_3, Br))[:,1]**2), np.array((R_1, R_2, R_3, t_1, t_2, t_3, Br))[:,0])
R0 = R_0(*np.array((R_1, R_2, R_3, t_1, t_2, t_3, Br, Ar))[:,0]), prenos_chyb(R_0, np.diag(np.array((R_1, R_2, R_3, t_1, t_2, t_3, Br, Ar))[:,1]**2), np.array((R_1, R_2, R_3, t_1, t_2, t_3, Br, Ar))[:,0])

B = Br[0]/R0[0], np.sqrt(Br[0]**2/R0[0]**4*R0[1]**2+Br[1]**2/R0[0]**2)
A = Ar[0]/R0[0], np.sqrt(Ar[0]**2/R0[0]**4*R0[1]**2+Ar[1]**2/R0[0]**2)
R0 = R0[0], R0[1]

ce = c(*np.array((eps_1, eps_2, eps_3, t_1, t_2, t_3))[:,0]), prenos_chyb(c, np.diag(np.array((eps_1, eps_2, eps_3, t_1, t_2, t_3))[:,1]**2), np.array((eps_1, eps_2, eps_3, t_1, t_2, t_3))[:,0])
be = b(*np.array((eps_1, eps_2, eps_3, t_1, t_2, t_3, ce))[:,0]), prenos_chyb(b, np.diag(np.array((eps_1, eps_2, eps_3, t_1, t_2, t_3, ce))[:,1]**2), np.array((eps_1, eps_2, eps_3, t_1, t_2, t_3, ce))[:,0])
ae = a(*np.array((eps_1, eps_2, eps_3, t_1, t_2, t_3, ce, be))[:,0]), prenos_chyb(a, np.diag(np.array((eps_1, eps_2, eps_3, t_1, t_2, t_3, ce, be))[:,1]**2), np.array((eps_1, eps_2, eps_3, t_1, t_2, t_3, ce, be))[:,0])

print(f'$R_0 = {round_to(*R0)[1:-1]}$')
print(f'$A = {round_to(*A)[1:-1]}$')
print(f'$B = {round_to(*B)[1:-1]}$')

print(f'$a = {round_to(*ae)[1:-1]}$')
print(f'$b = {round_to(*be)[1:-1]}$')
print(f'$c = {round_to(*ce)[1:-1]}$')

#%%
def t_r(R, R0, A, B):
    return 2*(1-R/R0)/(-A-np.sqrt(A**2+4*(R-R0)/R0*B))

def t_u(epsilon, a, b, c):
    return (-b+np.sqrt(b**2-4*c*(a-epsilon)))/(2*c)

# def dtdb(R, R0, A, B):
#     return -4*(1-R/R0)**2/(-A-np.sqrt(A**2+4*(R-R0)/R0*B))**2/(np.sqrt(A**2+4*(R-R0)/R0*B))

# def chybat_r(R, R0, A, B):
#     return 

# print(format_to_latex(t_r))
# print()
# print(format_to_latex(t_u))

# glob_counter += 1
# fig, ax = plt.subplots(1,1)
# ax.set_xlabel('t [s]')
# ax.set_ylabel('T [°C]')
# ax.grid()
# ts = t_u(u_l['U']*1000, ae[0], be[0], ce[0])
# ax.plot(u_l['t'], ts, label = 'termočlánok')

# stds = sigma_epsilon(u_l['U'], epsilon_M)
# chyby = []
# for i, std in enumerate(stds):
#     chyby.append(prenos_chyb(t_u, np.diag([(std*1000)**2, ae[1]**2, be[1]**2, ce[1]**2]), [u_l['U'][i]*1000, ae[0], be[0], ce[0]]))
# ax.fill_between(u_l['t'], ts+chyby, ts-chyby, alpha = 0.2)

# ax.legend()


def Tt(R_v, u_v):
    global glob_counter
    glob_counter += 1
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('T [°C]')
    ax.grid()

    ts = t_r(R_v['R [kOhm]']*1000, R0[0], A[0], B[0])
    ax.plot(R_v['t [s]'], ts, label = 'odporový teplomer')

    stds = sigma_R(R_v['R [kOhm]'], R_M)
    chyby = []
    for i, std in enumerate(stds):
        chyby.append(prenos_chyb(t_r, np.diag([(std*1000)**2, R0[1]**2, A[1]**2, B[1]**2]), [R_v['R [kOhm]'][i]*1000, R0[0], A[0], B[0]]))
    ax.fill_between(R_v['t [s]'], ts+chyby, ts-chyby, alpha = 0.15)

    ts = t_u(u_v['U']*1000, ae[0], be[0], ce[0])
    ax.plot(u_v['t'], ts, label = 'termočlánok')

    stds = sigma_epsilon(u_v['U'], epsilon_M)
    chyby = []
    for i, std in enumerate(stds):
        chyby.append(prenos_chyb(t_u, np.diag([(std*1000)**2, ae[1]**2, be[1]**2, ce[1]**2]), [u_v['U'][i]*1000, ae[0], be[0], ce[0]]))
    ax.fill_between(u_v['t'], ts+chyby, ts-chyby, alpha = 0.2)

    ax.legend()

Tt(R_v, u_v)
plt.savefig(f'Figure_{glob_counter}.pdf', format = 'pdf')

glob_counter += 1
fig, ax = plt.subplots(1,1)
ax.set_xlabel('t [s]')
ax.set_ylabel('T [°C]')
ax.grid()
ts = t_u(u_c['U']*1000, ae[0], be[0], ce[0])
ax.plot(u_c['t'], ts, label = 'termočlánok')

stds = sigma_epsilon(u_c['U'], epsilon_M)
chyby = []
for i, std in enumerate(stds):
    chyby.append(prenos_chyb(t_u, np.diag([(std*1000)**2, ae[1]**2, be[1]**2, ce[1]**2]), [u_c['U'][i]*1000, ae[0], be[0], ce[0]]))
ax.fill_between(u_c['t'], ts+chyby, ts-chyby, alpha = 0.2)
ax.legend()
plt.savefig(f'Figure_{glob_counter}.pdf', format = 'pdf')