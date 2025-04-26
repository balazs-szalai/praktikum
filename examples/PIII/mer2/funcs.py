# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 12:42:56 2024

@author: balazs
"""

import numpy as np
from scipy.interpolate import splrep, splev

lamb = [656.3, 587.6, 546.1, 486.1, 435.8, 404.7][::-1]
N1 = [1.73268, 1.74042, 1.74676, 1.75955, 1.77591, 1.7906][::-1]
dN1dt = [[0.82, 0.9, 1.01, 1.21, 1.41, 1.66][::-1],
         [0.81, 0.89, 1.0, 1.19, 1.39, 1.64][::-1],
         [0.78, 0.86, 0.97, 1.16, 1.35, 1.6][::-1],
         [0.76, 0.84, 0.93, 1.12, 1.31, 1.55][::-1],
         [0.73, 0.81, 0.9, 1.08, 1.27, 1.5][::-1],
         [0.71, 0.78, 0.88, 1.05, 1.23, 1.46][::-1],
         [0.69, 0.76, 0.86, 1.03, 1.2, 1.43][::-1],
         [0.68, 0.75, 0.84, 1.01, 1.19, 1.41][::-1],
         [0.68, 0.75, 0.84, 1.01, 1.18, 1.4][::-1]]
ts = [15, 20, 30, 40, 50, 60, 70, 80, 85]

def intp(x, xs, ys, n = 1):
    tck = splrep(xs, ys, k = 5)
    if isinstance(x, (int, float)):
        return splev(x, tck)
        # i = 0
        # while x > xs[i]:
        #     i += 1
        # i -= 1 
        # y = (ys[i+1]-ys[i])/(xs[i+1]-xs[i]) * (x - xs[i]) + ys[i]
        # return y
    elif isinstance(x, (list, tuple, np.ndarray)):
        y = np.zeros((len(x),))
        for j in range(len(x)):
            y[j] = splev(x[j], tck)
            # i = 0
            # while x[j] > xs[i]:
            #     i += 1
            # i -= 1 
            # y[j] = (ys[i+1]-ys[i])/(xs[i+1]-xs[i]) * (x[j] - xs[i]) + ys[i]
        return y

def integral_lin(x, x0, x1, y0, y1):
    a = (y1-y0)/(x1-x0) * (x - x0)
    return y0*(x-x0) + 1/2*a*(x-x0)**2

def integral_der_arr(x, der_arr_x, der_arr_y, y0):
    delta = 0
    i = 0 
    while x > der_arr_x[i]:
        if x > der_arr_x[i+1]:
            delta += integral_lin(der_arr_x[i+1], der_arr_x[i], der_arr_x[i+1], der_arr_y[i], der_arr_y[i+1])
        i += 1 
    i -= 1 
    delta += integral_lin(x, der_arr_x[i], der_arr_x[i+1], der_arr_y[i], der_arr_y[i+1])
    return y0 + delta

N1 = [N1[i] - integral_lin(20, 15, 20, dN1dt[0][i]/10**5, dN1dt[1][i]/10**5) for i in range(len(N1))]

def n(lamb0, t):
    N = [integral_der_arr(t, ts, [dN1dt[j][i]/10**5 for j in range(len(dN1dt))], N1[i]) for i in range(len(N1))]
    return intp(lamb0, lamb, N)

N2 = [1.73329, 1.74093, 1.74724, 1.75994, 1.7762, 1.79073][::-1]
dN2dt = [[0.62, 0.68, 0.77, 0.92, 1.08, 1.29][::-1],
         [0.63, 0.69, 0.78, 0.93, 1.09, 1.3][::-1],
         [0.64, 0.71, 0.81, 0.96, 1.12, 1.33][::-1],
         [0.66, 0.73, 0.81, 0.98, 1.15, 1.36][::-1],
         [0.67, 0.74, 0.83, 1.0, 1.17, 1.39][::-1],
         [0.68, 0.75, 0.84, 1.01, 1.18, 1.4][::-1]]
ts2 = [15, 20, 30, 40, 50, 60]

N2 = [N2[i] - integral_lin(20, 15, 20, dN2dt[0][i]/10**5, dN2dt[1][i]/10**5) for i in range(len(N2))]

def n2(lamb0, t):
    N = [integral_der_arr(t, ts2, [dN2dt[j][i]/10**5 for j in range(len(dN2dt))], N2[i]) for i in range(len(N2))]
    return intp(lamb0, lamb, N)