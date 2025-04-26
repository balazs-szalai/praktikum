# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 23:16:05 2024

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

def my_sinc(x):
    if x == 0:
        return 1 
    else:
        return sin(x)/x

i_max = 10
def sinc_sum(fx, a, b, d):
    y = np.sinc(d*fx)
    
    for i in range(1, i_max):
        y += np.sinc(a*i/b)*np.sinc(d*(fx-i/b)) + np.sinc(-a*i/b)*np.sinc(d*(fx+i/b))
    
    return y

img = cv2.imread('profilometr/delta03.png')
img = np.sum(img, axis = 2)
center = (360, 660)
center = (345, 645)
# parms, errs = curve_fit(I_sq, x, data, p0 = [3, 600, 6, 0.01, 1, -0.2, 0])

mpp = 0.00345*2
lamb = 594e-6
f0 = 1000

def bg(x, B0, B1, B2, B3):
    return B0 + 1/(B1 + B2*(x-B3)**2)

def I_sq(x, a, A, B0, C):
    return A*np.sinc(a*(x-C)/(lamb*f0))**2 + B0

def I_cy(x, r, A, B0, C):
    return A*(spec.j1(2*np.pi*r*(x-C)/(lamb*f0))/(2*np.pi*r*(x-C)/(lamb*f0)))**2 + B0

def I_lin(x, a, b, A, B0, C, D):
    return A*sinc_sum((x-C)/(lamb*f0), a, b, D)**2 + B0

def get_x(x, mpp):
    return x*mpp

def FWHM(data, n = 10):
    dt = np.convolve(data, np.ones(n)/n, mode = 'same')
    dt -= np.min(dt)
    dt = np.abs(dt-np.max(dt)/2)
    x1, x2 = np.argsort(dt)[:2]
    return np.abs(x2-x1)

def circular_average(img, center):
    center_x, center_y = center
    dist_from_wall = min([center_x, img.shape[1]-center_x, center_y, img.shape[0]-center_y])
    
    ind_arr = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    
    dist_arr = np.zeros((*img.shape, 2))
    dist_arr[:, :, 0] = np.sqrt((ind_arr[0]-center_x)**2 + (ind_arr[1]-center_y)**2)
    dist_arr[:, :, 1] = img
    
    # for i in range(len(img[:, 0])):
    #     for j in range(len(img[0, :])):
    #         d = int(sqrt((i-center_x)**2 + (j-center_y)**2))
    #         dist_arr[i, j, 0] = d 
    # dist_arr[:, :, 1] = img
    
    bins = int(np.max(dist_arr[:, :, 0]))
    hist, edges = np.histogram(dist_arr[:, :, 0], bins)
    
    averaged_data = np.zeros(int(np.max(dist_arr[:, :, 0]))+1)
    # averaged_data[dist_arr[:, :, 0].astype(np.int64).reshape(img.shape[0]*img.shape[1])] += img.reshape(img.shape[0]*img.shape[1])
    # averaged_data[:int(dist_from_wall)] /= hist[:int(dist_from_wall)]
    
    for i in range(len(img[:, 0])):
        for j in range(len(img[0, :])):
            d = int(dist_arr[i, j, 0])
            if d < dist_from_wall:
                averaged_data[d] += dist_arr[i, j, 1]/hist[d]
    # averaged_data[:int(dist_from_wall)] /= hist[:int(dist_from_wall)]
    return averaged_data[: int(dist_from_wall)]
    

def get_centered_data(file, center, symmetric = True):
    img = cv2.imread(file)
    img = np.sum(img, axis = 2)
    
    data = img[center[0],:]
    x = (np.arange(img.shape[1])).astype(np.float64)
    
    parms, _ = opt.curve_fit(bg, x, data, p0 = [data[0], 1/np.max(data), 4/(np.max(data)*FWHM(data, 100)**2), np.argmax(data)])
    center_x = parms[-1]
    
    data = img[:, int(center_x)]
    x = (np.arange(img.shape[0])).astype(np.float64)
    
    parms, _ = opt.curve_fit(bg, x, data, p0 = [data[0], 1/np.max(data), 4/(np.max(data)*FWHM(data, 100)**2), np.argmax(data)])
    center_y = parms[-1]
    
    print(center_x, center_y)
    
    centered_data = circular_average(img, (center_x, center_y))
    if symmetric:
        ret = np.append(centered_data[::-1], centered_data)
        x = (np.arange(len(ret))-len(centered_data)).astype(np.float64)*mpp
    else:
        ret = centered_data
        x = np.arange(len(ret)).astype(np.float64)*mpp
    return x, ret

def get_data(file, center):
    img = cv2.imread(file)
    img = np.sum(img, axis = 2)
    
    data = img[center[0],:]
    x = (np.arange(img.shape[1])).astype(np.float64)
    
    parms, _ = opt.curve_fit(bg, x, data, p0 = [data[0], 1/np.max(data), 4/(np.max(data)*FWHM(data)**2), np.argmax(data)])
    center_x = parms[-1]
    
    data = img[:, int(center_x)]
    x = (np.arange(img.shape[0])).astype(np.float64)
    
    parms, _ = opt.curve_fit(bg, x, data, p0 = [data[0], 1/np.max(data), 4/(np.max(data)*FWHM(data)**2), np.argmax(data)])
    center_y = parms[-1]
    
    data = img[int(center_y),:]
    x = (np.arange(img.shape[1])-int(center_x)).astype(np.float64)
    
    x *= mpp
    s_x = p.prenos_chyb_multi(get_x, [[1]*len(x), [0]*len(x)], [x, [mpp]*len(x)])
    
    print(center_x, center_y)
     
    return x, s_x, data

#%% circle

# x, data = get_centered_data('profilometr/kappa02_0.5mm.png', (300, 600))

x, s_x, data = get_data('profilometr/kappa02_0.5mm.png', (300, 600))
s_x = np.array(s_x)

parms, errs = opt.curve_fit(I_cy, x[x != 0], data[x != 0], p0 = [0.5, 2500, 10, 0])

parms_cy, errs_cy = p.curve_fit(I_cy, x[x != 0], data[x != 0], p0 = [0.5, 2500, 10, 0], 
                                err = [s_x[x != 0], [3]*len(x[x != 0])],
                                global_vars={'lamb': lamb, 'f0': f0},
                                imports={'np': np, 'spec': spec})

p.default_plot([x], [data], 'x [mm]', 'I [-]', legend = ['kruhový otvor'], 
               fit=[[I_cy, *parms_cy]], marker='lines')

#%% square

x, s_x, data = get_data('profilometr/delta04_1mm.png', (300, 600))
s_x = np.array(s_x)

parms, errs = opt.curve_fit(I_cy, x[x != 0], data[x != 0], p0 = [0.5, 2500, 10, 0])

parms_sq, errs_sq = p.curve_fit(I_sq, x[x != 0], data[x != 0], p0 = [0.5, 2500, 10, 0], 
                                err = [s_x[x != 0], [3]*len(x[x != 0])],
                                global_vars={'lamb': lamb, 'f0': f0},
                                imports={'np': np, 'spec': spec})

p.default_plot([x], [data], 'x [mm]', 'I [-]', legend = ['štvorcový otvor'], 
               fit=[[I_sq, *parms_sq]], marker='lines')

#%% lines

x, s_x, data = get_data('profilometr/C07_a0.3b_0.8mm.png', (350, 600))
s_x = np.array(s_x)
#%%
parms, errs = opt.curve_fit(I_lin, x[x != 0], data[x != 0], p0 = [0.3, 1.5, 500, 10, 0, 3])
#%%
parms_lin, errs_lin = p.curve_fit(I_lin, x[x != 0], data[x != 0], p0 = [0.3, 1.5, 500, 10, 0, 3], 
                                err = [s_x[x != 0], [3]*len(x[x != 0])],
                                global_vars={'lamb': lamb, 'f0': f0, 'i_max': i_max},
                                imports={'np': np, 'spec': spec},
                                global_functions=[sinc_sum], n = 5000)

p.default_plot([x], [data], 'x [mm]', 'I [-]', legend = ['zvislé čiary'], 
               fit=[[I_lin, *parms_lin]], marker='lines')

#%%
def get_x_data(file, center):
    img = cv2.imread(file)
    img = np.sum(img, axis = 2)
    
    data = img[center[0],:]
    x = (np.arange(img.shape[1])-center[1]).astype(np.float64)
    return x, data

# i_max = 1000
# def step(x, a, b):
#     ret = b/a
#     for i in range(1, i_max):
#         ret += (np.sin(2*np.pi*i*x/a)*(1-cos(2*np.pi*i*b/a))+np.cos(2*np.pi*i*x/a)*sin(2*np.pi*i*b/a))/(np.pi*i)
#     return ret

# def scale(x, a, b, A, B, C):
#     return A*step(x-C, a, b) + B

#%% getting the scale
img = cv2.imread('mikroskop\\scale.jpg')
img = np.sum(img, axis = 2)
scale = img[900, :]

scale_fft = np.fft.rfft(scale)
freq = np.fft.rfftfreq(2048)

#%%
pow_spec_dens = np.abs(scale_fft)**2 
f1, f2 = freq[np.argsort(pow_spec_dens[1:])[-2:]+1]

mpp = f2/10

def ppm0(f):
    return f/10 

s_mpp = p.prenos_chyb(ppm0, [1/2048], [f2])
print(p.round_to(mpp, s_mpp))

#%%
def frac(a, b):
    return a/b#

print(p.round_to(frac(3, 1), p.prenos_chyb(frac, [0.5, 0], [3, 1])))
#%%
def fit_sq(x, x1, x2, x3, x4, A0, A1):
    ret = np.empty_like(x)
    ret[x < x1] = A1
    ret[(x >= x1)*(x <= x2)] = (A1 - A0)/(x2 - x1)*(x2 - x[(x >= x1)*(x <= x2)]) + A0
    ret[(x > x2)*(x < x3)] = A0
    ret[(x >= x3)*(x <= x4)] = (A0 - A1)/(x4 - x3)*(x4 - x[(x >= x3)*(x <= x4)]) + A1
    ret[x > x4] = A1
    return ret

img = cv2.imread('mikroskop/delta04.jpg')
img = np.sum(img, axis = 2)

data = np.mean(img, axis = 0)

parms, errs = p.curve_fit(fit_sq, np.arange(len(data)), data, p0 = [900, 920, 1200, 1250, 140, 40],
                          err = [[1]*len(data), [3]*len(data)], imports={'np': np})
p.default_plot([np.arange(len(data))], [data], 'pixel', 'priemer horizontálných riadkov fotky',
               fit=[[fit_sq, *parms]], marker='lines')
#%%

x1, x2, x3, x4 = parms[:4]
s_x1, s_x2, s_x3, s_x4 = errs[:4]

s_x1 = np.sqrt(s_x1**2 + s_x2**2 + np.std([x1, x2])**2)
x1 = np.mean([x1, x2])

s_x2 = np.sqrt(s_x3**2 + s_x4**2 + np.std([x3, x4])**2)
x2 = np.mean([x3, x4])
# x1, x2 = np.argsort(np.abs(data - np.max(data)/2))[:2]

d = abs(x2-x1)*mpp
def add(a, b):
    return a+b
def times(a, b):
    return a*b#

s_d = p.prenos_chyb(times, [p.prenos_chyb(add, [s_x1, s_x2], [x1, x2]), s_mpp], [abs(x2-x1), mpp])

print(p.round_to(d, s_d))

#%%
img = cv2.imread('mikroskop/kappa02.jpg')
img = np.sum(img, axis = 2)

data_x = np.mean(img, axis = 0)
data_y = np.mean(img, axis = 1)

data = data_x
x = np.arange(len(data))
parms, _ = opt.curve_fit(bg, x, data, p0 = [data[100], 1/np.max(data), 4/(np.max(data)*FWHM(data, 100)**2), np.argmax(data)])
center_x = parms[-1]
#%%
data = data_y
x = np.arange(len(data))
parms, _ = opt.curve_fit(bg, x, data, p0 = [data[100], 1/np.max(data), 4/(np.max(data)*FWHM(data, 100)**2), np.argmax(data)])
center_y = parms[-1]

centered_data = circular_average(img, (center_x, center_y))

data = centered_data
# x = np.arange(len(ret)).astype(np.float64)#*mpp

# _, data = get_centered_data('mikroskop/kappa02.jpg', (770, 1069),symmetric=False)
#%%
def fit(x, r1, r2, A0, A1):
    ret = np.empty_like(x)
    ret[(x >= r1)*(x <= r2)] = (A0 - A1)/(r2 - r1)*(r2 - x[(x >= r1)*(x <= r2)]) + A1
    ret[x <= r1] = A0 
    ret[x >= r2] = A1
    return ret

# def mean(a, b):
#     return (a+b)/2

# def fit(x, A, B, C, D):
#     return (A*np.arctan(D*(x-C)) + B)**2

parms, errs = p.curve_fit(fit, np.arange(len(data)), data, p0 = [130, 240, 800, 40], 
                          err = [[1]*len(data), [0]*len(data)], imports={'np': np})
p.default_plot([np.arange(len(data))], [data], 'pixel', 'kružnicový priemer okolo strede kruhu',
               fit=[[fit, *parms]], marker='lines')

#%%
r1, r2 = parms[:2]
s_r1, s_r2 = errs[:2]

r = np.mean([r1, r2])
s_r = np.sqrt(s_r1**2 + s_r2**2 + np.std([r1, r2])**2)

s_r = p.prenos_chyb(times, [s_r, s_mpp], [r, mpp])
r = times(r, mpp)

print(p.round_to(r, s_r))

#%%
n = 5
def fit(x, x1, x2, x3, x4, b, A0, A1):
    ret = [fit_sq(x - i*b, x1, x2, x3, x4, 1, 0) for i in range(n)]
    return (A0-A1)*np.sum(ret, axis = 0) + A1

img = cv2.imread('mikroskop/C07.jpg')
img = np.sum(img, axis = 2)

data = np.mean(img, axis = 0)
x = np.arange(len(data))

parms, errs = p.curve_fit(fit, x, data, p0 = [70, 120, 230, 280, 480, 750, 60],
                          global_vars={'n': n},
                          global_functions=[fit_sq],
                          imports={'np': np})
p.default_plot([np.arange(len(data))], [data], 'pixel', 'priemer horizontálných rezov fotky',
               fit=[[fit, *parms]], marker='lines')

x1, x2, x3, x4 = parms[:4]
s_x1, s_x2, s_x3, s_x4 = errs[:4]

s_x1 = np.sqrt(s_x1**2 + s_x2**2 + np.std([x1, x2])**2)
x1 = np.mean([x1, x2])

s_x2 = np.sqrt(s_x3**2 + s_x4**2 + np.std([x3, x4])**2)
x2 = np.mean([x3, x4])
# x1, x2 = np.argsort(np.abs(data - np.max(data)/2))[:2]

a = abs(x2-x1)*mpp
s_a = p.prenos_chyb(times, [p.prenos_chyb(add, [s_x1, s_x2], [x1, x2]), s_mpp], [abs(x2-x1), mpp])

b = parms[4]
s_b = p.prenos_chyb(times, [errs[4], s_mpp], [b, mpp])
b *= mpp

print(p.round_to(a, s_a))
print(p.round_to(b, s_b))