# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:09:06 2024

@author: balazs
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

def line(x1, y1, x2, y2, x3, y3):
    def ret(x):
        if x2 == x3:
            val = np.empty_like(x)
            val[x >= x2] = -np.inf
            val[x < x2] = np.inf
            return val
        return (y3 - y2)/(x3-x2)*(x-x1) + y1
    return ret

def in_interval(x, a, b):
    a1 = np.min(np.array([a, b]), axis=0)
    b1 = np.max(np.array([a, b]), axis=0)
    # print(b1)
    if np.inf in b1:
        ret = (b1 == np.inf)
        ret[b1 != np.inf] = (x[b1 != np.inf] > a1[b1 != np.inf])*(x[b1 != np.inf] < b1[b1 != np.inf])
        return ret
    return (x > a1)*(x < b1)
def triangle(x1, y1, x2, y2, x3, y3):
    
    p11 = line(x1, y1, x2, y2, x3, y3)
    p12 = line(x2, y2, x2, y2, x3, y3)
    p21 = line(x2, y2, x1, y1, x3, y3)
    p22 = line(x1, y1, x1, y1, x3, y3)
    p31 = line(x3, y3, x1, y1, x2, y2)
    p32 = line(x1, y1, x1, y1, x2, y2)
    
    def ret(x, y):
        val1 = in_interval(y, p11(x), p12(x))
        val2 = in_interval(y, p21(x), p22(x))
        val3 = in_interval(y, p31(x), p32(x))
        return val1*val2*val3
    return ret 

def high_pass(img, cutoff):
    w, h = img.shape
    img0 = img[:, :]
    img0[(w-cutoff)//2 : (w+cutoff)//2, (h-cutoff)//2 : (h+cutoff)//2] = 0 
    return img0

def high_pass_y(img, cutoff):
    w, h = img.shape
    img0 = img[:, :]
    img0[(w-cutoff)//2 : (w+cutoff)//2, :] = 0 
    return img0

def high_pass_x(img, cutoff):
    w, h = img.shape
    img0 = img[:, :]
    img0[:, (h-cutoff)//2 : (h+cutoff)//2] = 0 
    return img0

def step(x, a, b):
    return (x%a) < b

# def comb(x, a, b, n):
#     val = np.empty_like(x)
#     for i in range(n): 
#         val += step(x, i*a, i*a+b)
#     return val

#%%

n = 1000
a = 100
# Step 1: Define Input Image
# input_image = np.sum(255-cv2.imread('mikroskop/delta03.jpg'), axis = 2)
tr1 = triangle(-1, 0, 0.5, np.sqrt(3)/2, 0.50, -np.sqrt(3)/2)
x1 = np.linspace(-10, 10, 1000)
x = np.meshgrid(x1, x1)
input_image = tr1(*x)
fourier_spectrum = np.fft.fftshift(np.fft.fft2(input_image))

#filtering
fourier_spectrum = high_pass_x(fourier_spectrum, 20)

#multiplication
# fourier_spectrum *= step(x[0], 0.1, 0.05)*step(x[1], 0.1, 0.05)

# Step 2: Simulate Lens Focusing
focal_plane_intensity = np.abs(fourier_spectrum)**2  # Intensity is magnitude of Fourier spectrum

# Step 3: Image Formation
# Capture the intensity distribution in the focal plane (For simplicity, we use the magnitude of the Fourier spectrum)
reconstructed_image = np.fft.fft2(fourier_spectrum)

# Step 4: Optional Inverse Filtering
# Perform additional processing if needed

# Visualize the input image, Fourier spectrum, and reconstructed image
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Predmet v rovine $P_1$')

plt.subplot(1, 3, 2)
plt.imshow(np.log(1 + np.abs(focal_plane_intensity)), cmap='gray')
plt.title('Fourierov obraz predmetu')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(reconstructed_image), cmap='gray')
plt.title('Výsledný obraz v rovine $P_3$')

plt.show()