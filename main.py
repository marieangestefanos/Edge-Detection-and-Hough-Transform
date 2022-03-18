# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import convolve2d, correlate2d
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from Functions import contour, discgaussfft, showgrey, zerocrosscurves, \
    thresholdcurves, overlaycurves, locmax8, binsubsample

# plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams["image.cmap"] = 'gray'
pi = np.pi

def deltax():
    dx = np.zeros((5, 5))
    dx[2, 1:4] = [0.5, 0, -0.5]
    return dx
def deltay():
    return deltax().T


def Lv(inpic, mode='same'):
    dx = np.zeros((5, 5))
    dx[2, 1:4] = [0.5, 0, -0.5]
    dy = dx.T

    Lx = convolve2d(inpic, dx, mode='same')
    Ly = convolve2d(inpic, dy, mode='same')

    return np.sqrt(Lx**2+Ly**2)


def Lvvtilde(inpic, mode='same'):
    dx = np.zeros((5, 5))
    dx[2, 1:4] = [0.5, 0, -0.5]
    dy = dx.T

    dxx = np.zeros((5, 5))
    dxx[2, 1:4] = [1, -2, 1]
    dyy = dxx.T

    dxy = convolve2d(dx, dy, mode='same')

    Lx = convolve2d(inpic, dx, mode='same')
    Ly = convolve2d(inpic, dy, mode='same')
    Lxx = convolve2d(inpic, dxx, mode='same')
    Lyy = convolve2d(inpic, dyy, mode='same')
    Lxy = convolve2d(inpic, dxy, mode='same')

    return Lx**2*Lxx + 2*Lx*Ly*Lxy + Ly**2*Lyy


def Lvvvtilde(inpic, mode='same'):
    dx = np.zeros((5, 5))
    dx[2, 1:4] = [0.5, 0, -0.5]
    dy = dx.T

    dxx = np.zeros((5, 5))
    dxx[2, 1:4] = [1, -2, 1]
    dyy = dxx.T
    dxy = convolve2d(dx, dy, mode='same')

    dxxx = convolve2d(dxx, dx, mode='same')
    dyyy = convolve2d(dyy, dy, mode='same')
    dxxy = convolve2d(dxx, dy, mode='same')
    dxyy = convolve2d(dxy, dy, mode='same')

    Lx = convolve2d(inpic, dx, mode='same')
    Ly = convolve2d(inpic, dy, mode='same')
    Lxx = convolve2d(inpic, dxx, mode='same')
    Lyy = convolve2d(inpic, dyy, mode='same')
    Lxy = convolve2d(inpic, dxy, mode='same')
    Lxxx = convolve2d(inpic, dxxx, mode='same')
    Lyyy = convolve2d(inpic, dyyy, mode='same')
    Lxxy = convolve2d(inpic, dxxy, mode='same')
    Lxyy = convolve2d(inpic, dxyy, mode='same')

    return (Lx**3)*Lxxx+3*(Lx**2)*Ly*Lxxy+3*Lx*(Ly**2)*Lxyy+(Ly**3)*Lyyy


def extractedges(inpic, scale, threshold, mode='same'):
    blurred_pic = discgaussfft(inpic, scale)
    Lvpic = Lv(blurred_pic, mode)
    Lvvpic = Lvvtilde(blurred_pic, mode)
    Lvvvpic = Lvvvtilde(blurred_pic, mode)

    curves = zerocrosscurves(Lvvpic, (Lvvvpic < 0))
    curves = thresholdcurves(curves, (Lvpic > threshold))

    return curves


def houghline(curves, magnitude, nrho, ntheta, threshold, nlines, verbose):

    w, h = magnitude.shape

    acc = np.zeros((nrho, ntheta))

    diag_len = int(np.sqrt(w**2+h**2))
    thetas = np.linspace(-pi/2, pi/2, ntheta)
    rhos = np.linspace(-diag_len, diag_len, nrho)

    (Y, X) = curves
    nedges = len(X)
    for i in range(nedges):
        y, x = X[i], Y[i]
        
        if magnitude[x, y] > threshold:
            for theta_idx, theta in enumerate(thetas):
                rho = x*np.cos(theta) + y*np.sin(theta)
                rho_idx = np.where(rhos < rho)[0][-1]
                # Q9
                # acc[rho_idx, theta_idx] += 1
                # Q10
                # With just magnitude
                # acc[rho_idx, theta_idx] += magnitude[x, y]
                # With a log function
                acc[rho_idx, theta_idx] += np.log(magnitude[x, y])

    pos, value, _ = locmax8(acc)
    indexvector = np.argsort(value)[-nlines:]
    pos = pos[indexvector]
    if verbose==1:
        plt.imshow(np.log(1+acc))
        for idx in range(len(pos)):
            plt.scatter(pos[idx,0], pos[idx,1])
        plt.show()
    
    if len(pos) < nlines:
        nlines = len(pos)
    
    linepar = np.zeros((nlines, 2))
    
    for idx in range(nlines):
        rhoidxacc = pos[idx, 1]
        thetaidxacc = pos[idx, 0]
        if verbose == 1:
            print("#######")
            print("Calcul pos theta", pi/ntheta*pos[idx,0]-pi/2)
            print("Value theta:", thetas[thetaidxacc])
            
            print("Calcul pos rho", 2*diag_len/nrho*pos[idx,1]-diag_len)
            print("Value rho:", rhos[rhoidxacc])
        linepar[idx][0] = rhos[rhoidxacc]
        linepar[idx][1] = thetas[thetaidxacc]
        # linepar[idx][0] = 2*diag_len/nrho*pos[idx,1]-diag_len
        # linepar[idx][1] = pi/ntheta*pos[idx,0]-pi/2
    
    if verbose == 1:
        fig, axs = plt.subplots(1, 2)

        axs[0].imshow(magnitude)
        axs[0].set_xlim((0, w))
        axs[0].set_ylim((h, 0))
        for line in linepar:
            x0 = line[0]*np.cos(line[1])
            y0 = line[0]*np.sin(line[1])
            
            dx = diag_len*-np.sin(line[1]);
            dy = diag_len*(np.cos(line[1]));
    
            
            axs[0].plot([y0-dy, y0, y0+dy], [x0-dx, x0, x0+dx], 'r-')
            
        axs[1].imshow(np.log(1+acc), cmap='gray', extent=[180/pi*thetas[0], 180/pi*thetas[-1], rhos[-1], rhos[0]])
        axs[1].set_aspect('equal', adjustable='box')
        axs[1].set_title('Hough transform')
        axs[1].set_xlabel('Angles (degrees)')
        axs[1].set_ylabel('Distance (pixels)')
        axs[1].axis('image')
        
        for idx in range(len(pos)):
            plt.scatter(180/ntheta*pos[idx,0]-180/2, 2*diag_len/nrho*pos[idx,1]-diag_len, s=80)

        plt.show()
    
    return [linepar, acc]
    

def houghedgeline(pic, scale, edgethresh, gradmagnthreshold, nrho, ntheta, nlines, verbose):
    edgecurves = extractedges(pic, scale=scale, threshold=edgethresh, mode='same')
    magnitude = Lv(discgaussfft(pic, scale))
    w, h = magnitude.shape
    
    [linepar, acc] = houghline(edgecurves, magnitude, nrho=nrho, ntheta=ntheta, \
                               threshold=gradmagnthreshold, nlines=nlines, verbose=0)
        
    if verbose==1:
        fig, axs = plt.subplots(1, 2)
        fig.set_figheight(5)
        fig.set_figwidth(10)
        
        fig.suptitle(f"scake:{scale};edgethresh:{edgethresh};magthresh{gradmagnthreshold};nrho:{nrho}; ntheta{ntheta}")
        axs[0].imshow(pic), axs[0].set_title("Original"), axs[0].axis("off")
        
        axs[1].imshow(magnitude), axs[1].set_title(f"Hough ({nlines} lines)"), axs[1].axis("off")
        axs[1].set_xlim((0, w))
        axs[1].set_ylim((h, 0))
        for line in linepar:
            x0 = line[0]*np.cos(line[1])
            y0 = line[0]*np.sin(line[1])
            dx = 3000*-np.sin(line[1]);
            dy = 3000*(np.cos(line[1]));
            axs[1].plot([y0-dy, y0, y0+dy], [x0-dx, x0, x0+dx], 'r-')
        plt.show()

    return [linepar, acc]

# # Test zone
# # img = np.load("Images-npy/triangle128.npy")

# # edgecurves = extractedges(img, 1, 4, mode='same')

# # magnitude = Lv(img)#[1:img.shape[0]-1, 1:img.shape[1]-1]
# # [linepar, acc] = houghline(edgecurves, magnitude, nrho=1000, ntheta=1000, nlines=10, verbose=0)


# img = np.load("Images-npy/houghtest256.npy")
# for scale in [0.0001, 1, 4, 16, 64]:
#     for edgethresh in [0,10,15]:
#         for gradmagnthreshold in [0, 10, 50]:
#             houghedgeline(img, scale, edgethresh, gradmagnthreshold, nrho=100, ntheta=100, nlines=10, verbose=1)


# Image loading

tools = np.load("Images-npy/few256.npy")
house = np.load("Images-npy/godthem256.npy")


"""____________________"""
"""Difference Operators"""
"""____________________"""

# Q1

# dxtools = convolve2d(tools, deltax(), 'valid')
# dytools = convolve2d(tools, deltay(), 'valid')

# print("Original shape:", tools.shape)
# print("dx shape:", dxtools.shape)
# print("dy shape:", dytools.shape)

# fig, axs = plt.subplots(1, 3)
# for ax in axs:
#     ax.axis('off')

# axs[0].imshow(tools), axs[0].set_title("Image")
# axs[1].imshow(dxtools), axs[1].set_title("dxtools")
# axs[2].imshow(dytools), axs[2].set_title("dytools")

"""______________________________________________"""
"""Point–wise thresholding of gradient magnitudes"""
"""______________________________________________"""

# Histogram without Gaussian filtering
# gradmagtools = Lv(tools)
# _ = plt.hist(gradmagtools.flatten(), bins='auto', histtype='step')

# Histogram with Gaussian filtering
# for scale in [0.0001, 1, 4, 16, 64, 128]:
#     gradmagtools = Lv(discgaussfft(tools, scale))
#     _ = plt.hist(gradmagtools.flatten(), bins='auto', histtype='step')


# for thresh

# Q2
"""Smoothing flatten a little bit the histogram, but for images with a few
different bins in doesn't really help
"""

# Q3

        
# thresholds = [t for t in range(10,35, 5)]
# scales = [0.0001, 1.0, 4.0, 16, 64, 128]

# fig, axs = plt.subplots(1,5)
# for ax in axs:
#     ax.axis("off")

# foo = 0

# for i in range(5):
#     gradmagtools = Lv(discgaussfft(tools, scales[0]))
#     axs[i].imshow((gradmagtools > thresholds[foo]).astype(int)), axs[i].set_title(f'thresh:{thresholds[foo]}')
    
#     foo+=1
#     plt.show()
    
    
# fig, axs = plt.subplots(1,5)
# for ax in axs:
#     ax.axis("off")

# foo = 0

# for i in range(5):
#     gradmagtools = Lv(discgaussfft(tools, scales[1]))
#     axs[i].imshow((gradmagtools > thresholds[foo]).astype(int)), axs[i].set_title(f'thresh:{thresholds[foo]}')
    
#     foo+=1
#     plt.show()
    

# fig, axs = plt.subplots(1,5)
# for ax in axs:
#     ax.axis("off")

# foo = 0

# for i in range(5):
#     gradmagtools = Lv(discgaussfft(tools, scales[2]))
#     axs[i].imshow((gradmagtools > thresholds[foo]).astype(int)), axs[i].set_title(f'thresh:{thresholds[foo]}')
    
#     foo+=1
#     plt.show()

"""___________________________________________"""
"""Computing differential geometry descriptors"""
"""___________________________________________"""


# ### For house

# fig, axs = plt.subplots(2,5, gridspec_kw = {'wspace':0.05, 'hspace':0})
# fig.set_figheight(7)
# fig.set_figwidth(14)
# for ax in axs:
#     for x in ax:
#         x.axis("off")
# for i, scale in enumerate([0.0001, 1, 4, 16, 64]):
#     axs[0][i].imshow(contour(Lvvtilde(discgaussfft(house, scale), 'same')))
#     axs[0][i].set_title(f"scale:{scale}")
#
# for i, scale in enumerate([0.0001, 1, 4, 16, 64]):
#     axs[1][i].imshow(((Lvvvtilde(discgaussfft(house, scale), 'same')<0).astype(int)))
#
# plt.show()
# ### For tools

# fig, axs = plt.subplots(2,5, gridspec_kw = {'wspace':0.05, 'hspace':0})
# fig.set_figheight(7)
# fig.set_figwidth(14)
# for ax in axs:
#     for x in ax:
#         x.axis("off")
# for i, scale in enumerate([0.0001, 1, 4, 16, 64]):
#     axs[0][i].imshow(contour(Lvvtilde(discgaussfft(tools, scale), 'same')))
#     axs[0][i].set_title(f"scale:{scale}")
#
# for i, scale in enumerate([0.0001, 1, 4, 16, 64]):
#     axs[1][i].imshow(((Lvvvtilde(discgaussfft(tools, scale), 'same')<0).astype(int)))
# plt.show()

"""___________________________"""
"""Extraction of edge segments"""
"""___________________________"""

### For house

# thresholds = [0, 3,5,8,10]
# fig, axs = plt.subplots(len(thresholds),5)
# fig.set_figheight(30)
# fig.set_figwidth(40)
#
# for ax in axs:
#     for x in ax:
#         x.axis("off")
#
# for j, thresh in enumerate(thresholds):
#
#     for i, scale in enumerate([0.0001, 1, 4, 16, 64]):
#         [h, w] = np.shape(house)
#         rgb = np.zeros((h, w, 3), dtype=np.uint8)
#         rgb[:,:,0] = rgb[:,:,1] = rgb[:,:,2] = house
#         (Y, X) = extractedges(house, scale=scale, threshold=thresh)
#         rgb[Y, X, 0] = 0
#         rgb[Y, X, 1] = 0
#         rgb[Y, X, 2] = 255
#         axs[j][i].imshow(rgb)
#         axs[j][i].set_title(f"s:{scale};t:{thresh}")



# ### For tools

# thresholds = [0, 3,5,8,10]
# fig, axs = plt.subplots(len(thresholds),5)
# fig.set_figheight(30)
# fig.set_figwidth(40)

# for ax in axs:
#     for x in ax:
#         x.axis("off")

# for j, thresh in enumerate(thresholds):

#     for i, scale in enumerate([0.0001, 1, 4, 16, 64]):
#         [h, w] = np.shape(house)
#         rgb = np.zeros((h, w, 3), dtype=np.uint8)
#         rgb[:,:,0] = rgb[:,:,1] = rgb[:,:,2] = tools
#         (Y, X) = extractedges(tools, scale=scale, threshold=thresh)
#         rgb[Y, X, 0] = 0
#         rgb[Y, X, 1] = 0
#         rgb[Y, X, 2] = 255
#         axs[j][i].imshow(rgb)
#         axs[j][i].set_title(f"s:{scale};t:{thresh}")


"""________________"""
"""Hough transform"""
"""________________"""

# houghtest = np.load("Imaqges-npy/houghtest256.npy")
# for scale in [0.0001, 1, 4, 16, 64]:
#     houghedgeline(houghtest, scale, edgethresh=10, gradmagnthreshold=10, nrho=500, ntheta=500, nlines=3, verbose=1)
# houghedgeline(houghtest, scale=4, edgethresh=10, gradmagnthreshold=10, nrho=500, ntheta=500, nlines=3, verbose=1)
# triangle128 = np.load("Images-npy/triangle128.npy")
# houghedgeline(triangle128, scale=0.01, edgethresh=10, gradmagnthreshold=10, nrho=500, ntheta=500, nlines=3, verbose=1)

few = np.load("Images-npy/few256.npy")
# for scale in [0.0001, 1, 4, 16, 64]:
#     houghedgeline(few, scale, edgethresh=10, gradmagnthreshold=10, nrho=100, ntheta=100, nlines=10, verbose=1)

houghedgeline(few, scale=4, edgethresh=7, gradmagnthreshold=10, nrho=1000, ntheta=1000, nlines=40, verbose=1)

# phonecalc = np.load("Images-npy/phonecalc256.npy")
# for scale in [0.0001, 1, 4, 16, 64]:
#     houghedgeline(phonecalc, scale, edgethresh=10, gradmagnthreshold=10, nrho=100, ntheta=100, nlines=10, verbose=1)
#
# house = np.load("Images-npy/godthem256.npy")
# for scale in [0.0001, 1, 4, 16, 64]:
#     houghedgeline(phonecalc, scale, edgethresh=10, gradmagnthreshold=10, nrho=100, ntheta=100, nlines=10, verbose=1)

# import time
#
# for nrho in [5,10,100,500]:
#     tick = time.time()
#     img = np.load("Images-npy/phonecalc256.npy")
#     houghedgeline(img, scale=5, edgethresh=5, gradmagnthreshold=5, nrho=nrho, ntheta=100, nlines=20, verbose=0)
#     tock = time.time()
#
#     print("nrho:", nrho, tock-tick)

# plt.figure()
# [h, w] = np.shape(img)
# rgb = np.zeros((h, w, 3), dtype=np.uint8)
# # rgb[:,:,0] = rgb[:,:,1] = rgb[:,:,2] = img
# (Y, X) = extractedges(img, scale=5, threshold=5)
# rgb[Y, X, 0] = 0
# rgb[Y, X, 1] = 0
# rgb[Y, X, 2] = 255
# plt.imshow(rgb)










