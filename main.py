# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:11:03 2019

@author: Jimmy
"""
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from scipy import signal, ndimage

def gaussian_smooth(img, kernel_size=5, sigma=5):
    kernel = np.zeros((kernel_size, kernel_size))
    (h,w,c) = img.shape[:3]

    # guassuan filter
    for y in range(kernel_size):
        for x in range(kernel_size):
            kernel[y,x] = 1/(2*math.pi*sigma)*math.exp(-(x**2+y**2)/(2*sigma**2))
    
    # scipy convolution
    img_smooth = np.zeros((h,w,c))
    for i in range(c):
        img_smooth[:,:,i] = signal.convolve2d(img[:,:,i], kernel, boundary='fill', mode='same')
    
    return img_smooth

            
def sobel_edge_detection(img):
    (h, w) = img.shape[:2]
    
    # sobel operator    
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[ 1, 2, 1],
                   [ 0, 0, 0],
                   [-1,-2,-1]]) 
    
    # scipy convolution
    img_gx = signal.convolve2d(img, Gx, boundary='fill', mode='same')
    img_gy = signal.convolve2d(img, Gy, boundary='fill', mode='same')
    
    # get gradient magnitude and direction
    grad_mag = np.sqrt(np.square(img_gx)+np.square(img_gy))
    grad_dir = np.arctan(img_gy/(img_gx+1e-5))
    
    return grad_mag, grad_dir, img_gx, img_gy

    
def structure_tensor(ix, iy):
    ixx = ix ** 2
    iyy = iy ** 2
    ixy = ix * iy
    
    return ixx, iyy, ixy


def nms(ix, iy, win_size=3, k=0.04, local_size=5, sigma=5):
    ixx, iyy, ixy= structure_tensor(ix, iy)
    rec_win = np.ones((win_size,win_size)) # rectangular window
    
    # guassian window
    guass_win = np.zeros((win_size, win_size))
    for y in range(win_size):
        for x in range(win_size):
            guass_win[y,x] = 1/(2*math.pi*sigma)*math.exp(-(x**2+y**2)/(2*sigma**2))    
    
    # compute component of harris matrix
    hxx = signal.convolve2d(ixx, guass_win, boundary='fill', mode='same')
    hyy = signal.convolve2d(iyy, guass_win, boundary='fill', mode='same')
    hxy = signal.convolve2d(ixy, guass_win, boundary='fill', mode='same')
    
    # harris measure
    det = hxx * hyy - hxy**2
    tr = hxx + hyy
    R = det - k * (tr**2) # k – empirical constant, k = 0.04-0.06

    
    mask1 = R > 0.0003 # max(np.mean(R)+0.3*np.std(R),0) # find the pixels having response larger than threshold
    R_max_filt = ndimage.maximum_filter(R, size=local_size) 
    mask2 = abs(R - R_max_filt) < 3e-5 # find the pixels which are local maximum
    corner = mask1 & mask2
    
    return R, corner


def rotate(img, angle=30, center=None, scale=1.0):
    (h, w) = img.shape[:2]
    
    if center is None:
        center = (w / 2, h / 2)
    
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h))

    return rotated


def resize(img,scale=0.5,inter=cv2.INTER_AREA):
    (h, w) = img.shape[:2]
    dim = (int(w*scale),int(h*scale))
    resized = cv2.resize(img, dim, interpolation=inter)

    return resized                 


if __name__ == '__main__':
    
    # parameters
    kernel_size = 10 # kernel size of Guassian blur
    win_size = 30 # window size of structure tensor
    local_size = 5 # window size of finding local maximum in nms
    ROTATE = False # whether to rotate
    RESIZE = False # whether to resize
    
    # read image
    img = cv2.imread('original.jpg') / 255.0
    cv2.imshow('original', img)
    
    # rotate image
    if ROTATE:
        img = rotate(img)

    # scale image
    if RESIZE:
        img = resize(img)    
    
    # gaussian smooth
    img_smooth = gaussian_smooth(img,kernel_size=kernel_size)
    cv2.imshow('guassian smooth', img_smooth)
    # cv2.imwrite('./results/gaussian_smooth_5.jpg', img_smooth * 255.0)
    
    # convert to gray-scale
    b, g, r = cv2.split(img_smooth)
    img_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    cv2.imshow('gray scale', img_gray)
    
    # sobel edge detection
    grad_mag, grad_dir, ix, iy = sobel_edge_detection(img_gray)
    cv2.imshow('gradient magnitude', grad_mag)
    # cv2.imwrite('./results/gradient_magnitude_5.jpg',grad_mag * 255.0)
    
    # generate colour gradient direction image
    grad_color_dir = np.where(grad_mag>0.1,grad_dir,0)
    grad_color_dir_maskn = np.where(grad_color_dir<0,1,0)
    grad_color_dir_maskp = np.where(grad_color_dir>0,1,0)
    grad_color_dir_n = np.interp(grad_color_dir,[np.min(grad_color_dir),np.max(np.where(grad_color_dir < 0, grad_color_dir, -np.inf))],[0.1,0.5])
    grad_color_dir_p = np.interp(grad_color_dir,[np.min(np.where(grad_color_dir > 0, grad_color_dir, np.inf)),np.max(grad_color_dir)],[0.6,1.0])
    grad_color_dir = grad_color_dir_n * grad_color_dir_maskn + grad_color_dir_p * grad_color_dir_maskp
    plt.imshow(grad_color_dir,cmap="gnuplot")
    # plt.imsave('./results/gradient_direction_5.jpg',grad_color_dir,cmap="gnuplot")
    
    cv2.imshow('gradient x direction', ix)
    # cv2.imwrite('./results/gradient_x_5.jpg',ix * 255.0)
    cv2.imshow('gradient y direction', iy)
    # cv2.imwrite('./results/gradient_y_5.jpg',iy * 255.0)
    
    # nms
    R, corner = nms(ix, iy, win_size=win_size, local_size=local_size)
    
    # show corner on image
    img_corner = img
    for index, x in np.ndenumerate(corner):
        if x:
            index = (index[1], index[0])
            cv2.circle(img_corner, index, 2, (0,0,255), -1)
    cv2.imshow('corner', img_corner)
    # cv2.imwrite('./results/corner_win=30.jpg',img_corner * 255.0)
    

    print("corner # = {}".format(np.count_nonzero(corner == True)))
    
        
    


    
