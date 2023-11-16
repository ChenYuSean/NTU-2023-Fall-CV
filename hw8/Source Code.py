# %%
import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt

# %%
raw_img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
os.makedirs('a', exist_ok=True)
os.makedirs('b', exist_ok=True)
os.makedirs('c', exist_ok=True)
os.makedirs('d', exist_ok=True)
os.makedirs('e', exist_ok=True)

# %%
def GaussNoise(img, mean, sigma, amp):
    ret = np.zeros_like(img)
    noise = np.random.normal(mean, sigma, img.shape)
    for r in range(ret.shape[0]):
        for c in range(ret.shape[1]):
            ret[r,c] = img[r,c] + amp * noise[r,c]
            ret[r,c] = np.clip(ret[r,c], 0, 255)
            
    return ret.astype(np.uint8)

def SaltAndPepperNoise(img, prob):
    ret = np.zeros_like(img)
    noise = np.random.random(img.shape)
    for r in range(ret.shape[0]):
        for c in range(ret.shape[1]):
            ret[r,c] = 0 if noise[r,c] < prob else 255 if noise[r,c] > 1-prob else img[r,c]
    return ret.astype(np.uint8)

# %%
def box_filter(img, size):
    ret = np.zeros_like(img)
    kernel = []
    for i in range(-(size[0]//2),(size[0]//2)+1):
        for j in range(-(size[1]//2),(size[1]//2)+1):
            kernel.append([i,j])
    weight = 1/(size[0]*size[1]) 

    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            sum = 0
            for k in kernel:
                if r+k[0] < 0 or r+k[0] >= img.shape[0] or c+k[1] < 0 or c+k[1] >= img.shape[1]:
                    break
                else:
                    sum = sum + img[r+k[0],c+k[1]]
            ret[r,c] = sum * weight
    return ret.astype(np.uint8)

def median_filter(img, size):
    ret = np.zeros_like(img)
    kernel = []
    for i in range(-(size[0]//2),(size[0]//2)+1):
        for j in range(-(size[1]//2),(size[1]//2)+1):
            kernel.append([i,j])
    weight = 1/(size[0]*size[1]) 

    for r in range(img.shape[1]):
        for c in range(img.shape[0]):
            val = []
            for k in kernel:
                if r+k[0] < 0 or r+k[0] >= img.shape[0] or c+k[1] < 0 or c+k[1] >= img.shape[1]:
                    break
                else:
                    val.append(img[r+k[0],c+k[1]])
            ret[r,c] = np.median(val)
    return ret.astype(np.uint8)

# %%
class Topology:
    kernel = [[0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]]
    kernel = np.asarray(kernel,dtype=np.int32)

    def dilation(img, kernel):
        res_img = np.zeros(shape=img.shape,dtype=np.uint8)
        width = kernel.shape[0]
        height = kernel.shape[1]
        center = (height//2,width//2)
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                res_img[row,col] = img[row,col]
                # kernel
                max_val = 0
                for krow in range(kernel.shape[0]):
                    for kcol in range(kernel.shape[1]):
                        if kernel[krow,kcol] != 1:
                            continue
                        x = row + krow-center[0]
                        y = col + kcol-center[1]
                        if x < 0 or y < 0 or x >= img.shape[0] or y >= img.shape[1]:
                            continue
                        max_val = img[x,y] if img[x,y] > max_val else max_val
                res_img[row,col] = max_val
        return res_img

    def erosion(img, kernel):
        res_img = np.ones(shape=img.shape,dtype=np.uint8)
        width = kernel.shape[0]
        height = kernel.shape[1]
        center = (height//2,width//2)
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                res_img[row,col] = img[row,col]
                # kernel
                min_val = 255
                for krow in range(kernel.shape[0]):
                    for kcol in range(kernel.shape[1]):
                        if kernel[krow,kcol] != 1:
                            continue
                        x = row + krow-center[0]
                        y = col + kcol-center[1]
                        if x < 0 or y < 0 or x >= img.shape[0] or y >= img.shape[1]:
                            min_val = 0
                            break
                        min_val = img[x,y] if img[x,y] < min_val else min_val
                res_img[row,col] = min_val
                        
        return res_img
    def opening(img, kernel):
        return Topology.dilation(Topology.erosion(img,kernel),kernel)
    def closing(img, kernel):
        return Topology.erosion(Topology.dilation(img,kernel),kernel)

# %%
def SNR(signal, processed):
    # Noramlize
    signal = signal.astype(np.float64)
    processed = processed.astype(np.float64)
    for i in range(signal.shape[1]):
        for j in range(signal.shape[0]):
            signal[i,j] = signal[i,j] / 255
            processed[i,j] = processed[i,j] / 255
            
    signal_mean = 0.0
    signal_var = 0.0
    noise_mean = 0.0
    noise_var = 0.0

    # Mean
    for i in range(signal.shape[1]):
        for j in range(signal.shape[0]):
            signal_mean += signal[i,j]
            noise_mean += processed[i,j] - signal[i,j]
    signal_mean /= signal.shape[0]*signal.shape[1]
    noise_mean /= signal.shape[0]*signal.shape[1]
    
    # Variance
    for i in range(signal.shape[1]):
        for j in range(signal.shape[0]):
            signal_var += (signal[i,j] - signal_mean)**2
            noise_var += (processed[i,j] - signal[i,j] - noise_mean)**2
    signal_var /= signal.shape[0]*signal.shape[1]
    noise_var /= signal.shape[0]*signal.shape[1]        

    return 20*math.log10(math.sqrt(signal_var)/math.sqrt(noise_var))

# %%
gauss_10 = GaussNoise(raw_img, 0, 1, 10)
gauss_30 = GaussNoise(raw_img, 0, 1, 30)
sp_005 = SaltAndPepperNoise(raw_img, 0.05)
sp_010 = SaltAndPepperNoise(raw_img, 0.10)
plt.imsave('a/GaussNoise_10.png',gauss_10, cmap='gray')
plt.imsave('a/GaussNoise_30.png',gauss_30, cmap='gray')
plt.imsave('b/SaltAndPepper_0.05.png',sp_005, cmap='gray')
plt.imsave('b/SaltAndPepper_0.10.png',sp_010, cmap='gray')

# %%
box_3_a1 = box_filter(gauss_10,(3,3))
box_3_a2 = box_filter(gauss_30,(3,3))
box_3_b1 = box_filter(sp_005,(3,3))
box_3_b2 = box_filter(sp_010,(3,3))
plt.imsave('c/BoxFilter_3_a1.png',box_3_a1, cmap='gray')
plt.imsave('c/BoxFilter_3_a2.png',box_3_a2, cmap='gray')
plt.imsave('c/BoxFilter_3_b1.png',box_3_b1, cmap='gray')
plt.imsave('c/BoxFilter_3_b2.png',box_3_b2, cmap='gray')

# %%
box_5_a1 = box_filter(gauss_10,(5,5))
box_5_a2 = box_filter(gauss_30,(5,5))
box_5_b1 = box_filter(sp_005,(5,5))
box_5_b2 = box_filter(sp_010,(5,5))
plt.imsave('c/BoxFilter_5_a1.png',box_5_a1, cmap='gray')
plt.imsave('c/BoxFilter_5_a2.png',box_5_a2, cmap='gray')
plt.imsave('c/BoxFilter_5_b1.png',box_5_b1, cmap='gray')
plt.imsave('c/BoxFilter_5_b2.png',box_5_b2, cmap='gray')

# %%
med_3_a1 = median_filter(gauss_10,(3,3))
med_3_a2 = median_filter(gauss_30,(3,3))
med_3_b1 = median_filter(sp_005,(3,3))
med_3_b2 = median_filter(sp_010,(3,3))
plt.imsave('d/MedianFilter_3_a1.png',med_3_a1, cmap='gray')
plt.imsave('d/MedianFilter_3_a2.png',med_3_a2, cmap='gray')
plt.imsave('d/MedianFilter_3_b1.png',med_3_b1, cmap='gray')
plt.imsave('d/MedianFilter_3_b2.png',med_3_b2, cmap='gray')

# %%
med_5_a1 = median_filter(gauss_10,(5,5))
med_5_a2 = median_filter(gauss_30,(5,5))
med_5_b1 = median_filter(sp_005,(5,5))
med_5_b2 = median_filter(sp_010,(5,5))
plt.imsave('d/MedianFilter_5_a1.png',med_5_a1, cmap='gray')
plt.imsave('d/MedianFilter_5_a2.png',med_5_a2, cmap='gray')
plt.imsave('d/MedianFilter_5_b1.png',med_5_b1, cmap='gray')
plt.imsave('d/MedianFilter_5_b2.png',med_5_b2, cmap='gray')

# %%
open_close_a1 = Topology.closing(Topology.opening(gauss_10,Topology.kernel),Topology.kernel)
open_close_a2 = Topology.closing(Topology.opening(gauss_30,Topology.kernel),Topology.kernel)
open_close_b1 = Topology.closing(Topology.opening(sp_005,Topology.kernel),Topology.kernel)
open_close_b2 = Topology.closing(Topology.opening(sp_010,Topology.kernel),Topology.kernel)
plt.imsave('e/OpenClose_a1.png',open_close_a1, cmap='gray')
plt.imsave('e/OpenClose_a2.png',open_close_a2, cmap='gray')
plt.imsave('e/OpenClose_b1.png',open_close_b1, cmap='gray')
plt.imsave('e/OpenClose_b2.png',open_close_b2, cmap='gray')

# %%
close_open_a1 = Topology.opening(Topology.closing(gauss_10,Topology.kernel),Topology.kernel)
close_open_a2 = Topology.opening(Topology.closing(gauss_30,Topology.kernel),Topology.kernel)
close_open_b1 = Topology.opening(Topology.closing(sp_005,Topology.kernel),Topology.kernel)
close_open_b2 = Topology.opening(Topology.closing(sp_010,Topology.kernel),Topology.kernel)
plt.imsave('e/CloseOpen_a1.png',close_open_a1, cmap='gray')
plt.imsave('e/CloseOpen_a2.png',close_open_a2, cmap='gray')
plt.imsave('e/CloseOpen_b1.png',close_open_b1, cmap='gray')
plt.imsave('e/CloseOpen_b2.png',close_open_b2, cmap='gray')

# %%
print("SNR of GaussNoise_10: ",SNR(raw_img,gauss_10))
print("SNR of GaussNoise_30: ",SNR(raw_img,gauss_30))
print("SNR of SaltAndPepper_0.05: ",SNR(raw_img,sp_005))
print("SNR of SaltAndPepper_0.10: ",SNR(raw_img,sp_010))
print("SNR of BoxFilter_3_a1: ",SNR(raw_img,box_3_a1))
print("SNR of BoxFilter_3_a2: ",SNR(raw_img,box_3_a2))
print("SNR of BoxFilter_3_b1: ",SNR(raw_img,box_3_b1))
print("SNR of BoxFilter_3_b2: ",SNR(raw_img,box_3_b2))
print("SNR of BoxFilter_5_a1: ",SNR(raw_img,box_5_a1))
print("SNR of BoxFilter_5_a2: ",SNR(raw_img,box_5_a2))
print("SNR of BoxFilter_5_b1: ",SNR(raw_img,box_5_b1))
print("SNR of BoxFilter_5_b2: ",SNR(raw_img,box_5_b2))
print("SNR of MedianFilter_3_a1: ",SNR(raw_img,med_3_a1))
print("SNR of MedianFilter_3_a2: ",SNR(raw_img,med_3_a2))
print("SNR of MedianFilter_3_b1: ",SNR(raw_img,med_3_b1))
print("SNR of MedianFilter_3_b2: ",SNR(raw_img,med_3_b2))
print("SNR of MedianFilter_5_a1: ",SNR(raw_img,med_5_a1))
print("SNR of MedianFilter_5_a2: ",SNR(raw_img,med_5_a2))
print("SNR of MedianFilter_5_b1: ",SNR(raw_img,med_5_b1))
print("SNR of MedianFilter_5_b2: ",SNR(raw_img,med_5_b2))
print("SNR of OpenClose_a1: ",SNR(raw_img,open_close_a1))
print("SNR of OpenClose_a2: ",SNR(raw_img,open_close_a2))
print("SNR of OpenClose_b1: ",SNR(raw_img,open_close_b1))
print("SNR of OpenClose_b2: ",SNR(raw_img,open_close_b2))
print("SNR of CloseOpen_a1: ",SNR(raw_img,close_open_a1))
print("SNR of CloseOpen_a2: ",SNR(raw_img,close_open_a2))
print("SNR of CloseOpen_b1: ",SNR(raw_img,close_open_b1))
print("SNR of CloseOpen_b2: ",SNR(raw_img,close_open_b2))


