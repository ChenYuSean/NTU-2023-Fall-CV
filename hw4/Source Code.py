# %%

import cv2
import numpy as np
import matplotlib.pyplot as plt

# %%
img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i,j] =  0 if img[i,j] < 128 else 255
kernel = [[0, 1, 1, 1, 0],
          [1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1],
          [0, 1, 1, 1, 0]]
kernel = np.asarray(kernel,dtype=np.int32)

# %%
def dilation(img, kernel):
    res_img = np.zeros(shape=img.shape,dtype=np.uint8)
    width = kernel.shape[0]
    height = kernel.shape[1]
    center = (height//2,width//2)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            res_img[row,col] = img[row,col]
            if img[row,col] != 255 :
                continue
            # kernel
            for krow in range(kernel.shape[0]):
                for kcol in range(kernel.shape[1]):
                    if kernel[krow,kcol] != 1:
                        continue
                    x = row + krow-center[0]
                    y = col + kcol-center[1]
                    if x < 0 or y < 0 or x >= img.shape[0] or y >= img.shape[1]:
                        continue
                    res_img[x,y] = 255
                    
    return res_img
                        
                
dilation_img = dilation(img,kernel)
cv2.imshow("Dilation",dilation_img)
cv2.waitKey()
cv2.imwrite("Dilation.png",dilation_img)

# %%
def erosion(img, kernel):
    res_img = np.ones(shape=img.shape,dtype=np.uint8)
    width = kernel.shape[0]
    height = kernel.shape[1]
    center = (height//2,width//2)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            res_img[row,col] = img[row,col]
            # kernel
            val = 255
            for krow in range(kernel.shape[0]):
                for kcol in range(kernel.shape[1]):
                    if kernel[krow,kcol] != 1:
                        continue
                    x = row + krow-center[0]
                    y = col + kcol-center[1]
                    if x < 0 or y < 0 or x >= img.shape[0] or y >= img.shape[1]:
                        val = 0
                        break
                    if img[x,y] == 0:
                        val = 0
                        break
            res_img[row,col] = val
                    
    return res_img
                        
                
erosion_img = erosion(img,kernel)
cv2.imshow("Erosion",erosion_img)
cv2.waitKey()
cv2.imwrite("Erosion.png",erosion_img)

# %%
opening_img = dilation(erosion_img,kernel)
cv2.imshow("Opening",opening_img)
cv2.waitKey()
cv2.imwrite("Opening.png",opening_img)
closing_img = erosion(dilation_img,kernel)
cv2.imshow("Closing",closing_img)
cv2.waitKey()
cv2.imwrite("Closing.png",closing_img)

# %%
J = [[0, 0, 0],
     [1, 1, 0],
     [0, 1, 0]]
K = [[0, 1, 1],
     [0, 0, 1],
     [0, 0, 0]]
J = np.asarray(J,dtype=np.uint8)
K = np.asarray(K,dtype=np.uint8)

def hit_and_miss(img,ker1,ker2):
     inv_img = np.zeros(shape=img.shape)
     res_img = np.zeros(shape=img.shape)
     for i in range(img.shape[0]):
          for j in range(img.shape[1]):
               inv_img[i,j] = 0 if img[i,j] == 255 else 255
     A = erosion(img,ker1)
     B = erosion(inv_img,ker2)
     for row in range(img.shape[0]):
          for col in range(img.shape[1]):
               if (A[row,col] == 255 and B[row,col] == 255):
                    res_img[row,col] = 255
     return res_img

HMimg = hit_and_miss(img,J,K)
cv2.imshow("Hit and Miss",HMimg)
cv2.waitKey()
cv2.imwrite("Hit and Miss.png",HMimg)

# %%



