# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %%
raw_img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

# %%
def Binarize(img):
    ret = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ret[i,j] =  0 if img[i,j] < 128 else 255
    return ret
        
def DownSample(img):
    ret = np.zeros(shape=(64,64))
    for i in range(64):
        for j in range(64):
            ret[i,j] = img[i*8,j*8]
    return ret

# %%
def Yokoi(img):
    def h(b,c,d,e):
        if b == c and (d != b or e != b):
            return 'q'
        if b == c and (d == b and e == b):
            return 'r'
        if b != c:
            return 's'
    def f(a1,a2,a3,a4):
        if(a1 == a2 and a2 == a3 and a3 == a4 and a4 == 'r'):
            return 5
        else:
            return [a1,a2,a3,a4].count('q')
        
    ret = np.zeros(shape=(64,64))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[row,col] == 255:
                x0 = img[row,col]
                x1 = img[row,col+1] if col+1 < img.shape[1] else 0
                x2 = img[row-1,col] if row-1 >= 0 else 0
                x3 = img[row,col-1] if col-1 >= 0 else 0
                x4 = img[row+1,col] if row+1 < img.shape[0] else 0
                x5 = img[row+1,col+1] if row+1 < img.shape[0] and col+1 < img.shape[1] else 0
                x6 = img[row-1,col+1] if row-1 >= 0 and col+1 < img.shape[1] else 0
                x7 = img[row-1,col-1] if row-1 >= 0 and col-1 >= 0 else 0
                x8 = img[row+1,col-1] if row+1 < img.shape[0] and col-1 >= 0 else 0
                a1 = h(x0,x1,x6,x2)
                a2 = h(x0,x2,x7,x3)
                a3 = h(x0,x3,x8,x4)
                a4 = h(x0,x4,x5,x1)
                ret[row,col] = f(a1,a2,a3,a4)
    return ret
        

# %%
img = Binarize(raw_img)
img = DownSample(img)
YokoiMatrix = Yokoi(img)

# %%
matrix_list = YokoiMatrix.astype(int).tolist()
for i in range(len(matrix_list)):
    for j in range(len(matrix_list[i])):
        if matrix_list [i][j] == 0:
            matrix_list[i][j] = ' '
        else :
            matrix_list[i][j] = str(matrix_list[i][j])

output = []
for line in matrix_list:
    line.append('\n')
    output.append(''.join(line))
with open('./output.txt','w') as f:
    f.writelines(output)


