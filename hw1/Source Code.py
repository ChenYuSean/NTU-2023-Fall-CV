# %%
import cv2
import matplotlib.pyplot as plt

# %%
img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

cv2.imshow('Original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
upside_down = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
for i in range(img.shape[1]):
    for j in range(img.shape[0]):
        upside_down[i,j] = img[-1-i,j]
cv2.imshow('Upside-down',upside_down)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Upside-down.png',upside_down)

# %%
left_side_right = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
for i in range(img.shape[1]):
    for j in range(img.shape[0]):
        left_side_right[i,j] = img[i,-1-j]
cv2.imshow('Left-side-right',left_side_right)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Left-side-right.png',left_side_right)

# %%
diagonal_flip = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
for i in range(img.shape[1]):
    for j in range(img.shape[0]):
        diagonal_flip[i,j] = img[j,i]
cv2.imshow('Diagonal Flip',diagonal_flip)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Diagonal Flip.png',diagonal_flip)


