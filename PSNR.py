import numpy as np
import math
import cv2 as cv

def PSNR(original,compared):
	originalNorm = cv.normalize(original.astype('float64'),None,0.0,1.0,cv.NORM_MINMAX)
	comparedNorm = cv.normalize(compared.astype('float64'),None,0.0,1.0,cv.NORM_MINMAX)
	mse = np.mean((originalNorm-comparedNorm)**2)
	if mse == 0:
		return 100
	Pixel_max = 1.0
	return 20*math.log10(Pixel_max/math.sqrt(mse))
	
'''original = cv.imread('Inputs/85.png',cv.IMREAD_COLOR)
compared = cv.imread('Outputs/2_8.png',cv.IMREAD_COLOR)	
print(PSNR(original,compared))
print(cv.PSNR(original,compared))'''