import cv2 as cv
import numpy as np
import math

def AncutiCE(img,name='Haze'):
	#img2 = cv.resize(img,(0,0),fx=0.25,fy=0.25)
	total_shape_image = img.shape
	#print(total_shape_image)
	img_norm = cv.normalize(img.astype('float64'),None,0.0,1.0,cv.NORM_MINMAX)
	#print(img_norm)
	mean_image = np.mean(img_norm)
	#print(mean_image)
	mew = 2*(mean_image+0.5)
	#print(mew)
	sub_image = img_norm-mean_image
	output_image = mew * sub_image
	total_size = output_image.shape
	output_image_adjust = output_image
	for i in range(0,total_size[0]):
		for j in range(0,total_size[1]):
			for k in range(0,total_size[2]):
				if output_image[i][j][k] < 0.0:
					output_image_adjust[i][j][k] = 0.0
	#output_image_b,output_image_g,output_image_r = cv.split(output_image)
	#output_image_2 = cv.merge((output_image_b,output_image_g,output_image_r))
	#print(output_image_adjust)
	#output_image2 = np.hstack((img_norm,output_image))
	#cv.imwrite(name+'_CE.bmp',output_image.astype('uint8'))
	#cv.imshow('Ancuti CE',output_image_adjust)
	cv.waitKey(0)
	cv.destroyAllWindows()
	return output_image_adjust
	
#haha = AncutiCE(cv.imread('trees2.png',1))