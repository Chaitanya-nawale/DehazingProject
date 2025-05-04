import cv2 as cv
import numpy as np


def GammaCorrect(img,name='Haze'):
	alpha = 1
	gamma = 2.5
	#img2 = cv.resize(img,(0,0),fx=0.25,fy=0.25)
	total_shape = img.shape
	test_image = cv.normalize(img.astype('float64'),None,0.0,1.0,cv.NORM_MINMAX)
	test_image_2 = img.astype('float64') 
	for i in range(0,total_shape[0]):
		for j in range(0,total_shape[1]):
			for k in range(0,total_shape[2]):
				test_image_2[i][j][k] = test_image[i][j][k]**gamma
				
	test_image_3 = cv.normalize(img.astype('float64'),None,0.0,255.0,cv.NORM_MINMAX)
	#print(test_image_3)
	output_image = np.copy(test_image_2)
	output_image2 = np.hstack((test_image,output_image))
	
	#cv.imshow('Gamma Correct',output_image2)
	cv.waitKey(0)
	cv.destroyAllWindows()
	return output_image
	
#haha = GammaCorrect(cv.imread('Inputs/15_1.png',1))
#print(haha)