import cv2 as cv
import numpy as np


def histEqual(img,name="hazy"):

	img_b,img_g,img_r = cv.split(img)
	
	imgR_hist = cv.equalizeHist(img_r)
	imgG_hist = cv.equalizeHist(img_g)
	imgB_hist = cv.equalizeHist(img_b)
	
	output_image = cv.merge((imgB_hist,imgG_hist,imgR_hist))
	
	#cv.imshow('HistEqual',output_image)
	cv.waitKey(0)
	cv.destroyAllWindows()
	return output_image
	
'''img_actual = cv.imread('Inputs/14_1.png',1)
output_actual = histEqual(img_actual)'''