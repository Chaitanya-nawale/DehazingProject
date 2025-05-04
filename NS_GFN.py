import cv2 as cv
import numpy as np
from Gray_World import GrayWorld
#from Ancuti_Color_Enhance import AncutiCE
from Gamma_Correction import GammaCorrect

def main():
	img = cv.imread('Test.png',1)
	total_shape_image = img.shape
	height = math.ceil(total_shape_image[1]/8)*8
	width = math.ceil(total_shape_image[0]/8)*8
	img_sized = cv.resize(img,(height,width))
	#img_bit_changed = 255*img
	#img_changed = (img_bit_changed.astype('uint8'))
	img_WB = GrayWorld(img,name='Test')
	#actual_WB = img_WB / 255 ;
	#img_CE = AncutiCE(img,name='Test')
	img_GC = GammaCorrect(img,name='Test')
	cv.imshow('Actual WB',img_WB)
	
	total_output = cv.addWeighted(img_WB,0.5,img_GC,0.5,1)
	cv.imwrite('Test_Revise.png',total_output)
	cv.imshow('Haze Removal',total_output)
	cv.waitKey(0)
	cv.destroyAllWindows()

main()