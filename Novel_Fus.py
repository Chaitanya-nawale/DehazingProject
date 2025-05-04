import cv2 as cv
import numpy as np
from Gray_World import GrayWorld
from Ancuti_Color_Enhance import AncutiCE
from Gamma_Correction import GammaCorrect
from HistEqual import histEqual
import math
from Denormalization import DeNormalize
from PIL import Image

def Novel(img,name="haze"):
	size = img.shape
	'''if size[0]>450 and size[1]>450:
		img = cv.resize(img,(0,0),fx=0.5,fy=0.5)'''
	img_wb = GrayWorld(img,name)
	img_gc = GammaCorrect(img,name)
	img_ce = AncutiCE(img,name)
	
	img_norm = cv.normalize(img.astype('float64'),None,0.0,1.0,cv.NORM_MINMAX)
	
	img_fus = img
	
	img_Equalized = histEqual(img_fus)
	img_equ = cv.normalize(img_Equalized.astype('float64'),None,0.0,1.0,cv.NORM_MINMAX)
	
	img_fus_both = 0.5*img_equ + 0.35*img_gc + 0.15*img_ce
	img_fus_wenqi = 0.33*img_wb + 0.33*img_gc + 0.33*img_ce
	img_fus_wenqi_revise = 0.35*img_gc + 0.25*img_ce + 0.1*img_wb + 0.30*img_equ	
	img_fus_both2 = np.hstack((img_norm,img_fus_wenqi_revise))
	
	#cv.imwrite('Fusion_By_Addition.png',DeNormalize(img_fus_wenqi_revise,0.0,1.0,0,255))
	#cv.imshow('Actual',img_fus_both2)
	cv.waitKey(0)
	cv.destroyAllWindows()
	'''entropy_actual = Actual_entropy(img_norm)
	entropy_wb = Actual_entropy(img_wb)
	entropy_ce = Actual_entropy(img_ce)
	entropy_gc = Actual_entropy(img_gc)'''
	#print(entropy_actual,entropy_wb,entropy_ce,entropy_gc)
	return img_norm,img_wb,img_ce,img_gc,img_equ,img_fus_both,img_fus_wenqi,img_fus_wenqi_revise
	
#img = cv.imread('Inputs/22_1.png',1)
#Novel(img,'Trees')