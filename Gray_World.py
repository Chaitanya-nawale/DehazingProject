import cv2 as cv
import numpy as np
from PIL import Image
from Denormalization import DeNormalize

def GrayWorld(img,name='Haze'):
	#img2 = cv.resize(img,(0,0),fx=0.25,fy=0.25)
	img_norm = cv.normalize(img.astype('float64'),None,0.0,1.0,cv.NORM_MINMAX)
	b,g,r = cv.split(img_norm)
	avg_red = np.mean(r)
	avg_green = np.mean(g)
	avg_blue = np.mean(b)
	
	avgRGB = [avg_red,avg_green,avg_blue]
	
	total_gray_value = (avg_red+avg_green+avg_blue)/3
	
	scaleValue = []
	
	for i in range(0,len(avgRGB)):
		scaleValue.append(total_gray_value/(avgRGB[i]+0.001))
		
		
	R = scaleValue[0]*r 
	G = scaleValue[1]*g 
	B = scaleValue[2]*b
	
	#print(R)
	
	total_size = img.shape
	
	for i in range(0,total_size[0]):
		for j in range(0,total_size[1]):
			if R[i][j] > 255:
				R[i][j] = 255
			if G[i][j] > 255:
				G[i][j] = 255
			if B[i][j] > 255:
				B[i][j] = 255
				
	
	output_image = cv.merge((B,G,R)) ;
	#cv.imwrite(name+'_WB.png',output_image)
	#output_image2 = np.hstack((img_norm,output_image))
	#cv.imshow('GrayWorld WB',output_image2)
	cv.waitKey(0)
	cv.destroyAllWindows()
	return output_image
	
"""haha = GrayWorld(cv.imread('trees2.png',1))
haha_deNorm = DeNormalize(haha,0.0,1.0,0,255)
haha_rgb = cv.cvtColor(haha_deNorm, cv.COLOR_BGR2RGB)
haha_r,haha_g,haha_b = cv.split(haha_rgb)
img_pil = Image.fromarray(haha_rgb)
img_pil_r = Image.fromarray(haha_r)
img_pil_g = Image.fromarray(haha_g)
img_pil_b = Image.fromarray(haha_b)

average_shannon = (shannon_entropy(img_pil_r)+shannon_entropy(img_pil_g)+shannon_entropy(img_pil_b))/3

print(shannon_entropy(img_pil))
print(average_shannon)"""
