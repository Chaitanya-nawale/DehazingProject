import math
import cv2 as cv
import SSIM
import PSNR
from Denormalization import DeNormalize
import glob
from Entropy import Actual_entropy
import numpy as np

def _round(number):
	testMod1 = (number*10)%10
	numberTest = number*100
	testMod2 = numberTest%100
	returnVal = int(number)
	if testMod1>=4 and testMod2>=4:
		returnVal = math.ceil(number)
	else:
		returnVal = math.floor(number)
	return returnVal
	
def _imresize(img):
	#img = cv.resize(img,(0,0),fx=0.5,fy=0.5)
	total_shape = img.shape
	height,width = _round(total_shape[0]/256)*256,_round(total_shape[1]/256)*256
	#maxedOut = max(width,height)
	if height > width:
		img = cv.resize(img,(height,height))
	else:
		img = cv.resize(img,(width,height))
		
	'''if height/256 < 6 or width < 6:
		img = img
	elif 6 <= height/256 < 11 or 6 <= width/256 < 11:
		img = cv.resize(img,(0,0),fx=0.50,fy=0.50)
	else:
		img = cv.resize(img,(0,0),fx=0.50,fy=0.50)'''
	return img
	
outImageName = glob.glob('testIP\\*jpg')
#outImageName = ['E:\\IPCV Data\\O-HAZE\OHaze\\GT\\35_outdoor_GT.jpg']
'''newNames = []
for names in outImageName:
	if names not in outImageNameDone:
		newNames.append(names)'''
outImages = []

cutter = len('testIP\\')
print(len(outImageName))

for name in outImageName:
	img = cv.imread(name,1)
	#img = cv.resize(img,(0,0),fx=0.25,fy=0.25)
	imgResize = _imresize(img)
	outImages.append(imgResize)
	
index = 0
#checker = outImageName[index]
#print(checker[cutter:-4])
for i in range(len(outImages)):
	nameToCut = outImageName[i]
	cv.imwrite('testMov/'+nameToCut[cutter:-4]+'.jpg',outImages[i])
	print(index)
	index = index + 1