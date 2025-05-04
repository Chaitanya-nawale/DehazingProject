import math
import cv2 as cv
import SSIM
import PSNR
from Denormalization import DeNormalize
import glob
from Entropy import Actual_entropy

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
	total_shape = img.shape
	height,width = _round(total_shape[0]/256)*256,_round(total_shape[1]/256)*256
	img = cv.resize(img,(width,height))
	return img
	
DCPDNFusNameList = glob.glob('outputFus/*png')
hazyImageName = glob.glob('Inputs/*png')

#print(DCPDNFusNameList,hazyImageName)

print('PSNR(CV and Manual) and SSIM of DCPDN')

DCPDNFusImages = []
hazyImages = []

	
for nameHazy in hazyImageName:
	imgRead = cv.imread(nameHazy,1)
	imgRead = _imresize(imgRead)
	hazyImages.append(imgRead)
	
	
for nameDCPDNFus in DCPDNFusNameList:
	imgReadDCPDNFus = cv.imread(nameDCPDNFus,1)
	DCPDNFusImages.append(imgReadDCPDNFus)
	
totEntropy = [0]*8
totEntropyH = 0
index2 = 0
for indexA in range(len(hazyImages)):
	Entropy1H = Actual_entropy(hazyImages[indexA])
	print('Hazy')
	print(Entropy1H)
	totEntropyH = totEntropyH + Entropy1H
	print('------------------------------------------------------------')
	for index3 in range(8):
		print(index2)
		Entropy1 = Actual_entropy(DCPDNFusImages[index2])
		print(Entropy1)
		print('-------------------------------------------------------------')
		totEntropy[index3] = totEntropy[index3] + Entropy1
		index2 = index2 + 1

print(totEntropyH/len(hazyImages))

print('---------------------------------------------------------')
		
for test in range(8):
	print(totEntropy[test]/len(hazyImages))
	print('\n')
