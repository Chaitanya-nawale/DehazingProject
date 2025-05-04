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
	total_shape = img.shape
	height,width = _round(total_shape[0]/256)*256,_round(total_shape[1]/256)*256
	img = cv.resize(img,(width,height))
	return img
	
GTNameList = glob.glob('GroundTruth/IH/*jpg')
FinalNameList = glob.glob('PriorBasedOutput/IH/*jpg')

print(len(GTNameList),len(FinalNameList))

print('PSNR(CV and Manual) and SSIM of DCPDN')

GTImages = []
FinalImages = []
totPSNR1 = 0
totPSNR2 = 0
totSSIM = 0


for name in GTNameList:
	imgRead = cv.imread(name,1)
	#imgRead = _imresize(imgRead)
	GTImages.append(imgRead)

	
for nameFinal in FinalNameList:
	imgFinal = cv.imread(nameFinal,1)
	FinalImages.append(imgFinal)
	
totPSNR1Fus = 0
totPSNR2Fus = 0
totSSIMFus = 0
totEntropy = 0
for indexA in range(len(GTImages)):
	PSNR1 = cv.PSNR(GTImages[indexA],FinalImages[indexA])
	PSNR2 = PSNR.PSNR(GTImages[indexA],FinalImages[indexA])
	SSIM1 = SSIM.calculate_ssim(GTImages[indexA],FinalImages[indexA])
	Entropy1 = Actual_entropy(FinalImages[indexA])
	print('Hazy')
	print(PSNR1,PSNR2,SSIM1,Entropy1)
	totPSNR1 = totPSNR1 + PSNR1
	totPSNR2 = totPSNR2 + PSNR2
	totSSIM = totSSIM + SSIM1
	totEntropy = totEntropy + Entropy1
	print('------------------------------------------------------------')
	

print(totPSNR1/len(GTImages),totPSNR2/len(GTImages),totSSIM/len(GTImages),totEntropy/len(GTImages))

print('---------------------------------------------------------')
		

