import os
import cv2 as cv
import numpy as np
from Denormalization import DeNormalize
from Novel_Fus import Novel
from PIL import Image
import glob

onlyOutfiles = glob.glob('testMov\\*jpg')
fields=['Original','Gray World WB','Ancuti CE','Gamma Correct','Histogram Equalize','Fusion 1','Fusion 2','Fusion 3']
path_file = 'Entropy.csv'
print(len(onlyOutfiles))
actualOut_images = []
actual_field_value = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
list_of_all_imagesOut = []
actualIn_images = []
list_of_all_imagesIn = []
rootOut = 'testFusOP\\'

for name in onlyOutfiles:
	img = cv.imread(name,1)
	actualOut_images.append(img)


#print(len(actual_images))
print('imRead Complete')
for everymage in actualOut_images:
	returned_normalize = Novel(everymage)
	list_of_all_imagesOut.append([returned_normalize[6]])
print('EnOut Complete')
i = 0
for each_list in list_of_all_imagesOut:
	j=0
	nameToCut = onlyOutfiles[i]
	for each_image in each_list:
		img_write = DeNormalize(each_image,0.0,1.0,0,255)
		cv.imwrite('testFusOP/'+nameToCut[len(rootOut):-4]+'_'+str(j)+'.jpg',img_write)
		j=j+1
		print(i,j)
	i = i+1
print('WriteOut Complete')