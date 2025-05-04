import math
import numpy as np
import cv2 as cv
import glob

def compute_dark_channel(img,patch_size=15):
    b, g, r = cv.split(img)
    bgr_min_img = cv.min(cv.min(b, g), r)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (patch_size,patch_size))
    dark_channel_img = cv.erode(bgr_min_img, kernel)
    return dark_channel_img
	
	
def averageIllumination(image):

	area = image.shape[0] * image.shape[1]
	rgb_image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
	hsv_image = cv.cvtColor(rgb_image,cv.COLOR_RGB2HSV)
	sumBright = np.sum(hsv_image[:,:,2])
	
	illumin = sumBright / area
	return illumin
def classifyy(image):

	avg = averageIllumination(image)
	if avg > 105:
		return 1
	else:
		return 0

	return

def main():
	count = 0
	for i in glob.glob("images\\Day\\*jpg"):
		img = cv.imread(i)
		#darkChannels = compute_dark_channel(img,patch_size=15)
		count += classifyy(img)
		#cv.imshow("Dark Channels", darkChannels)
		#cv.waitKey(0)
		#cv.destroyAllWindows()
	print((count*100)/200)
if __name__ == '__main__':
	main()