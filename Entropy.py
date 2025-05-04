import math
import cv2 as cv
import numpy as np
from PIL import Image
def shannon_entropy(img):
	#img = Image.open(path)
	histogram = img.histogram()
	histogram_length = sum(histogram)

	samples_probability = [float(h) / histogram_length for h in histogram]

	return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])

def Actual_entropy(img_cv):
	img_b,img_g,img_r = cv.split(img_cv)
	entropy_b = shannon_entropy(Image.fromarray(img_b))
	entropy_g = shannon_entropy(Image.fromarray(img_g))
	entropy_r = shannon_entropy(Image.fromarray(img_r))
	
	total_entropy = (entropy_b+entropy_g+entropy_r)/3
	
	return total_entropy

'''original = cv.imread('C:/Users/ajink/OneDrive/Desktop/Gated_Fusion_Network/inputs/PipingLive.png',cv.IMREAD_COLOR)
compared = cv.imread('C:/Users/ajink/OneDrive/Desktop/Gated_Fusion_Network/results/PipingLive_dehazed.png',cv.IMREAD_COLOR)	
print(Actual_entropy(original))
print(Actual_entropy(compared))'''