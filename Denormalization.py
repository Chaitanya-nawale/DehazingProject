import cv2 as cv
import numpy as np
import math

def DeNormalize(image,from_min,from_max,to_min,to_max):
	from_range = from_max-from_min
	to_range = to_max-to_min
	converted = np.array((image-from_min)/float(from_range),dtype="float")
	actual_convert = to_min+(converted*to_range)
	total_shape = actual_convert.shape
	for i in range(0,total_shape[0]):
		for j in range(0,total_shape[1]):
			for k in range(0,total_shape[2]):
				if actual_convert[i][j][k] < 0:
					actual_convert[i][j][k] = 0
				if actual_convert[i][j][k] > 255:
					actual_convert[i][j][k] = 255
	#print(actual_convert)
	return actual_convert.astype('uint8')


