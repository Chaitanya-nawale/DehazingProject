#image fusing using wavelet transform.

import pywt
import cv2
import numpy as np

def upscaling(img,x,y,row,col) :

    upscaling_img = np.zeros((x*row,y*col),np.uint8)

    i, m = 0, 0

    while m < row :

        j, n = 0, 0
        while n < col:
            upscaling_img[i, j] = img[m, n]

            j += y

            n += 1

        m += 1

        i += x

    return upscaling_img




def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


# This function does the coefficient fusing according to the fusion method
def fuseCoeff(cooef1, cooef2, method):

    if (method == 'mean'):
        cooef = (cooef1 + cooef2) / 2
    elif (method == 'min'):
        cooef = np.minimum(cooef1,cooef2)
    elif (method == 'max'):
        cooef = np.maximum(cooef1,cooef2)
    else:
        cooef = []

    return cooef

def waveLetTrans(I1,I2):
	# Params
	FUSION_METHOD = 'mean' # Can be 'min' || 'max || anything you choose according theory

	# Read the two image
	
	print(I1.shape,I2.shape)
	
	# We need to have both images the same size
	#I2 = cv2.resize(I2,I1.shape) # I do this just because i used two random images

	## Fusion algo

	# First: Do wavelet transform on each image
	wavelet = 'db1'
	cooef1 = pywt.wavedec2(I1[:,:], wavelet)
	cooef2 = pywt.wavedec2(I2[:,:], wavelet)

	# Second: for each level in both image do the fusion according to the desire option
	fusedCooef = []
	for i in range(len(cooef1)-1):

		# The first values in each decomposition is the apprximation values of the top level
		if(i == 0):

			fusedCooef.append(fuseCoeff(cooef1[0],cooef2[0],FUSION_METHOD))

		else:

			# For the rest of the levels we have tupels with 3 coeeficents
			c1 = fuseCoeff(cooef1[i][0],cooef2[i][0],FUSION_METHOD)
			c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], FUSION_METHOD)
			c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], FUSION_METHOD)

			fusedCooef.append((c1,c2,c3))

	# Third: After we fused the cooefficent we nned to transfor back to get the image
	fusedImage = pywt.waverec2(fusedCooef, wavelet)

	# Forth: normmalize values to be in uint8
	fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
	fusedImage = fusedImage.astype(np.uint8)
	print(fusedImage.shape)
	#fusedImage = cv2.resize(fusedImage,(I1.shape[1],I1.shape[0]))
	# Fith: Show image
	print(fusedImage.shape, I1.shape, I2.shape)
	#cv2.imshow("win",fusedImage)
	#cv2.imwrite("final.jpg",fusedImage)
	return fusedImage

I1 = cv2.imread('C:\\Users\\ajink\\OneDrive\\Desktop\\PriorBased\\PriorOP\\PriorGC.png',1)
I2 = cv2.imread('C:\\Users\\ajink\\OneDrive\\Desktop\\Iteration1\\testDCPDNOP\\outdoor_hazy_7_DCPCN.png',1)

#I1 = cv2.normalize(I1.astype('float64'),None,0.0,1.0,cv2.NORM_MINMAX)
#I2 = cv2.normalize(I2.astype('float64'),None,0.0,1.0,cv2.NORM_MINMAX)

print(I1.shape,I2.shape)

I1B, I1G, I1R = cv2.split(I1)
I2B, I2G, I2R = cv2.split(I2)
IB = waveLetTrans(I1B, I2B)
IG = waveLetTrans(I1G, I2G)
IR = waveLetTrans(I1R, I2R)

finalImage = cv2.merge((IB,IG,IR))
row,col,plane = finalImage.shape

#sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
#sharpenedFinal = cv2.filter2D(finalImage, -1, sharpen_kernel)
'''sharpenedFinalB = upscaling(IB,2,2,row,col)
sharpenedFinalG = upscaling(IG,2,2,row,col)
sharpenedFinalR = upscaling(IR,2,2,row,col)

sharpenedFinal = cv2.merge((sharpenedFinalB,sharpenedFinalG,sharpenedFinalR))'''

#fin2 = np.hstack((I1,I2,sharpenedFinal))
cv2.imshow('Test',sharpenedFinal)
#cv2.imshow('Final',fin2)
cv2.waitKey()
cv2.destroyAllWindows()