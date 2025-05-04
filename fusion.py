import cv2
import SSIM
import glob
from Entropy import Actual_entropy
def matchDimensions(img1,img2):
    src1_shape =  img1.shape
    src2_shape =  img2.shape
    height =0
    width =0
    if(src1_shape[0]<src2_shape[0]):
        height = src1_shape[0]
    else:
        height = src2_shape[0]
    
    if(src1_shape[1]<src2_shape[1]):
        width = src1_shape[1]
    else:
        width = src2_shape[1]
    img1 = cv2.resize(img1,(width,height))
    img2 = cv2.resize(img2,(width,height))
    return img1,img2
    
if __name__ == '__main__':
	print('0: Reference Image Score\n1: Non-Reference Image Score')
	choice = int(input('Enter the Choice : '))
	counter = 0
	if choice == 0:
		src1Names = glob.glob("PriorBasedOutput\\*.jpg")
		src2Names = glob.glob("DataOrientedOutput\\*.jpg")
		cutter = len("PriorBasedOutput\\")
		gt = glob.glob("GroundTruth\\*.jpg")
		#print(gt)
		#print(len(src1Names),len(src2Names),len(gt))
		for src1,src2,gt1 in zip(src1Names,src2Names,gt):
			imgPrior = cv2.imread(src1,1)
			imgData = cv2.imread(src2,1)
			imgGT = cv2.imread(gt1,1)
			print(imgPrior.shape)
			print(imgData.shape)
			print(imgGT.shape)
			imgPrior,imgData = matchDimensions(imgPrior,imgData)
			imgPrior,imgGT = matchDimensions(imgPrior,imgGT)
			
			ssim1 = SSIM.calculate_ssim(imgGT, imgPrior)
			ssim2 = SSIM.calculate_ssim(imgGT, imgData)
			ssim_diff = abs(ssim2 - ssim1)
			ssim_total = ssim1 + ssim2
			final_SSIM = cv2.addWeighted(imgPrior,ssim1/ssim_total,imgData,ssim2/ssim_total,0)
			#print(final.shape)
			cv2.imwrite('Final_Fusion\\'+src1[cutter:-4]+'.jpg',final_SSIM)
			counter = counter + 1
	elif choice == 1:
		counter = 0
		src1Names = glob.glob("PriorBasedOutput\\*.jpg")
		src2Names = glob.glob("DataOrientedOutput\\*.jpg")
		cutter = len("PriorBasedOutput\\")
		#print(gt)
		#print(len(src1Names),len(src2Names),len(gt))
		for src1,src2 in zip(src1Names,src2Names):
			imgPrior = cv2.imread(src1,1)
			imgData = cv2.imread(src2,1)
			print(imgPrior.shape)
			print(imgData.shape)
			imgPrior,imgData = matchDimensions(imgPrior,imgData)
			
			entropy1 = Actual_entropy(imgPrior)
			entropy2 = Actual_entropy(imgData)
			entropy_diff = abs(entropy2 - entropy1)
			entropy_total = entropy1 + entropy2
			final_entropy = cv2.addWeighted(imgPrior,entropy1/entropy_total,imgData,entropy2/entropy_total,0)
			#print(final.shape)
			cv2.imwrite('Final_Fusion\\'+src1[cutter:-4]+'.jpg',final_entropy)
			counter = counter + 1
	else:
		print("Wrong Choice...")
	print("done")