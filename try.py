import cv2;
import math;
import numpy as np;
import glob
from skimage.color import rgb2hsv

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def getBrightChannels(input,patch_size = 15):
	b ,g, r = cv2.split(input)
	maxChannels = cv2.max(cv2.max(r,g),b)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(patch_size, patch_size))
	brightChannels = cv2.dilate(maxChannels, kernel)
	return brightChannels
	
def AtmLightDCP(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def AtmLight(im):
	
    #parameters
    alpha = 10
    gamma = 0.4
    
    rows = len(im)
    columns = len(im[0])
    #print(rows)
    #print(columns)
    
    hsv_img = rgb2hsv(im)
    v_img = hsv_img[:, :, 2]
    #print(v_img)
    S1 = [x for sublist in v_img for x in sublist]
    H1= S1[:]
    S1.sort(reverse = True)
    
    #print(S1[(int)(len(S1)/alpha)])
    min_val_ind = (int)(len(S1)/alpha)
    max_val_ind = (int)(min_val_ind*gamma)
    #print(min_val_ind)
    #print(max_val_ind)
    min_val = S1[min_val_ind]
    max_val = S1[max_val_ind]
    #print(min_val)
    #print(max_val)
    
    L1 = [i for i in range(len(H1)) if H1[i] >= min_val and H1[i] < max_val]
    #print(L1)
    S2 = [H1[i] for i in L1]
    S2.sort(reverse = True)
    #print(S2)
    #print(len(S2))
    #print(len(L1))
    
    max_elem = max(S2)
    #print(max_elem)
    L2 = [i for i in L1 if H1[i] == max_elem]
    #print(L2)
    
    A = np.zeros((1,3))
    
    rs = 1
    gs = 1
    bs = 1
    for close in L2:
        k = im[(int)(close/columns),close%columns,:]
        '''if(rs<k[2]):
            rs =k[2]
        if(gs<k[1]):
            gs =k[1]
        if(bs<k[0]):
            bs =k[0]'''
        rs += k[2]
        gs += k[1]
        bs += k[0]
        '''if(rs>k[2]):
            rs =k[2]
        if(gs>k[1]):
            gs =k[1]
        if(bs>k[0]):
            bs =k[0]'''
    rs /= len(L2)
    gs /= len(L2)
    bs /= len(L2)
    A[0, 2] = rs
    A[0, 1] = gs
    A[0, 0] = bs
    #print(A)
	
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission
	
def getBrightTransEstimate(input, atmosLight, patch_size = 15):
	b1,g1,r1 = cv2.split(input)
	temp = cv2.max(cv2.max((b1-atmosLight[0][0])/(1-atmosLight[0][0]),(g1-atmosLight[0][1])/(1-atmosLight[0][1])),(r1-atmosLight[0][2])/(1-atmosLight[0][2]))
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size,patch_size))
	brightTransMap = cv2.dilate(temp, kernel)
	return brightTransMap	

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

def GammaCorrect(img,name='Haze'):
	alpha = 10
	gamma = 0.5
	#img2 = cv2.resize(img,(0,0),fx=0.25,fy=0.25)
	total_shape = img.shape
	test_image = img
	#test_image = cv2.normalize(img.astype('float64'),None,0.0,1.0,cv2.NORM_MINMAX)
	test_image_2 = img.astype('float64') 
	for i in range(0,total_shape[0]):
		for j in range(0,total_shape[1]):
			for k in range(0,total_shape[2]):
				test_image_2[i][j][k] = test_image[i][j][k]**gamma
				
	#test_image_3 = cv2.normalize(img.astype('float64'),None,0.0,255.0,cv2.NORM_MINMAX)
	#print(test_image_3)
	output_image = np.copy(test_image_2)
	output_image2 = np.hstack((test_image,output_image))
	
	#cv.imshow('Gamma Correct',output_image2)
	#cv.imwrite("GC_OP.jpg",output_image*255)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return output_image
	
def Clahe(im, clip):
	#im = im*255
	im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit = 5) 
	final_img = clahe.apply(im) + 30  
	#final_img = final_img/255
	return final_img

def i2f(i_image):
    f_image = np.float32(i_image)/255.0
    return f_image

def f2i(f_image):
    i_image = np.uint8(f_image*255.0)
    return i_image
if __name__ == '__main__':
    counter = 0
    cutter = len('testMov\\')
    for img in glob.glob("testMov\\*.jpg"):
        #fn = "F:\\Datasets\\Datasets\\# O-HAZY NTIRE 2018\\hazy\\14_outdoor_hazy.jpg"
        
        src = cv2.imread(img);
        #src = cv2.imread("Thepade_Code\\3.jpg")
        
        I = src.astype('float64')/255;
        dark = DarkChannel(I,15);
        #bright = getBrightChannels(I);
        #A = AtmLightDCP(I,dark);
        A = AtmLight(I);
        teDark = TransmissionEstimate(I,A,1);
        teBright = getBrightTransEstimate(I,A,1);
        td = TransmissionRefine(src,teDark);
        tb = TransmissionRefine(src,teBright);
        t = cv2.addWeighted(td,0.995,tb,0.005,0);
        J = Recover(I,t,A,0.1);
        J_GC = GammaCorrect(J);
        nameToCut = img
       #cv2.imwrite("PriorOP\\NHOP\\"+nameToCut[cutter:-4]+'.png',J*255);
        cv2.imwrite("PriorBasedOutput\\"+nameToCut[cutter:-4]+'_GC'+'.jpg',J_GC*255);
        #cv2.imwrite("check_Cl.jpg",J_CLAHE*255);
        counter+=1
        print("{0} done.".format(counter))
    print("Done")