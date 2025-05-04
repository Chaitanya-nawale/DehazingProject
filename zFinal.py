import os
import numpy as np
import cv2 as cv
import math
import glob

#os.system("python hello.py")

os.system("python mover.py")

os.system("python Output_Create.py")

os.system("python generate_testsample.py")

os.system("python demo.py --dataroot testDCPDNIP --valDataroot testDCPDNIP --netG demo_model\\netG1.pth")



print("Success")