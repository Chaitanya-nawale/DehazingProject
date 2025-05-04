import os
import numpy as np
import cv2 as cv
import math
import glob
import shutil
import stat

movDir = 'testMov\\'
fusDir = 'testFusOP\\'
dcpdnIPDir = 'testDCPDNIP\\'
dcpdnOPDir = 'testDCPDNOP\\'
priorDir = 'PriorBasedOutput\\'
dataDir = 'DataOrientedOutput\\'

os.mkdir(movDir)
os.mkdir(fusDir)
os.mkdir(dcpdnIPDir)
os.mkdir(dcpdnOPDir)
os.mkdir(priorDir)
os.mkdir(dataDir)

os.system("python zFinal.py")
sourceDir = 'testDCPDNOP\\'
destDir = 'DataOrientedOutput\\'
destDir2 = 'ActualDataOriented\\'
fileName = os.listdir(sourceDir)
fileName = fileName[0]
shutil.copy(os.path.join(sourceDir,fileName),destDir)

shutil.copy(os.path.join(sourceDir,fileName),destDir2)

os.system("python try.py")
sourceDir = 'PriorBasedOutput\\'
destDir = 'ActualPriorBased\\'
fileName = os.listdir(sourceDir)[0]
shutil.copy(os.path.join(sourceDir,fileName),destDir)

os.system("python fusion.py")

rootFolder = 'C:\\Users\\ajink\\OneDrive\\Desktop\\Final\\'

shutil.rmtree(movDir)
shutil.rmtree(fusDir)
shutil.rmtree(dcpdnIPDir)
shutil.rmtree(dcpdnOPDir)
shutil.rmtree(priorDir)
shutil.rmtree(dataDir)

'''for movName,fusName,dcpdnIPName,dcpdnOPName,priorName,dataName in zip(movDir,fusDir,dcpdnIPDir,dcpdnOPDir,priorDir,dataDir):
	os.chmod(movName,stat.S_IWRITE)
	os.chmod(fusName,stat.S_IWRITE)
	os.chmod(dcpdnIPName,stat.S_IWRITE)
	os.chmod(dcpdnOPName,stat.S_IWRITE)
	os.chmod(priorName,stat.S_IWRITE)
	os.chmod(dataName,stat.S_IWRITE)
	os.remove(rootFolder+movName)
	os.remove(rootFolder+fusName)
	os.remove(rootFolder+dcpdnIPName)
	os.remove(rootFolder+dcpdnOPName)
	os.remove(rootFolder+priorName)
	os.remove(rootFolder+dataName)'''

