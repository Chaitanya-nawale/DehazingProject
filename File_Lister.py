import os
import glob
myPath = 'Inputs'
list_of_all = glob.glob(myPath+'/*png')
root = os.path.abspath(myPath)
print(list_of_all,root)
