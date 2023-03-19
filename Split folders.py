import splitfolders

import os
print(os.listdir())

"""
Breast-Cancer-ROI-original this is the base directory
Inside it there are two folders which are the classes 
These classes will be splitted into train test folders
ratio=(.8, .1, .1) will create three folders called train, test, val
output="Breast-Cancer-ROI-splited" this is the folder where all the files will be copied
if don't use any value for output there will be a folder called output.
"""

path = "Breast-Cancer-ROI-original/"

splitfolders.ratio(path, output="Breast-Cancer-ROI-splited",seed=1337, ratio=(.8, .2))