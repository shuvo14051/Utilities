import os
import numpy as np
from PIL import Image

input_folder_path = "benign_png"

output_folder_path = "benign_final"

i = 1
    
for filename in os.listdir(input_folder_path):
    
    image = Image.open(input_folder_path+"/"+filename)
    image = image.convert("RGB")
    image.save("benign_final/file{}.png".format(i))
    i = i+1