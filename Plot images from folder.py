import os
import matplotlib.pyplot as plt

folder_path = 'dataset\image'
image_list = cl1
rows = 6
cols = 5

fig, axs = plt.subplots(rows, cols, figsize=(12, 8))
fig.suptitle('Red Rose')

for i, img_name in enumerate(image_list):
    row = i // cols
    col = i % cols
    
    img_path = os.path.join(folder_path, img_name)
    img = plt.imread(img_path)
    
    axs[row, col].imshow(img)
    axs[row, col].axis('off')
plt.show()
