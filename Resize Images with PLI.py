import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from tensorflow.keras.datasets.mnist import load_data
from PIL import Image


(x_train, y_train), (x_test, y_test) = load_data()
x_train_zeros = x_train[y_train==9][:500]

for i in range(x_train_zeros.shape[0]):
    img = Image.fromarray(x_train_zeros[i])
    img = img.resize((256, 256))
    img = np.array(img)
    img = np.stack([img]*3, axis=-1) # stack the 3 color channels
    Image.fromarray(img).save(f"znine_{i}.png")



