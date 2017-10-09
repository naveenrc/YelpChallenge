import numpy as np
import os
from utilities import model as md
import scipy
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

path = os.path.dirname(__file__)
param_path = os.path.join(path, '../yelpData/') + 'parameters_4.npy'
params = np.load(param_path).item()
img_size = int(input('Enter image size, example x if (x,x): '))
photo_name = input('Enter image to be predict without extension(.png or jgp): ')
image_pt = os.path.join(path, '../yelpData/resized64/') + photo_name + '.png'
image = np.array(ndimage.imread(image_pt, flatten=False))
my_image = scipy.misc.imresize(image, size=(img_size,img_size)).\
            reshape((1, img_size*img_size*3)).T
my_image_prediction = md.predict(my_image, params)
"""
0     food  121267
1    drink    6587
2   inside   47799
3  outside   19565
4     menu    1060
5    other       0
"""
plt.imshow(image)
plt.show()
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
