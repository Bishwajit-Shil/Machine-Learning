# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:57:39 2020

@author: JIt Shil
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA



digits= load_digits()
#print(digits)

x_digits, y_digits = digits.data, digits.target

images_and_label= list(zip(digits.images, digits.target))


for index, (image, label) in enumerate(images_and_label[:6]):
    plt.subplot(2,3 , index + 1 )
    plt.imshow(image, cmap= plt.cm.gray_r , interpolation= 'nearest')
    plt.title('Target % i' % label)
    
plt.show()    


estimator = PCA(n_components= 10)
x_pca= estimator.fit_transform(x_digits)

color= ['black','blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'gray', 'gray']

for i in range(len(color)):
    px = x_pca[:, 0][y_digits == i]
    py = x_pca[:, 1][y_digits == i]
    plt.scatter(px, py, c=color[i])
    plt.legend(digits.target_names)
    plt.xlabel('First principle component')
    plt.ylabel('Second principle component')


plt.show()
