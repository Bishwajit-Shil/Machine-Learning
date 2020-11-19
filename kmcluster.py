# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 23:22:00 2020

@author: JIt Shil
"""


import  numpy as np
from sklearn.cluster import KMeans
import  matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs 


x,y = make_blobs(n_samples = 1009 , centers = 5, random_state = 0, cluster_std = 2)
plt.scatter(x[:,0], x[:,1], s=73)
plt.show()


est = KMeans(7)
est.fit(x)
predictions = est.predict(x)


plt.scatter(x[:,0], x[:,1], c= predictions, s=75,  cmap = 'rainbow')
plt.show()

