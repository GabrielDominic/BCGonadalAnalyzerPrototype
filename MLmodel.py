import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dir = ''
categories = ['developing', 'spawning']

data = []

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image_item = cv2.imread(img_path, 0)
        image_item = cv2.resize(image_item, (64, 64))
        image = np.array(image_item).flatten()

        data.append([image, label])
        


cv2.waitKey(0)
cv2.destroyAllWindows()