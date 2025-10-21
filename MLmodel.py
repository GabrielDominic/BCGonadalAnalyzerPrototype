import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


            ## #Data Preparation
# dir = 'C:\\GitProjects\\BCGonadalAnalyzerPrototype\\imagedataset'
# categories = ['spawning', 'spent']

# data = []

# for category in categories:
#     path = os.path.join(dir, category)
#     label = categories.index(category)

#     for img in os.listdir(path):
#         img_path = os.path.join(path, img)
#         image_item = cv2.imread(img_path, 0)
#         try:    
#             image_item = cv2.resize(image_item, (64, 64))
#             image = np.array(image_item).flatten()

#             data.append([image, label])
#         except Exception as e:
#             pass

# print(len(data))

            # Saving Data
# pick_in = open('datafile.pickle', 'wb')
# pickle.dump(data, pick_in)
# pick_in.close()

            # Loading Data
pick_in = open('datafile.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

features = []
labels = []

random.shuffle(data)

for feature, label in data:
    features.append(feature)
    labels.append(label)

                #Training
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.25)

# model = SVC(C=1.0, kernel='poly', gamma='auto')
# model.fit(xtrain, ytrain)

                #Save the model
# pick = open('svm_model.pickle', 'wb')
# pickle.dump(model, pick)
# pick.close()

                #Testing the model
pick  = open('svm_model.pickle', 'rb')
model = pickle.load(pick)
pick.close()

prediction = model.predict(xtest)
accuracy = model.score(xtest, ytest)

categories = ['spawning', 'spent']

print("Accuracy:", accuracy)
print('Prediction is:', categories[prediction[0]])

gonadal_stage = xtest[0].reshape(64, 64)
plt.imshow(gonadal_stage, cmap='gray')
plt.show()