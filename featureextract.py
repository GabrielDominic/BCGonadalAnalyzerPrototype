import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.preprocessing import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np
import cv2
import pickle

#loading Model
model = VGG16()
#Restructure
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
print(model.summary())

dir = 'C:\\GitProjects\\BCGonadalAnalyzerPrototype\\imagedataset'
categories = ['spawning', 'spent']

data = []
features = []
labels = []

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
    
        try:    
            image_item = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image_item = cv2.resize(image_item, (224, 224))

            image_item = cv2.cvtColor(image_item, cv2.COLOR_BGR2RGB)
            image = np.asarray(image_item, dtype=np.float32)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            feature = model.predict(image, verbose=0)
            feature = feature.flatten()

            data.append([feature, label])

        except Exception as e:
            print(f'Failed to process {img_path}: {e}')

print(f'Features Extracted: {len(data)}')

            # Saving Data
pick_in = open('featurefile.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()

        