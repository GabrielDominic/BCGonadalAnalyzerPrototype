import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import cv2
import random

# Load saved features
pick_in = open('mfeaturefile.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()
print("Number of samples:", len(data))
if len(data) > 0:
    print("First feature shape:", np.array(data[0][0]).shape)
    print("First label:", data[0][1])

X = []
y = []
files = []

random.shuffle(data)

for feature, label, filenames in data:
    feature = feature.reshape(-1)
    X.append(feature)
    y.append(label)
    files.append(filenames)

X = np.vstack(X)
y = np.array(y, dtype=np.int32)

print("Loaded X.shape:", X.shape, "y.shape:", y.shape)

# Train/test split
xtrain, xtest, ytrain, ytest, ftrain, ftest = train_test_split(
    X, y, files, test_size=0.25, random_state=42, stratify=y
)

categories = ['spawning', 'spent']

# Baseline fast classifier
model = LinearSVC(max_iter=10000)
model.fit(xtrain, ytrain)

# Save model
with open('svm_model.pickle', 'wb') as pf:
    pickle.dump(model, pf)

# Evaluate
pred = model.predict(xtest)
acc = accuracy_score(ytest, pred)
print("Accuracy:", acc)
print("Prediction is: ", categories[pred[0]])

# Show an example prediction (first test sample)
idx = 0
pred_label = pred[idx]
true_label = categories[ytest[idx]]
pred_file = ftest[idx]
print(f"Test sample: file={pred_file}, true={true_label}, pred={categories[int(pred_label)]}")

#Verifying Consistency
print('Classification report:')
print(classification_report(ytest, pred_label))


# Load and display the original image
img = cv2.imread(pred_file, cv2.IMREAD_COLOR)
if img is not None:
    img = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f"True: {true_label}  Pred: {categories[int(pred_label)]}")
    plt.axis('off')
    plt.show()
else:
    print("Could not load image for display:", pred_file)


# consistency = (model.score(xtest, ytest) == (pred == ytest).mean())
# print('Score equals mean(y_pred == y_test)?', consistency)
# print()