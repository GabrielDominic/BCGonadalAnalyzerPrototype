import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import cv2

# Load saved features
pick_in = open('femalefeaturefile.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()
print("Number of samples:", len(data))
if len(data) > 0:
    print("First feature shape:", np.array(data[0][0]).shape)
    print("First label:", data[0][1])

X = []
y = []
files = []

# random.shuffle(data)

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

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95, svd_solver='full', random_state=42)),
    ('SVC', SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True, class_weight='balanced'))
])
# # # # #Scale
# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(xtrain)
# x_test_scaled = scaler.transform(xtest)

# # # #PCA
# pca = PCA(n_components=0.95, svd_solver='full', random_state=42)
# pca.fit(x_train_scaled)
# xtrain_pca = pca.transform(x_train_scaled)
# xtest_pca = pca.transform(x_test_scaled)
# print("PCA n_components:", pca.n_components_)

categories = ['developing','maturing','spawning', 'spent']

# Train SVM
pipe.fit(xtrain, ytrain)

# # Save model
# with open('svm_model.pickle', 'wb') as pf:
#     pickle.dump(pipe, pf)

# Evaluate
print(f'List of test samples: {len(ytest)}')
for idx in range(len(ytest)):
    tlabel = categories[ytest[idx]]
    print(f"Test sample index: {idx} label: {tlabel}")

pred = pipe.predict(xtest)
acc = accuracy_score(ytest, pred)
print("Accuracy:", acc)
print("Prediction is: ", categories[pred[0]])

# # # # Show an example prediction (first test sample)
idx = 7

pred_label = pred[idx]
true_label = categories[ytest[idx]]
pred_file = ftest[idx]
print(f"Test sample: file={pred_file}, true={true_label}, pred={categories[int(pred_label)]}")

# # # #Verifying Consistency
print('Classification report:')
print(classification_report(ytest, pred))


# # # # Load and display the original image
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