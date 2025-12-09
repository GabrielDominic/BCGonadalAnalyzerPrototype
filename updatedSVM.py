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
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import cv2

# Load saved features
pick_in = open('completemalefeaturefile.pickle', 'rb')
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
    X, y, files, test_size=0.20, random_state=42, stratify=y
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95, svd_solver='full', random_state=42)),
    ('smote', SMOTE(random_state=42)),
    # ('MLP', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42))
    ('Random Forest', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, min_samples_split=2))
    # ('SVC', SVC (kernel='linear', C=1,  gamma='scale', random_state=42, probability=True))
])
# # # # #Scale
# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(xtrain)
# x_test_scaled = scaler.transform(xtest)

# # # # #PCA
# pca = PCA(n_components=0.95, svd_solver='full', random_state=42)
# pca.fit(x_train_scaled)
# xtrain_pca = pca.transform(x_train_scaled)
# xtest_pca = pca.transform(x_test_scaled)
# print("PCA n_components:", pca.n_components_)

categories = ['developing','maturing','spawning', 'spent']

# Train SVM
pipe.fit(xtrain, ytrain)

# # Save model
# with open('LR_model.pickle', 'wb') as pf:
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
idx = 3

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

#Calculating Feature Importance
print(f'Calculating Feature Importance')

num_features = xtrain.shape[1]

# 1. GLCM Features
# You iterate through ['contrast', 'homogeneity', 'energy', 'correlation']
# and calculate mean and std for each.
glcm_props = ['contrast', 'homogeneity', 'energy', 'correlation']
glcm_names = []
for prop in glcm_props:
    glcm_names.extend([f'glcm_{prop}_mean', f'glcm_{prop}_std'])

# 2. LBP Features
# P=24, method='uniform' results in P + 2 bins (patterns 0 to P+1)
P = 24
lbp_names = [f'lbp_bin_{i}' for i in range(P + 2)]

# 3. Color Moments
# cv2.imread loads in BGR format by default.
# Your loop passes the raw 'img' (BGR) to this function.
channels = ['B', 'G', 'R']
moments = ['mean', 'std', 'skew']
color_names = []
for channel in channels:
    for moment in moments:
        color_names.append(f'color_{channel}_{moment}')

# 4. Morphological Features
# The function returns exactly these three in order
morph_names = ['morph_area_foreground', 'morph_area_contour', 'morph_circularity']

# 5. Edge Features
# The function returns exactly these three in order
edge_names = ['edge_sobel_mean', 'edge_sobel_std', 'edge_canny_density']

# --- Combine All Feature Names ---
# This order matches your: np.hstack([glcm_feat, lbp_feat, cm_feat, morph_feat, edge])
feature_names = glcm_names + lbp_names + color_names + morph_names + edge_names

# Verification
print(f"Total Feature Names: {len(feature_names)}")
print("Feature Names List:", feature_names)
print(f'Number of features: {num_features}')

target_model = pipe
result = permutation_importance(
    target_model,
    xtest,
    ytest,
    n_repeats=10,
    random_state=42,
    n_jobs=1
)

sorted_idx = result.importances_mean.argsort()

plt.figure(figsize=(10,6))
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    tick_labels=[feature_names[i] for i in sorted_idx]
)
plt.title("Permutation Importances (Test Set")
plt.xlabel("Decrease in Accuracy Score")
plt.tight_layout()
plt.show()