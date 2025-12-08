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
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt

categories = ['developing', 'maturing', 'spawning', 'spent']

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

for feature, label, filenames in data:
    feature = feature.reshape(-1)
    X.append(feature)
    y.append(label)
    files.append(filenames)

X = np.vstack(X)
y = np.array(y, dtype=np.int32)

print("Loaded X.shape:", X.shape, "y.shape:", y.shape)

unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))

# Train/test split
xtrain, xtest, ytrain, ytest, ftrain, ftest = train_test_split(
    X, y, files, test_size=0.20, random_state=42, stratify=y
)

# # #Scale
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(xtrain)
x_test_scaled = scaler.transform(xtest)

# #PCA
pca = PCA(n_components=0.95, svd_solver='full', random_state=42)
pca.fit(x_train_scaled)
xtrain_pca = pca.transform(x_train_scaled)
xtest_pca = pca.transform(x_test_scaled)
print("PCA n_components:", pca.n_components_)

svc_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA (n_components=0.95, svd_solver='full', random_state=42)),
    ('SVC', SVC (kernel='linear', C=1, random_state=42, probability=True, class_weight='balanced'))
])

rf_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA (n_components=0.95, svd_solver='full', random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'))
])

knn_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA (n_components=0.95, svd_solver='full', random_state=42)),
    ('KNN', KNeighborsClassifier(n_neighbors=5))
])

lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA (n_components=0.95, svd_solver='full', random_state=42)),
    ('Logistic Regression', LogisticRegression(max_iter=200, random_state=42, class_weight='balanced'))
])

mlp_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA (n_components=0.95, svd_solver='full', random_state=42)),
    ('MLP', MLPClassifier(hidden_layer_sizes=(1000), max_iter=1000, random_state=42))
])

# # Train SVM
# svc_model.fit(xtrain, ytrain)
# rf_model.fit(xtrain, ytrain)
# knn_model.fit(xtrain, ytrain)
# lr_model.fit(xtrain, ytrain)
# mlp_model.fit(xtrain, ytrain)

# Save model
# with open('svm_model.pickle', 'wb') as pf:
#     pickle.dump(pipe, pf)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Accuracy Scores
svc_accuracy = cross_val_score(svc_model, xtrain, ytrain, cv=kf, scoring='accuracy')
rf_accuracy = cross_val_score(rf_model, xtrain, ytrain, cv=kf, scoring='accuracy')
knn_accuracy = cross_val_score(knn_model, xtrain, ytrain, cv=kf, scoring='accuracy')
lr_accuracy = cross_val_score(lr_model, xtrain, ytrain, cv=kf, scoring='accuracy')
mlp_accuracy = cross_val_score(mlp_model, xtrain, ytrain, cv=kf, scoring='accuracy')

#Precision scores
svc_precision = cross_val_score(svc_model, xtrain, ytrain, cv=kf, scoring='precision_weighted')
rf_precision = cross_val_score(rf_model, xtrain, ytrain, cv=kf, scoring='precision_weighted')
knn_precision = cross_val_score(knn_model, xtrain, ytrain, cv=kf, scoring='precision_weighted')
lr_precision = cross_val_score(lr_model, xtrain, ytrain, cv=kf, scoring='precision_weighted')
mlp_precision = cross_val_score(mlp_model, xtrain, ytrain, cv=kf, scoring='precision_weighted')

#Recall Scores
svc_recall = cross_val_score(svc_model, xtrain, ytrain, cv=kf, scoring='recall_macro')
rf_recall = cross_val_score(rf_model, xtrain, ytrain, cv=kf, scoring='recall_macro')
knn_recall = cross_val_score(knn_model, xtrain, ytrain, cv=kf, scoring='recall_macro')
lr_recall = cross_val_score(lr_model, xtrain, ytrain, cv=kf, scoring='recall_macro')
mlp_recall = cross_val_score(mlp_model, xtrain, ytrain, cv=kf, scoring='recall_macro')

#F1 Scores
svc_f1 = cross_val_score(svc_model, xtrain, ytrain, cv=kf, scoring='f1_weighted')
rf_f1 = cross_val_score(rf_model, xtrain, ytrain, cv=kf, scoring='f1_weighted')
knn_f1 = cross_val_score(knn_model, xtrain, ytrain, cv=kf, scoring='f1_weighted')
lr_f1 = cross_val_score(lr_model, xtrain, ytrain, cv=kf, scoring='f1_weighted')
mlp_f1 = cross_val_score(mlp_model, xtrain, ytrain, cv=kf, scoring='f1_weighted')

#roc_auc scores
svc_roc_auc = cross_val_score(svc_model, xtrain, ytrain, cv=kf, scoring='roc_auc_ovr')
rf_roc_auc = cross_val_score(rf_model, xtrain, ytrain, cv=kf, scoring='roc_auc_ovr')
knn_roc_auc = cross_val_score(knn_model, xtrain, ytrain, cv=kf, scoring='roc_auc_ovr')
lr_roc_auc = cross_val_score(lr_model, xtrain, ytrain, cv=kf, scoring='roc_auc_ovr')
mlp_roc_auc = cross_val_score(mlp_model, xtrain, ytrain, cv=kf, scoring='roc_auc_ovr')

print("Cross-validation scores:")
print("Accuracy:")
print(f"\tSVC accuracy          : {svc_accuracy} \tmean: {np.mean(svc_accuracy)}")
print(f"\tRandom Forest accuracy: {rf_accuracy} \tmean: {np.mean(rf_accuracy)}")
print(f"\tKNN accuracy          : {knn_accuracy} \tmean: {np.mean(knn_accuracy)}")
print(f"\tLogistic Regression accuracy: {lr_accuracy} \tmean: {np.mean(lr_accuracy)}")
print(f"\tMLP accuracy          : {mlp_accuracy} \tmean: {np.mean(mlp_accuracy)}")
print("Precision:")
print(f"\tSVC                   : {svc_precision} \tmean: {np.mean(svc_precision)}")
print(f"\tRandom Forest         : {rf_precision} \tmean: {np.mean(rf_precision)}")
print(f"\tKNN                   : {knn_precision} \tmean: {np.mean(knn_precision)}")
print(f"\tLogistic Regression   : {lr_precision} \tmean: {np.mean(lr_precision)}")
print(f"\tMLP                   : {mlp_precision} \tmean: {np.mean(mlp_precision)}")
print("Recall:")
print(f"\tSVC                   : {svc_recall} \tmean: {np.mean(svc_recall)}")
print(f"\tRandom Forest         : {rf_recall} \tmean: {np.mean(rf_recall)}")
print(f"\tKNN                   : {knn_recall} \tmean: {np.mean(knn_recall)}")
print(f"\tLogistic Regression   : {lr_recall} \tmean: {np.mean(lr_recall)}")
print(f"\tMLP                   : {mlp_recall} \tmean: {np.mean(mlp_recall)}")
print("F1:")
print(f"\tSVC                   : {svc_f1} \tmean: {np.mean(svc_f1)}")
print(f"\tRandom Forest         : {rf_f1} \tmean: {np.mean(rf_f1)}")
print(f"\tKNN                   : {knn_f1} \tmean: {np.mean(knn_f1)}")
print(f"\tLogistic Regression   : {lr_f1} \tmean: {np.mean(lr_f1)}")
print(f"\tMLP                   : {mlp_f1} \tmean: {np.mean(mlp_f1)}")
print("ROC_AUC:")
print(f"\tSVC                   : {svc_roc_auc} \tmean: {np.mean(svc_roc_auc)}")
print(f"\tRandom Forest         : {rf_roc_auc} \tmean: {np.mean(rf_roc_auc)}")
print(f"\tKNN                   : {knn_roc_auc} \tmean: {np.mean(knn_roc_auc)}")
print(f"\tLogistic Regression   : {lr_roc_auc} \tmean: {np.mean(lr_roc_auc)}")
print(f"\tMLP                   : {mlp_roc_auc} \tmean: {np.mean(mlp_roc_auc)}")