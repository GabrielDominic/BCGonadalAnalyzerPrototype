import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score,  make_scorer, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
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

# Train/test split
xtrain, xtest, ytrain, ytest, ftrain, ftest = train_test_split(
    X, y, files, test_size=0.20, random_state=42, stratify=y
)

print("Class distribution:", dict(zip(*np.unique(ytrain, return_counts=True))))

svc_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA (random_state=42)),
    ('smote', SMOTE(random_state=42)),
    ('SVC', SVC (random_state=42, probability=True))
])

rf_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA (random_state=42)),
    ('smote', SMOTE(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42))
])

knn_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA (random_state=42)),
    ('smote', SMOTE(random_state=42)),
    ('KNN', KNeighborsClassifier())
])

lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA (random_state=42)),
    ('smote', SMOTE(random_state=42)),
    ('Logistic Regression', LogisticRegression(random_state=42))
])

mlp_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA (random_state=42)),
    ('smote', SMOTE(random_state=42)),
    ('MLP', MLPClassifier(random_state=42))
])

#Hyperparameter tuning
print("Starting Hyperparameter Tuning...")

#SVC
svc_param_grid = {
    'pca__n_components': [0.90, 0.95, 0.99],
    'SVC__C': [0.1, 1, 10],
    'SVC__kernel': ['linear', 'rbf'],
    'SVC__gamma': ['scale', 'auto']
}

svc_grid = GridSearchCV(
    estimator=svc_model,
    param_grid=svc_param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)

svc_grid.fit(xtrain, ytrain)

print("Best SVC params:", svc_grid.best_params_)

#Random Forest
rf_param_grid = {
    'Random Forest__n_estimators': [100, 200, 300],
    'Random Forest__max_depth': [None, 10, 20, 30],
    'Random Forest__min_samples_split': [2, 5, 10],
    'pca__n_components': [0.90, 0.95, 0.99]
}

rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
rf_grid.fit(xtrain, ytrain)

print("Best RF params: ",rf_grid.best_params_)

#KNN
knn_param_grid = {
    'KNN__n_neighbors': [3, 5, 7, 9, 11],
    'KNN__weights': ['uniform', 'distance'],
    'pca__n_components': [0.90, 0.95, 0.99]
}

knn_grid = GridSearchCV(knn_model, knn_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
knn_grid.fit(xtrain, ytrain)

print("Best KNN params: ",knn_grid.best_params_)

#Logistic Regression
lr_param_grid = {
    'Logistic Regression__C': [0.01, 0.1, 1, 10],
    'Logistic Regression__max_iter': [500, 1000, 2000],
    'Logistic Regression__penalty': ['l2'],
    'pca__n_components': [0.90, 0.95, 0.99]
}

lr_grid = GridSearchCV(lr_model, lr_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
lr_grid.fit(xtrain, ytrain)

print("Best LR params: ",lr_grid.best_params_)

#MLP
mlp_param_grid = {
    'MLP__hidden_layer_sizes': [(32,), (64,), (64,32), (128,64)],
    'MLP__max_iter': [500, 1000, 2000],
    'MLP__early_stopping': [True],
    'MLP__activation': ['relu', 'tanh'],
    'MLP__alpha': [0.0001, 0.001, 0.01],
    'MLP__learning_rate_init': [0.001, 0.01],
    'pca__n_components': [0.90, 0.95]
}

mlp_grid = GridSearchCV(mlp_model, mlp_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
mlp_grid.fit(xtrain, ytrain)

print("Best MLP params: ",mlp_grid.best_params_)

# # Train SVM
# svc_model.fit(xtrain, ytrain)
# rf_model.fit(xtrain, ytrain)
# knn_model.fit(xtrain, ytrain)
# lr_model.fit(xtrain, ytrain)
# mlp_model.fit(xtrain, ytrain)

# Save model
# with open('svm_model.pickle', 'wb') as pf:
#     pickle.dump(pipe, pf)

#Overwrite models with best estimators from grid search
best_svc_model = svc_grid.best_estimator_
best_rf_model = rf_grid.best_estimator_
best_knn_model = knn_grid.best_estimator_
best_lr_model = lr_grid.best_estimator_
best_mlp_model = mlp_grid.best_estimator_

##EVALUATION
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
precision_weighted = make_scorer(precision_score, average='weighted', zero_division=0)
# recall_macro = make_scorer(recall_score, average='macro', zero_division=0)
# f1_weighted = make_scorer(f1_score, average='weighted', zero_division=0)

def evaluate_model(name, model, xtrain, ytrain, kf):
    print(f"Evaluating {name}...")
    acc = cross_val_score(model, xtrain, ytrain, cv=kf, scoring='accuracy')
    precision = cross_val_score(model, xtrain, ytrain, cv=kf, scoring=precision_weighted)
    recall_macro = cross_val_score(model, xtrain, ytrain, cv=kf, scoring='recall_macro')
    f1 = cross_val_score(model, xtrain, ytrain, cv=kf, scoring='f1_weighted')
    roc_auc = cross_val_score(model, xtrain, ytrain, cv=kf, scoring='roc_auc_ovr_weighted')

    print(f"{name} Cross-validation scores:")
    print(f"\tAccuracy          : {acc} \tmean: {np.mean(acc):.2f} standard deviation: {np.std(acc):.2f}")
    print(f"\tPrecision         : {precision} \tmean: {np.mean(precision):.2f} standard deviation: {np.std(precision):.2f}")
    print(f"\tRecall            : {recall_macro} \tmean: {np.mean(recall_macro):.2f} standard deviation: {np.std(recall_macro):.2f}")
    print(f"\tF1                : {f1} \tmean: {np.mean(f1):.2f} standard deviation: {np.std(f1):.2f}")
    print(f"\tROC_AUC           : {roc_auc} \tmean: {np.mean(roc_auc):.2f} standard deviation: {np.std(roc_auc):.2f}")

evaluate_model("SVC", best_svc_model, xtrain, ytrain, kf)
evaluate_model("Random Forest", best_rf_model, xtrain, ytrain, kf)
evaluate_model("KNN", best_knn_model, xtrain, ytrain, kf)
evaluate_model("Logistic Regression", best_lr_model, xtrain, ytrain, kf)
evaluate_model("MLP", best_mlp_model, xtrain, ytrain, kf)

#Confusion Matrix
ConfusionMatrixDisplay.from_estimator(best_svc_model, xtest, ytest, display_labels=categories, cmap=plt.cm.Blues)
plt.title("SVC Confusion Matrix")
ConfusionMatrixDisplay.from_estimator(best_rf_model, xtest, ytest, display_labels=categories, cmap=plt.cm.Blues)
plt.title("Random Forest Confusion Matrix")
ConfusionMatrixDisplay.from_estimator(best_knn_model, xtest, ytest, display_labels=categories, cmap=plt.cm.Blues)
plt.title("KNN Confusion Matrix")
ConfusionMatrixDisplay.from_estimator(best_lr_model, xtest, ytest, display_labels=categories, cmap=plt.cm.Blues)
plt.title("Logistic Regression Confusion Matrix")
ConfusionMatrixDisplay.from_estimator(best_mlp_model, xtest, ytest, display_labels=categories, cmap=plt.cm.Blues)
plt.title("MLP Confusion Matrix")
plt.show()