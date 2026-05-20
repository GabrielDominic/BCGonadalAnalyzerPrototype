import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score,  make_scorer, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

categories = ['developing', 'mature', 'spawning', 'spent']

# Load saved features

data = []

# files = ['completefemalefeaturefile.pickle', 'femalefeaturefile.pickle']

# for fname in files:
#     with open(fname, 'rb') as f:
#         data.extend(pickle.load(f))  # combine datasets

pick_in = open('FemaleFeatures[Balanced].pickle', 'rb')
# pick_in = open('CTGAFmaleupdatedFeatures.pickle', 'rb')
# pick_in = open('finalfemalefeaturefile.pickle', 'rb')
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
    ('smote', SMOTE(random_state=42)),
    ('pca', PCA (random_state=42)),
    ('SVC', SVC (random_state=42, probability=True))
])

rf_model = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42))
])

knn_model = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('pca', PCA (random_state=42)),
    ('KNN', KNeighborsClassifier())
])

lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('pca', PCA (random_state=42)),
    ('Logistic Regression', LogisticRegression(random_state=42))
])

mlp_model = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('pca', PCA (random_state=42)),
    ('MLP', MLPClassifier(random_state=42))
])

gb_model = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42))
])

xgboost_model = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('XGBoost', xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'))
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

#Gradient Boosting
gb_param_grid = {
    'Gradient Boosting__n_estimators': [100, 200, 300],
    'Gradient Boosting__learning_rate': [0.01, 0.1, 0.2],
    'Gradient Boosting__max_depth': [3, 5, 7],
}

gb_grid = GridSearchCV(gb_model, gb_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
gb_grid.fit(xtrain, ytrain)

print("Best GB params: ",gb_grid.best_params_)

#XGBoost
xgb_param_grid = {
    'XGBoost__n_estimators': [100, 200, 300],
    'XGBoost__learning_rate': [0.01, 0.1, 0.2],
    'XGBoost__max_depth': [3, 5, 7],
}
xgb_grid = GridSearchCV(xgboost_model, xgb_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
xgb_grid.fit(xtrain, ytrain)

print("Best XGB params: ",xgb_grid.best_params_)

#Overwrite models with best estimators from grid search
best_svc_model = svc_grid.best_estimator_
best_rf_model = rf_grid.best_estimator_
best_knn_model = knn_grid.best_estimator_
best_lr_model = lr_grid.best_estimator_
best_mlp_model = mlp_grid.best_estimator_
best_gb_model = gb_grid.best_estimator_
best_xgb_model = xgb_grid.best_estimator_

print("Best SVC CV score:", svc_grid.best_score_)
print("Best RF CV score:", rf_grid.best_score_)
print("Best KNN CV score:", knn_grid.best_score_)
print("Best LR CV score:", lr_grid.best_score_)
print("Best MLP CV score:", mlp_grid.best_score_)
print("Best GB CV score:", gb_grid.best_score_)
print("Best XGB CV score:", xgb_grid.best_score_)


#Evaluate on test set
def test_evaluate(name, model):
    y_pred = model.predict(xtest)
    
    print(f"\n{name} Test Performance:")
    print("Accuracy:", accuracy_score(ytest, y_pred))
    f1 = f1_score(ytest, y_pred, average='weighted', zero_division=0)
    print("F1 Score (weighted):", f1)
    print(classification_report(ytest, y_pred, target_names=categories))

test_evaluate("SVC", best_svc_model)
test_evaluate("Random Forest", best_rf_model)
test_evaluate("KNN", best_knn_model)
test_evaluate("Logistic Regression", best_lr_model)
test_evaluate("MLP", best_mlp_model)
test_evaluate("Gradient Boosting", best_gb_model)
test_evaluate("XGBoost", best_xgb_model)

with open('best_xgb_model_F(Balanced).pickle', 'wb') as f:
    pickle.dump(best_xgb_model, f)

print("XGBoost model saved to best_xgb_model_M.pickle")

with open('best_gradient_boosting_model_F(Balanced).pickle', 'wb') as f:
    pickle.dump(best_gb_model, f)
print("Gradient Boosting model saved to best_gradient_boosting_model_F.pickle")

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
ConfusionMatrixDisplay.from_estimator(best_gb_model, xtest, ytest, display_labels=categories, cmap=plt.cm.Blues)
plt.title("Gradient Boosting Confusion Matrix")
ConfusionMatrixDisplay.from_estimator(best_xgb_model, xtest, ytest, display_labels=categories, cmap=plt.cm.Blues)
plt.title("XGBoost Confusion Matrix")
plt.show()


import pandas as pd

# feature importance from the XGBoost step of the pipeline
importances = best_xgb_model.named_steps['XGBoost'].feature_importances_

# list of feature names in the order they are concatenated
glcm_names = ['contrast_mean', 'contrast_std', 'homogeneity_mean', 'homogeneity_std', 
              'energy_mean', 'energy_std', 'correlation_mean', 'correlation_std']

# Define P for LBP features
P = 24
lbp_names = [f'lbp_bin_{i}' for i in range(P + 2)]  # 26 bins
cm_names = ['R_mean', 'R_std', 'R_skew', 'G_mean', 'G_std', 'G_skew', 'B_mean', 'B_std', 'B_skew']
morph_names = ['area_foreground', 'area_contour', 'circularity']
edge_names = ['sobel_mean', 'sobel_std', 'edge_density']
gamete_names = ['total_tissue_pixels', 'gamete_pixels', 'area_fraction']

# feature_names = glcm_names + lbp_names + morph_names + edge_names + gamete_names
feature_names = glcm_names + lbp_names + cm_names + morph_names + edge_names + gamete_names

feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print(feature_df.sort_values(by='Importance', ascending=False))