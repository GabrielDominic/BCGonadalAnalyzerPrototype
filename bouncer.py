from sklearn.ensemble import IsolationForest
import joblib
import pickle
import numpy as np

pick_in = open('CTGAFmaleupdatedFeatures.pickle', 'rb')
# pick_in = open('finalfemalefeaturefile.pickle', 'rb')
# pick_in = open('FemaleFeatures[Balanced].pickle', 'rb')

data = pickle.load(pick_in)
pick_in.close()

print("Number of samples:", len(data))
X = []

for feature, label, filenames in data:
    feature = feature.flatten()
    X.append(feature)

X_train = np.array(X)
bouncer = IsolationForest(contamination=0.01, random_state=42)
bouncer.fit(X_train)

joblib.dump(bouncer, "histology_bouncer_male.joblib")
print("Bouncer model trained and saved Successfully.")