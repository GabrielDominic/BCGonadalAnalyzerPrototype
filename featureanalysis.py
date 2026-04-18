import numpy as np
import pickle

def analyze_features_stat(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    print("Total samples:", len(data))
    if len(data) == 0:
        print("No data found in the pickle file.")
        return

    feature_shapes = {}
    label_counts = {}

    for feature, label, filenames in data:
        shape = feature.shape
        feature_shapes[shape] = feature_shapes.get(shape, 0) + 1
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\nFeature shape distribution:")
    for shape, count in feature_shapes.items():
        print(f"Shape: {shape}, Count: {count}")
    
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"Label: {label}, Count: {count}")
    
def analyze_GAF_mean(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    if len(data) == 0:
        print("No data found in the pickle file.")
        return

    developing_mean_values = []
    mature_mean_values = []
    spawning_mean_values = []
    spent_mean_values = []
    gamete_features = []

    for feature, label, filenames in data:
        gamete_vals = feature[-3:]
        gamete_features.append(gamete_vals)
        if label == 0:  # developing
            developing_mean_values.append(feature[-1:].mean())
        elif label == 1:  # maturing
            mature_mean_values.append(feature[-1:].mean())
        elif label == 2:  # spawning
            spawning_mean_values.append(feature[-1:].mean())
        elif label == 3:  # spent
            spent_mean_values.append(feature[-1:].mean())
        
    gamete_features = np.array(gamete_features)
    print(f'Gamete features shape: {gamete_features.shape}')
    print(f'Developing mean of last feature: {np.mean(developing_mean_values):.4f}')
    print(f'Maturing mean of last feature: {np.mean(mature_mean_values):.4f}')
    print(f'Spawning mean of last feature: {np.mean(spawning_mean_values):.4f}')    
    print(f'Spent mean of last feature: {np.mean(spent_mean_values):.4f}')


def analyze_LBP_mean(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        #Analyze LBP


pickle_file = 'CTGAFmaleupdatedFeatures.pickle'
analyze_features_stat(pickle_file)
analyze_GAF_mean(pickle_file)
analyze_LBP_mean(pickle_file)