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

    label_names = ['developing', 'maturing', 'spawning', 'spent']
    label_gaf = {name: [] for name in label_names}
    gamete_features = []

    for feature, label, filenames in data:
        gamete_vals = feature[-3:]
        gamete_features.append(gamete_vals)

        if 0 <= label < len(label_names):
            label_gaf[label_names[label]].append(gamete_vals)
        else:
            print(f'Warning: unknown label {label} for file {filenames}')

    gamete_features = np.array(gamete_features)
    print(f'\nGamete features shape: {gamete_features.shape}')

    for name in label_names:
        values = np.array(label_gaf[name], dtype=np.float32)
        if values.size == 0:
            print(f'{name.capitalize()} has no GAF samples.')
            continue

        mean_total_tissue = np.mean(values[:, 0])
        mean_gamete_pixels = np.mean(values[:, 1])
        mean_area_fraction = np.mean(values[:, 2])
        std_total_tissue = np.std(values[:, 0])
        std_gamete_pixels = np.std(values[:, 1])
        std_area_fraction = np.std(values[:, 2])

        print(f'{name.capitalize()} means:')
        print(f'  total_tissue_pixels: {mean_total_tissue:.4f}')
        print(f'  gamete_pixels: {mean_gamete_pixels:.4f}')
        print(f'  gamete_area_fraction: {mean_area_fraction:.4f}')
        print(f'{name.capitalize()} std devs:')
        print(f'  total_tissue_pixels: {std_total_tissue:.4f}')
        print(f'  gamete_pixels: {std_gamete_pixels:.4f}')
        print(f'  gamete_area_fraction: {std_area_fraction:.4f}\n')

def analyze_LBP_mean(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        #Analyze LBP

def analyze_Sobel_features(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    if len(data) == 0:
        print("No data found in the pickle file.")
        return
    
    label_names = ['developing', 'maturing', 'spawning', 'spent']
    label_sobel = {name: [] for name in label_names}
    sobel_features = []

    for feature, label, filenames in data:
        sobel_vals = feature[-3:]
        sobel_features.append(sobel_vals)

        if 0 <= label < len(label_names):
            label_sobel[label_names[label]].append(sobel_vals)
        else:
            print(f'Warning: unknown label {label} for file {filenames}')

    for name in label_names:
        values = np.array(label_sobel[name], dtype=np.float32)
        if values.size == 0:
            print(f'{name.capitalize()} has no Sobel samples.')
            continue

        sobel_mean = np.mean(values[:, 0])
        sobel_std = np.std(values[:, 1])
        mean_edge_density = np.mean(values[:, 2])
        std_edge_density = np.std(values[:, 2])

        print(f'{name.capitalize()} Sobel means:')
        print(f'  mean_sobel_value: {sobel_mean:.4f}')
        print(f'  std_sobel_value: {sobel_std:.4f}')
        print(f'  mean_edge_density: {mean_edge_density:.4f}')
        print(f'  std_edge_density: {std_edge_density:.4f}\n')


pickle_file = 'CTGAFmaleupdatedFeatures.pickle'
analyze_features_stat(pickle_file)
analyze_GAF_mean(pickle_file)
analyze_LBP_mean(pickle_file)
analyze_Sobel_features(pickle_file)