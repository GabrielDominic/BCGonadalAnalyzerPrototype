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
    
        
def analyze_GLCM_mean(pickle_file):
    # glcm_names = ['contrast_mean', 'contrast_std', 'homogeneity_mean', 'homogeneity_std', 
    #           'energy_mean', 'energy_std', 'correlation_mean', 'correlation_std']
    
    developing_glcm = []
    maturing_glcm = []
    spawning_glcm = []
    spent_glcm = []

    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        #Analyze GLCM
    print("\nAnalyzing GLCM features:")
    for feature, label, filenames in data:
        glcm_vals = feature[0:8]  # Assuming GLCM features are the first 8 values
        if label == 0:
            developing_glcm.append(glcm_vals)
        elif label == 1:
            maturing_glcm.append(glcm_vals)
        elif label == 2:
            spawning_glcm.append(glcm_vals)
        elif label == 3:
            spent_glcm.append(glcm_vals)

    print(f'\nDeveloping \n\tContrast mean GLCM: {np.mean(developing_glcm[0])} \n\tcontrast std GLCM: {np.mean(developing_glcm[1])} \n\thomogeneity mean GLCM: {np.mean(developing_glcm[2])} \n\thomogeneity std GLCM: {np.mean(developing_glcm[3])} \n\tenergy mean GLCM: {np.mean(developing_glcm[4])} \n\tenergy std GLCM: {np.mean(developing_glcm[5])} \n\tcorrelation mean GLCM: {np.mean(developing_glcm[6])} \n\tcorrelation std GLCM: {np.mean(developing_glcm[7])}')
    print(f'\nMaturing \n\tContrast mean GLCM: {np.mean(maturing_glcm[0])} \n\tcontrast std GLCM: {np.mean(maturing_glcm[1])} \n\thomogeneity mean GLCM: {np.mean(maturing_glcm[2])} \n\thomogeneity std GLCM: {np.mean(maturing_glcm[3])} \n\tenergy mean GLCM: {np.mean(maturing_glcm[4])} \n\tenergy std GLCM: {np.mean(maturing_glcm[5])} \n\tcorrelation mean GLCM: {np.mean(maturing_glcm[6])} \n\tcorrelation std GLCM: {np.mean(maturing_glcm[7])}')
    print(f'\nSpawning \n\tContrast mean GLCM: {np.mean(spawning_glcm[0])} \n\tcontrast std GLCM: {np.mean(spawning_glcm[1])} \n\thomogeneity mean GLCM: {np.mean(spawning_glcm[2])} \n\thomogeneity std GLCM: {np.mean(spawning_glcm[3])} \n\tenergy mean GLCM: {np.mean(spawning_glcm[4])} \n\tenergy std GLCM: {np.mean(spawning_glcm[5])} \n\tcorrelation mean GLCM: {np.mean(spawning_glcm[6])} \n\tcorrelation std GLCM: {np.mean(spawning_glcm[7])}')
    print(f'\nSpent \n\tContrast mean GLCM: {np.mean(spent_glcm[0])} \n\tcontrast std GLCM: {np.mean(spent_glcm[1])} \n\thomogeneity mean GLCM: {np.mean(spent_glcm[2])} \n\thomogeneity std GLCM: {np.mean(spent_glcm[3])} \n\tenergy mean GLCM: {np.mean(spent_glcm[4])} \n\tenergy std GLCM: {np.mean(spent_glcm[5])} \n\tcorrelation mean GLCM: {np.mean(spent_glcm[6])} \n\tcorrelation std GLCM: {np.mean(spent_glcm[7])}')   


def analyze_LBP_mean(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        #Analyze LBP
    
    print("\nAnalyzing LBP features:")
    P = 24
    # lbp_names = [f'lbp_bin_{i}' for i in range(P + 2)]  # 26 bins
    
    developing_mean_lbp = []
    maturing_mean_lbp = []
    spawning_mean_lbp = []
    spent_mean_lbp = []

    developing_std_lbp = []
    maturing_std_lbp = []
    spawning_std_lbp = []
    spent_std_lbp = []

    for feature, label, filenames in data:
        lbp_vals = feature[9:9+P+2]  # Assuming LBP features start at index 9
        if label == 0:
            developing_mean_lbp.append(np.mean(lbp_vals))
            developing_std_lbp.append(np.std(lbp_vals))
            
        elif label == 1:
            maturing_mean_lbp.append(np.mean(lbp_vals))
            maturing_std_lbp.append(np.std(lbp_vals))
            
        elif label == 2:
            spawning_mean_lbp.append(np.mean(lbp_vals))
            spawning_std_lbp.append(np.std(lbp_vals))
            
        elif label == 3:
            spent_mean_lbp.append(np.mean(lbp_vals))
            spent_std_lbp.append(np.std(lbp_vals))
        
    print(f'Developing mean LBP: {np.mean(developing_mean_lbp):.4f}, std LBP: {np.mean(developing_std_lbp):.4f}')
    print(f'Mature mean LBP: {np.mean(maturing_mean_lbp):.4f}, std LBP: {np.mean(maturing_std_lbp):.4f}')
    print(f'Spawning mean LBP: {np.mean(spawning_mean_lbp):.4f}, std LBP: {np.mean(spawning_std_lbp):.4f}')
    print(f'Spent mean LBP: {np.mean(spent_mean_lbp):.4f}, std LBP: {np.mean(spent_std_lbp):.4f}')

def analyze_Color_Moments(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        #Analyze Color Moments

    # cm_names = ['R_mean', 'R_std', 'R_skew', 'G_mean', 'G_std', 'G_skew', 'B_mean', 'B_std', 'B_skew']
    developing_cm = []
    maturing_cm = []
    spawning_cm = []
    spent_cm = []
    print("\nAnalyzing Color Moment features:")
    for feature, label, filenames in data:
        cm_feat = feature[34:34+9]  # Assuming color moment features start at index 34
        if label == 0:
            developing_cm.append(cm_feat)
        elif label == 1:
            maturing_cm.append(cm_feat)
        elif label == 2:
            spawning_cm.append(cm_feat)
        elif label == 3:
            spent_cm.append(cm_feat)
    
    print(f'Developing\n\tR_mean: {np.mean(developing_cm[0])} \n\tR_std: {np.mean(developing_cm[1])} \n\tR_skew: {np.mean(developing_cm[2])} \n\tG_mean: {np.mean(developing_cm[3])} \n\tG_std: {np.mean(developing_cm[4])} \n\tG_skew: {np.mean(developing_cm[5])} \n\tB_mean: {np.mean(developing_cm[6])} \n\tB_std: {np.mean(developing_cm[7])} \n\tB_skew: {np.mean(developing_cm[8])}')
    print(f'Maturing\n\tR_mean: {np.mean(maturing_cm[0])} \n\tR_std: {np.mean(maturing_cm[1])} \n\tR_skew: {np.mean(maturing_cm[2])} \n\tG_mean: {np.mean(maturing_cm[3])} \n\tG_std: {np.mean(maturing_cm[4])} \n\tG_skew: {np.mean(maturing_cm[5])} \n\tB_mean: {np.mean(maturing_cm[6])} \n\tB_std: {np.mean(maturing_cm[7])} \n\tB_skew: {np.mean(maturing_cm[8])}')
    print(f'Spawning\n\tR_mean: {np.mean(spawning_cm[0])} \n\tR_std: {np.mean(spawning_cm[1])} \n\tR_skew: {np.mean(spawning_cm[2])} \n\tG_mean: {np.mean(spawning_cm[3])} \n\tG_std: {np.mean(spawning_cm[4])} \n\tG_skew: {np.mean(spawning_cm[5])} \n\tB_mean: {np.mean(spawning_cm[6])} \n\tB_std: {np.mean(spawning_cm[7])} \n\tB_skew: {np.mean(spawning_cm[8])}')
    print(f'Spent\n\tR_mean: {np.mean(spent_cm[0])} \n\tR_std: {np.mean(spent_cm[1])} \n\tR_skew: {np.mean(spent_cm[2])} \n\tG_mean: {np.mean(spent_cm[3])} \n\tG_std: {np.mean(spent_cm[4])} \n\tG_skew: {np.mean(spent_cm[5])} \n\tB_mean: {np.mean(spent_cm[6])} \n\tB_std: {np.mean(spent_cm[7])} \n\tB_skew: {np.mean(spent_cm[8])}')

def analyze_morphological_features(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        #Analyze Morphological features

    # morph_names = ['area_foreground', 'area_contour', 'circularity']
    developing_morph = []
    maturing_morph = []
    spawning_morph = []
    spent_morph = []
    print("\nAnalyzing Morphological features:")
    for feature, label, filenames in data:
        morph_feat = feature[43:43+3]  # Assuming morphological features start at index 43
        if label == 0:
            developing_morph.append(morph_feat)
        elif label == 1:
            maturing_morph.append(morph_feat)
        elif label == 2:
            spawning_morph.append(morph_feat)
        elif label == 3:
            spent_morph.append(morph_feat)
    
    print(f'Developing\n\tarea_foreground: {np.mean(developing_morph[0])} \n\tarea_contour: {np.mean(developing_morph[1])} \n\tcircularity: {np.mean(developing_morph[2])}')
    print(f'Maturing\n\tarea_foreground: {np.mean(maturing_morph[0])} \n\tarea_contour: {np.mean(maturing_morph[1])} \n\tcircularity: {np.mean(maturing_morph[2])}')
    print(f'Spawning\n\tarea_foreground: {np.mean(spawning_morph[0])} \n\tarea_contour: {np.mean(spawning_morph[1])} \n\tcircularity: {np.mean(spawning_morph[2])}')
    print(f'Spent\n\tarea_foreground: {np.mean(spent_morph[0])} \n\tarea_contour: {np.mean(spent_morph[1])} \n\tcircularity: {np.mean(spent_morph[2])}')    

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
        sobel_vals = feature[46:49]  # Assuming Sobel features are at indices 46-48
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

        print(f'\n{name.capitalize()} Sobel means:')
        print(f'  mean_sobel_value: {sobel_mean:.4f}')
        print(f'  std_sobel_value: {sobel_std:.4f}')
        print(f'  mean_edge_density: {mean_edge_density:.4f}')
        print(f'  std_edge_density: {std_edge_density:.4f}\n')

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
        # std_total_tissue = np.std(values[:, 0])
        # std_gamete_pixels = np.std(values[:, 1])
        # std_area_fraction = np.std(values[:, 2])

        print(f'{name.capitalize()} means:')
        print(f'  total_tissue_pixels: {mean_total_tissue:.4f}')
        print(f'  gamete_pixels: {mean_gamete_pixels:.4f}')
        print(f'  gamete_area_fraction: {mean_area_fraction:.4f}')
        # print(f'{name.capitalize()} std devs:')
        # print(f'  total_tissue_pixels: {std_total_tissue:.4f}')
        # print(f'  gamete_pixels: {std_gamete_pixels:.4f}')
        # print(f'  gamete_area_fraction: {std_area_fraction:.4f}\n')

#Change Name to corresponding extracted features file
# pickle_file = 'CTGAFmaleupdatedFeatures.pickle' #Male Samples
pickle_file = 'FemaleFeatures[Balanced].pickle' #Female Samples

analyze_features_stat(pickle_file)
analyze_GLCM_mean(pickle_file)
analyze_LBP_mean(pickle_file)
analyze_Color_Moments(pickle_file)
analyze_morphological_features(pickle_file)
analyze_Sobel_features(pickle_file)
analyze_GAF_mean(pickle_file)
