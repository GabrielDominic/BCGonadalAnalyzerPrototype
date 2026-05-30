import numpy as np
import cv2 
import os 

# input_dir = 'imagedataset/M/spent' Change to the directory of the images you want to normalize
input_dir = 'D:\\SP\\BCDataset15-4-2026\\M\\mature'
input_image_list = os.listdir(input_dir)
output_dir = 'normalized_updated_dataset/M/maturing'

def get_mean_and_std(image):
    x_mean, x_std = cv2.meanStdDev(image)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std = np.hstack(np.around(x_std, 2))
    return x_mean, x_std

reference_img = cv2.imread('reference_images/3 M-20x_Developing.jpg')
reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2LAB).astype(np.float32)

ref_mean, ref_std = get_mean_and_std(reference_img)
ref_mean = ref_mean.reshape(1,1,3)
ref_std = ref_std.reshape(1,1,3)

for img in (input_image_list):
    if img.endswith(('.jpg', '.jpeg', '.png')):
        print(f"Processing {img}...")
        input_img = cv2.imread(os.path.join(input_dir, img))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        input_mean, input_std = get_mean_and_std(input_img)

        # Subtract the mean from the input image
        normalized_img = (input_img - input_mean) * (ref_std / input_std) + ref_mean
        normalized_img = np.clip(normalized_img, 0, 255).astype(np.uint8)

        output_img = cv2.cvtColor(normalized_img, cv2.COLOR_LAB2BGR)

        cv2.imwrite(os.path.join(output_dir, img), output_img)
    # print(f"Processing {img}...")
    # input_img = cv2.imread(os.path.join(input_dir, img))
    # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
    
    # input_mean, input_std = get_mean_and_std(input_img)

    # # Subtract the mean from the input image
    # normalized_img = (input_img - input_mean) * (ref_std / input_std) + ref_mean
    # normalized_img = np.clip(normalized_img, 0, 255).astype(np.uint8)

    # output_img = cv2.cvtColor(normalized_img, cv2.COLOR_LAB2BGR)

    # cv2.imwrite(os.path.join(output_dir, img), output_img)