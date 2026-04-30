import numpy as np
import cv2

def get_mean_and_std(image):
    x_mean, x_std = cv2.meanStdDev(image)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std = np.hstack(np.around(x_std, 2))
    return x_mean, x_std


def reinhard_normalization(input_bgr, reference_path):
    # Convert input image to LAB
    input_lab = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Load reference image
    ref_img = cv2.imread(reference_path)
    ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Compute statistics
    ref_mean, ref_std = get_mean_and_std(ref_lab)
    ref_mean = ref_mean.reshape(1, 1, 3)
    ref_std = ref_std.reshape(1, 1, 3)

    input_mean, input_std = get_mean_and_std(input_lab)
    input_std = np.where(input_std == 0, 1, input_std)
    # Apply Reinhard normalization
    normalized = (input_lab - input_mean) * (ref_std / input_std) + ref_mean
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # Convert back to BGR
    output_bgr = cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)

    return output_bgr