import cv2
import numpy as np


def resize_images(normal_image, mask_image, target_size):
    resized_normal = cv2.resize(normal_image, target_size)
    resized_mask = cv2.resize(mask_image, target_size)
    return resized_normal, resized_mask

def apply_acoustic_effects(normal_image, mask_image, margin=5):
    mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    _, tumor_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        tumor_contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(tumor_contour)
        roi_under_tumor = normal_image[y + height // 2:, x:x + width]
        mean_tumor_brightness = np.mean(normal_image[tumor_mask == 255])
        mean_under_tumor_brightness = np.mean(roi_under_tumor)
        brightness_difference = abs(mean_tumor_brightness - mean_under_tumor_brightness)
        if brightness_difference < 16.5:
            effect_label = "Acoustic Shadowing"
        elif mean_under_tumor_brightness > mean_tumor_brightness:
            effect_label = "Acoustic Enhancement"
        else:
            effect_label = "Acoustic Shadowing"
        return effect_label, brightness_difference
    else:
        print("No tumor detected in the mask image.")
        return None, None

normal_image = cv2.imread(r"C:\Users\asyao\PycharmProjects\USCAI\_datasets\MASKED DATASETS\Dataset_BUSI_with_GT\benign\benign (3).png")
mask_image = cv2.imread(r"C:\Users\asyao\PycharmProjects\USCAI\_datasets\MASKED DATASETS\Dataset_BUSI_with_GT\benign\benign (3)_mask.png")

target_size = (500, 500)
resized_normal, resized_mask = resize_images(normal_image, mask_image, target_size)
effect_label, brightness_difference = apply_acoustic_effects(resized_normal, resized_mask)
if effect_label is not None:
    print("Acoustic Effect:", effect_label)
    print("Brightness Difference:", brightness_difference)
