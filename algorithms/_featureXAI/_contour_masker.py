import numpy as np
import cv2

def extract_tumor_border(mask_image):
    grayscale = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5), np.uint8)
    gradient = cv2.morphologyEx(grayscale, cv2.MORPH_GRADIENT, kernel)
    _, border = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return border

mask_image = cv2.imread(r"C:\Users\asyao\PycharmProjects\USCAI\_datasets\MASKED DATASETS\Dataset_BUSI_with_GT\benign\benign (1)_mask.png")
tumor_border = extract_tumor_border(mask_image)
cv2.imshow("Tumor Border", tumor_border)
cv2.waitKey(0)
cv2.destroyAllWindows()
