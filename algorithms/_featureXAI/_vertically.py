import cv2

def calculate_tumor_dimensions(mask_image):
    grayscale = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(largest_contour)
        if width > height:
            orientation = "Horizontal"
        else:
            orientation = "Vertical"
        return width, height, orientation
    else:
        return None, None, None


mask_image_path = r"C:\Users\asyao\PycharmProjects\USCAI\_datasets\MASKED DATASETS\Dataset_BUSI_with_GT\benign\benign (1)_mask.png"
mask_image = cv2.imread(mask_image_path)
width, height, orientation = calculate_tumor_dimensions(mask_image)

if width is not None and height is not None and orientation is not None:
    print("Tumor Dimensions:")
    print("Width:", width)
    print("Height:", height)
    print("Orientation:", orientation)
else:
    print("No tumor detected in the mask image.")
