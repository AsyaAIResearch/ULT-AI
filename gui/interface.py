# import libraries
import os
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk
from PIL import ImageFilter
from keras.models import load_model

from tools import easy_qt

image_path = None
original_image_label = None
cropped_image_label = None
cropped_image_path = None

# paths & load models
feature_model_paths = {
    "f1": r"models/XAImodels/vertically.h5",
    "f2": r"models/XAImodels/shodowing.h5",
    "f3": r"models/XAImodels/enhancement.h5",
    "f4": r"models/XAImodels/echogenity.h5",
    "f5": r"models/XAImodels/contour.h5",
}
tumor_type_model_path = r"models\CNN_3_bm.h5"
segment_model_path = r"models\segmentation.h5"
tumor_type_model = load_model(tumor_type_model_path)
segment_model = load_model(segment_model_path)
feature_models = {name: load_model(path) for name, path in feature_model_paths.items()}

# root
root = tk.Tk()
root.title("ULT - AI: Cancer Diagnostic System - ULT-AI")
width, height = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry(f"{width}x{height}")


tumor_region_label = tk.Label(root, text="", font=("Courier", 20))
tumor_region_label.place(x=661, y=794)


def apply_smooth_blur(mask_pred):
    smoothed_mask = (
        cv2.GaussianBlur(mask_pred.astype(np.uint8) * 255, (15, 15), 0) / 255.0
    )
    return smoothed_mask


def apply_laplacian_sharpen(image):
    image_8bit = (image * 255).astype(np.uint8)
    laplacian_image = cv2.Laplacian(image_8bit, cv2.CV_64F)
    sharpened_image = image_8bit - 45.0 * laplacian_image
    return sharpened_image / 255.0


def blend_mask_with_image(original_image, mask):
    original_image_rgba = original_image.convert("RGBA")
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel=np.ones((15, 15), np.uint8))
    mask_image = Image.new("RGBA", original_image_rgba.size)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if dilated_mask[i, j] == 255:
                mask_image.putpixel((j, i), (0, 0, 0, 192))
            else:
                mask_image.putpixel((j, i), (255, 255, 255, 100))
    mask_image_blurred = mask_image.filter(ImageFilter.GaussianBlur(radius=5))
    blended_image = Image.alpha_composite(original_image_rgba, mask_image_blurred)
    blended_image_rgb = blended_image.convert("RGB")

    return blended_image_rgb


def display_tumor_analysis_results(
    width, height, orientation, effect_label, brightness_difference
):
    x_coordinate = 550
    y_coordinate = 785

    width_label = tk.Label(root, text=f"Width: {width} px", font=("Courier", 18))
    width_label.place(x=x_coordinate, y=y_coordinate)

    y_coordinate += 40
    height_label = tk.Label(root, text=f"Height: {height} px", font=("Courier", 18))
    height_label.place(x=x_coordinate, y=y_coordinate)

    y_coordinate += 40
    orientation_label = tk.Label(
        root, text=f"Orientation: {orientation}", font=("Courier", 18)
    )
    orientation_label.place(x=x_coordinate, y=y_coordinate)

    y_coordinate += 40
    effect_label = tk.Label(
        root, text=f"Acoustic Effect: {effect_label}", font=("Courier", 18)
    )
    effect_label.place(x=x_coordinate, y=y_coordinate)

    y_coordinate += 40
    brightness_difference_label = tk.Label(
        root,
        text=f"Brightness Difference: {brightness_difference:.2f}",
        font=("Courier", 18),
    )
    brightness_difference_label.place(x=x_coordinate, y=y_coordinate)


def segmentation(data):
    masks_pred, _ = segment_model.predict(data)
    masks_pred = np.array(masks_pred)
    masks_pred = (masks_pred >= 0.5).astype("int32")
    processed_mask = apply_smooth_blur(masks_pred[0])
    sharpened_mask = apply_laplacian_sharpen(processed_mask)
    return np.expand_dims(sharpened_mask, axis=0)


def tumor_analysis(mask):
    if len(mask.shape) == 3:
        grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    elif len(mask.shape) == 2:
        grayscale = mask
    else:
        raise ValueError("either 3-channel (BGR) or 1-channel (grayscale) ")
    grayscale = np.uint8(grayscale)
    contours, _ = cv2.findContours(
        grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(largest_contour)
        if height > width:
            orientation = "Horizontal"
        else:
            orientation = "Vertical"
        return width, height, orientation
    else:
        return None, None, None


def apply_acoustic_effects(normal_image, mask_image, margin=5):
    normal_image_array = np.array(normal_image)
    mask_image_array = np.array(mask_image)
    tumor_pixels = normal_image_array[mask_image_array == 255]
    mean_under_tumor_brightness = np.mean(tumor_pixels)
    mean_tumor_brightness = np.mean(normal_image_array[mask_image_array == 0])
    brightness_difference = abs(mean_under_tumor_brightness - mean_tumor_brightness)
    if (
        mean_under_tumor_brightness > mean_tumor_brightness
        or brightness_difference > 80
    ):
        effect_label = "Enhancement"
    else:
        effect_label = "Shadowing"
    return effect_label, brightness_difference


def detect_and_crop_image():
    print("None")


def select_image():
    file_path = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Resim Seç",
        filetypes=[("Resim dosyaları", "*.jpg;*.jpeg;*.png")],
    )
    if file_path:
        classify_image(file_path)


def classify_image(file_path):
    global img_label, result_label, probability_label, tumor_region_label

    if file_path:
        main()
        img = Image.open(file_path)
        img_resized_128 = img.resize((128, 128))
        img_resized_300 = img.resize((300, 300))
        img_300_photo = ImageTk.PhotoImage(img_resized_300)
        img_label = tk.Label(root, image=img_300_photo)
        img_label.image = img_300_photo
        img_label.place(x=68, y=181)

        image_normalized = np.array(img_resized_128) / 255.0
        image_input = np.expand_dims(image_normalized, axis=0)
        masks_pred = segmentation(image_input)

        mask_resized_np1 = masks_pred[0].reshape((128, 128)) * 255
        mask_resized_np = cv2.resize(
            mask_resized_np1, (300, 300), interpolation=cv2.INTER_NEAREST
        )
        blended_image = blend_mask_with_image(img_resized_300, mask_resized_np)

        blended_image_tk = ImageTk.PhotoImage(blended_image)
        mask_label = tk.Label(root, image=blended_image_tk)
        mask_label.image = blended_image_tk
        mask_label.place(x=584, y=205)

        classification = masks_pred[0].mean()
        result_label = tk.Label(root, text="Result:", font=("Courier", 20))
        result_label.place(x=762, y=574)

        abnormality_percentage_label = tk.Label(root, text=" ", font=("Courier", 18))
        abnormality_percentage_label.place(x=954, y=574)

        tumor_area_label = tk.Label(root, text=" ", font=("Courier", 20))
        tumor_area_label.place(x=763, y=652)

        tumor_width, tumor_height, tumor_orientation = tumor_analysis(mask_resized_np1)
        effect_label, brightness_difference = apply_acoustic_effects(
            img_resized_128, mask_resized_np1
        )
        display_tumor_analysis_results(
            tumor_width,
            tumor_height,
            tumor_orientation,
            effect_label,
            brightness_difference,
        )

        prediction_label = tk.Label(root, font=("Courier", 25))
        prediction_label.place(x=1481, y=221)

        threshold_classification = 0.6
        abnormality = classification > threshold_classification

        if masks_pred.any() and classification > 0:
            result_label.config(text=f"{'Abnormal'}")
            pred_class = 1 if abnormality else 0
            prediction_label.config(text=f"{'Malignant' if pred_class else 'Benign'}")

            abnormality_percentage = (1 - classification) * 100
            abnormality_percentage_label.config(text=f"{abnormality_percentage:.2f}")

            tumor_area = np.count_nonzero(mask_resized_np) / (300 * 300) * 100
            tumor_area_label.config(text=f"{tumor_area:.2f} px²")

        else:
            result_label.config(text=f"{'Normal'}")
            prediction_label.config(text="Normal")
            abnormality_percentage_label.config(text="0.00%")
            tumor_area_label.config(text="0.00%")


def exit_application():
    root.destroy()


def entry():
    easy_qt.clear_screen(root)
    easy_qt.create_image(
        root,
        r"C:\Users\asyao\PycharmProjects\USCAI\Versions\version - may\gui_assets\Enterance.png",
        0,
        0,
    )

    def open_contact_us():
        pass

    def open_language_options():
        pass

    def select_image():
        file_path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")],
        )
        if file_path:
            classify_image(file_path)

    def exit_application():
        root.destroy()

    long_load_btn = easy_qt.create_button_with_image(
        root, "gui_assets/long_load.png", 632, 675, select_image
    )
    exit_btn = easy_qt.create_button_with_image(
        root, "gui_assets/exit.png", 110, 914, exit_application
    )
    contact_btn = easy_qt.create_button_with_image(
        root, "gui_assets/contact.png", 1633, 944, open_contact_us
    )
    language_btn = easy_qt.create_button_with_image(
        root, "gui_assets/language_option.png", 1763, 114, open_language_options
    )

    terms_image_path = "C:/Users/asyao/PycharmProjects/USCAI/Versions/version - may/gui_assets/terms_of service.png"
    terms_label = easy_qt.create_image(root, terms_image_path, 662, 593)

    ticked_image_path = r"C:\Users\asyao\PycharmProjects\USCAI\Versions\version - may\gui_assets\ticked.png"
    unticked_image_path = r"C:\Users\asyao\PycharmProjects\USCAI\Versions\version - may\gui_assets\unticked.png"

    terms_var = tk.BooleanVar(root)
    terms_var.set(False)

    def toggle_terms():
        if terms_var.get():
            terms_checkbox.config(image=ticked_image)
        else:
            terms_checkbox.config(image=unticked_image)

    ticked_image = tk.PhotoImage(file=ticked_image_path).subsample(5)
    unticked_image = tk.PhotoImage(file=unticked_image_path).subsample(5)

    terms_checkbox = tk.Checkbutton(
        root,
        variable=terms_var,
        onvalue=True,
        offvalue=False,
        image=unticked_image,
        command=toggle_terms,
        bd=0,
        highlightthickness=0,
        selectcolor="#FFFFFF",
    )
    terms_checkbox.place(x=600, y=600)
    root.attributes("-fullscreen", True)


def main():
    easy_qt.clear_screen(root)
    easy_qt.create_image(root, "gui_assets/Main.png", 0, 0)

    original_image_label = tk.Label(root)
    cropped_image_label = tk.Label(root)

    def load_image():
        print("Downloads the image / this feature is in the application")

    def download_results():
        print("Downloads the results / this feature is in the application")

    def refresh_interface():
        print("Restarts / this feature is in the application")

    def exit_application():
        root.destroy()

    upload_button = easy_qt.create_button_with_image(
        root, "gui_assets/download.png", 201, 778, load_image
    )
    load_image_button = easy_qt.create_button_with_image(
        root, "gui_assets/load.png", 37, 677, select_image
    )
    refresh_button = easy_qt.create_button_with_image(
        root, "gui_assets/refreash.png", 366, 775, refresh_interface
    )
    exit_button = easy_qt.create_button_with_image(
        root, "gui_assets/exit.png", 51, 941, exit_application
    )

    switch_on_image = Image.open("gui_assets/swich_on.png").resize((100, 50))
    switch_off_image = Image.open("gui_assets/swich_off.png").resize((100, 50))

    switch_on_photo = ImageTk.PhotoImage(switch_on_image)
    switch_off_photo = ImageTk.PhotoImage(switch_off_image)

    def switch_interface():
        global image_path

        if switch_button.cget("text") == "Crop Image":
            detect_and_crop_image()
            switch_button.config(text="Show Original", image=switch_off_photo)
        else:
            img = Image.open(image_path)
            img_resized_300 = img.resize((300, 300))
            img_300_photo = ImageTk.PhotoImage(img_resized_300)
            original_image_label.config(image=img_300_photo)
            original_image_label.image = img_300_photo
            cropped_image_label.config(text="")
            switch_button.config(text="Crop Image", image=switch_on_photo)

    switch_button = tk.Button(
        root,
        text="Crop Image",
        image=switch_off_photo,
        command=switch_interface,
        borderwidth=0,
        highlightthickness=0,
    )
    switch_button.place(x=51, y=601)
    root.attributes("-fullscreen", True)


entry()
root.mainloop()
