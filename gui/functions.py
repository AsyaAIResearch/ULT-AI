import json
import sys

import cv2
import numpy as np
from PIL import Image, ImageFilter
from keras.models import load_model

MODEL_PATHS = {
    "segment": r"C:\Users\asyao\PycharmProjects\USCAI\models\segmentation.h5",
    "tumor_type": r"C:\Users\asyao\PycharmProjects\USCAI\models\CNN_3_bm.h5",
    "features": {
        "f1": r"C:\Users\asyao\PycharmProjects\USCAI\XAImodels\vertically.h5",
        "f2": r"C:\Users\asyao\PycharmProjects\USCAI\XAImodels\shodowing.h5",
        "f3": r"C:\Users\asyao\PycharmProjects\USCAI\XAImodels\enhancement.h5",
        "f4": r"C:\Users\asyao\PycharmProjects\USCAI\XAImodels\echogenity.h5",
        "f5": r"C:\Users\asyao\PycharmProjects\USCAI\XAImodels\contour.h5",
    },
}


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


def load_models():
    return {
        "segment": load_model(MODEL_PATHS["segment"]),
        "tumor_type": load_model(MODEL_PATHS["tumor_type"]),
        "features": {k: load_model(v) for k, v in MODEL_PATHS["features"].items()},
    }


def process_image(image_path, models):
    results = {"tumor_detected": False}
    orig_img = Image.open(image_path).convert("RGB")
    original_size = orig_img.size
    seg_img = orig_img.resize((128, 128))
    seg_array = np.expand_dims(np.array(seg_img) / 255.0, axis=0)
    mask_pred = models["segment"].predict(seg_array, verbose=0)[0]
    mask = (mask_pred > 0.5).astype(np.uint8).squeeze()
    if np.sum(mask) == 0:
        return results

    results["tumor_detected"] = True
    results["visualization"] = create_visualization(orig_img, mask, original_size)
    results.update(analyze_tumor_metrics(mask))
    tumor_type_img = Image.open(image_path).convert("RGB").resize((224, 224))
    tumor_array = np.expand_dims(np.array(tumor_type_img) / 255.0, axis=0)
    results.update(classify_tumor_type(tumor_array, models))
    results.update(analyze_radiomic_features(orig_img, mask, models))

    return results


def create_visualization(original_img, mask, original_size):
    resized_mask = cv2.resize(
        mask.astype(np.uint8),
        (original_size[0], original_size[1]),
        interpolation=cv2.INTER_NEAREST,
    )
    mask_img = Image.fromarray((resized_mask * 200).astype(np.uint8))
    overlay_color = Image.new("RGBA", original_size, (0, 0, 255, 100))
    blended = Image.composite(overlay_color, original_img.convert("RGBA"), mask_img)

    output_path = "result_visualization.png"
    blended.save(output_path)
    return output_path


def analyze_tumor_metrics(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return {}

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    return {
        "width": w,
        "height": h,
        "orientation": "Vertical" if h > w else "Horizontal",
        "area": float(cv2.contourArea(largest)),
        "aspect_ratio": h / w if w != 0 else 0,
    }


def classify_tumor_type(tumor_array, models):
    pred = models["tumor_type"].predict(tumor_array, verbose=0)[0][0]
    return {
        "tumor_type": "Malignant" if pred > 0.5 else "Benign",
        "malignant_prob": float(pred),
        "benign_prob": float(1 - pred),
    }


def analyze_radiomic_features(img, mask, models):
    img_128 = img.resize((128, 128))
    img_array = np.expand_dims(np.array(img_128) / 255.0, axis=0)
    mask_3ch = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)

    feature_inputs = {
        "f1": np.expand_dims(mask_3ch, axis=0),
        "f2": img_array,
        "f3": img_array,
        "f4": img_array,
        "f5": np.expand_dims(mask_3ch, axis=0),
    }

    features = {}
    for feat, model in models["features"].items():
        input_data = feature_inputs[feat]
        pred = model.predict(input_data, verbose=0)[0][0]
        features[feat] = sigmoid(pred)

    features.update(
        {
            "vertical_orientation": features["f1"] > 0.5,
            "acoustic_effect": (
                "Shadowing" if features["f2"] > features["f3"] else "Enhancement"
            ),
            "echo_pattern": classify_echogenity(features["f4"]),
            "margin_type": classify_contour(features["f5"]),
        }
    )

    return features


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def classify_echogenity(value):
    if value < 0.3:
        return "Anechoic"
    if value < 0.6:
        return "Hypoechoic"
    if value < 0.8:
        return "Isoechoic"
    return "Hyperechoic"


def classify_contour(value):
    if value < 0.3:
        return "Smooth"
    if value < 0.6:
        return "Lobulated"
    return "Irregular"


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze.py <image-path>")
        return

    try:
        models = load_models()
        results = process_image(sys.argv[1], models)

        print("\nFull Analysis Report:")
        print(json.dumps(results, indent=2))

        if results.get("tumor_detected"):
            print("\nClinical Summary:")
            print(f"Tumor Type: {results.get('tumor_type')}")
            print(f"Dimensions: {results.get('width')}x{results.get('height')} px")
            print(f"Orientation: {results.get('orientation')}")
            print(f"Acoustic Effect: {results.get('acoustic_effect')}")
            print(f"Echo Pattern: {results.get('echo_pattern')}")
            print(f"Margin: {results.get('margin_type')}")
            print(f"Visualization: {results.get('visualization')}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
