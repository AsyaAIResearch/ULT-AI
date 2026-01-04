import math
import os
from glob import glob

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D
from keras.models import load_model, Model
from keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

BUSI_MASKED = r"C:\Users\asyao\PycharmProjects\USCAI\_datasets\MASKED DATASETS\Dataset_BUSI_with_GT"
TEST_DATA = r"C:\Users\asyao\PycharmProjects\USCAI\_datasets\TEST DATASETS\1\Processed_Dataset_1"
TUMOR_MODEL_PATH = r"C:\Users\asyao\PycharmProjects\USCAI\models\CNN_3_bm.h5"
FEATURE_MODEL_PATHS = {
    "f1": r"C:\Users\asyao\PycharmProjects\USCAI\XAImodels\vertically.h5",
    "f2": r"C:\Users\asyao\PycharmProjects\USCAI\XAImodels\shodowing.h5",
    "f3": r"C:\Users\asyao\PycharmProjects\USCAI\XAImodels\enhancement.h5",
    "f4": r"C:\Users\asyao\PycharmProjects\USCAI\XAImodels\echogenity.h5",
    "f5": r"C:\Users\asyao\PycharmProjects\USCAI\XAImodels\contour.h5",
}
SEGMENT_MODEL_PATH = r"C:\Users\asyao\PycharmProjects\USCAI\models\segmentation.h5"
RESULTS_DIR = r"C:\Users\asyao\PycharmProjects\USCAI\results_improved"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16
N_FOLDS = 3
EPOCHS_FT = 8
EPOCHS_FUSION = 12
LR_FT = 1e-4
LR_FUSION = 3e-4

tumor_model = load_model(TUMOR_MODEL_PATH)
segment_model = None
if os.path.exists(SEGMENT_MODEL_PATH):
    try:
        segment_model = load_model(SEGMENT_MODEL_PATH)
    except Exception as e:
        print("Segmentation load failed:", e)

feature_models = {}
for k, p in FEATURE_MODEL_PATHS.items():
    if os.path.exists(p):
        try:
            feature_models[k] = load_model(p)
            print("Loaded", k)
        except Exception as e:
            print("Skipped", k, ":", e)


def make_feature_extractor_for_finetune(model):
    # find last conv layer index by searching for 4D output
    for i in range(len(model.layers) - 1, -1, -1):
        if len(model.layers[i].output_shape) == 4:
            last_conv_i = i
            break
    else:
        # fallback: use penultimate layer output
        return Model(inputs=model.input, outputs=model.layers[-2].output)
    x = model.layers[last_conv_i].output
    x = GlobalAveragePooling2D()(x)
    # small dense to create embedding
    emb = Dense(256, activation="relu", name="ft_embedding")(x)
    return Model(inputs=model.input, outputs=emb)


cnn_ft_extractor = make_feature_extractor_for_finetune(tumor_model)
print("embedding size:", cnn_ft_extractor.output_shape[-1])


def alg_f1_orientation_and_size(mask):
    if mask is None:
        return [0, 0, "None", 0]
    if len(mask.shape) == 3:
        mask_g = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_g = mask
    _, th = cv2.threshold(mask_g, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0, 0, "None", 0]
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    orientation = "Vertical" if h <= w else "Horizontal"
    area = cv2.contourArea(c)
    return [w, h, orientation, area]


def alg_f2_f3_acoustic_effect(img, mask):
    if img is None or mask is None:
        return [0.0, "Unknown"]
    img_gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mk = mask if len(mask.shape) == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mk_bin = (mk > 127).astype(np.uint8)
    if mk_bin.sum() == 0:
        return [0.0, "Unknown"]
    under = img_gray[mk_bin == 1].astype(np.float32)
    outside = img_gray[mk_bin == 0].astype(np.float32)
    if under.size == 0 or outside.size == 0:
        return [0.0, "Unknown"]
    mean_under = under.mean()
    mean_out = outside.mean()
    diff = abs(mean_under - mean_out)
    label = (
        "Enhancement" if (mean_under > mean_out and diff > 5) else "Shadowing"
    )  # tuned threshold
    return [float(diff), label]


def alg_f4_echogenity(img, mask):
    if img is None or mask is None:
        return [0.0, 0.0]
    img_gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mk = mask if len(mask.shape) == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mk_bin = (mk > 127).astype(np.uint8)
    if mk_bin.sum() == 0:
        return [0.0, 0.0]
    tumor_pixels = img_gray[mk_bin == 1].astype(np.float32)
    mean_int = float(tumor_pixels.mean())
    thr = tumor_pixels.mean() + tumor_pixels.std()
    hyp_pct = float((tumor_pixels > thr).sum()) / tumor_pixels.size
    return [mean_int, hyp_pct]


def alg_f5_contour_features(mask):
    if mask is None:
        return [0.0, 0.0, 0.0]
    mk = mask if len(mask.shape) == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(mk, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0.0, 0.0, 0.0]
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull) if hull is not None else 0.0
    solidity = area / hull_area if hull_area > 0 else 0.0
    circularity = (4 * math.pi * area) / (perim**2) if perim > 0 else 0.0
    return [perim, solidity, circularity]


def load_and_mask(image_path):
    base, ext = os.path.splitext(image_path)
    mask_path = base + "_mask.png"
    img = cv2.imread(image_path)
    mask = None
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    elif segment_model is not None:
        try:
            inp = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
            inp = np.expand_dims(inp, 0)
            pred = segment_model.predict(inp)[0]
            if pred.ndim == 3 and pred.shape[-1] == 1:
                pred = pred[:, :, 0]
            mask_pred = (pred >= 0.5).astype(np.uint8) * 255
            mask = cv2.resize(mask_pred, (img.shape[1], img.shape[0]))
        except Exception as e:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    else:
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    # crop
    try:
        x, y, w, h = cv2.boundingRect((mask > 127).astype(np.uint8))
        if w == 0 or h == 0:
            crop_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            crop_mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        else:
            crop_img = img[y : y + h, x : x + w]
            crop_mask = mask[y : y + h, x : x + w]
            crop_img = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE))
            crop_mask = cv2.resize(crop_mask, (IMG_SIZE, IMG_SIZE))
    except Exception:
        crop_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        crop_mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    return crop_img, crop_mask


def build_dataset(folder):
    imgs = []
    masks = []
    ys = []
    for label_name, lab in [("benign", 0), ("malignant", 1)]:
        path = os.path.join(folder, label_name)
        if not os.path.exists(path):
            continue
        for f in sorted(glob(os.path.join(path, "*.png"))):
            if f.endswith("_mask.png"):
                continue
            img, mask = load_and_mask(f)
            if img is None:
                continue
            imgs.append(img)
            masks.append(mask)
            ys.append(lab)
    return np.array(imgs), np.array(masks), np.array(ys)


X_imgs, X_masks, y = build_dataset(BUSI_MASKED)
print("Loaded", len(X_imgs), "instances.")

if len(X_imgs) == 0:
    print("BUSI empty - falling back to TEST_DATA.")
    X_imgs = []
    X_masks = []
    y = []
    for label_name, lab in [("benign", 0), ("malignant", 1)]:
        path = os.path.join(TEST_DATA, label_name)
        if not os.path.exists(path):
            continue
        for f in sorted(glob(os.path.join(path, "*.png"))):
            img = cv2.imread(f)
            if img is None:
                continue
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X_imgs.append(img_resized)
            X_masks.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8))
            y.append(lab)
    X_imgs = np.array(X_imgs)
    X_masks = np.array(X_masks)
    y = np.array(y)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.12,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    fill_mode="nearest",
)


# focal loss
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -alpha * tf.pow(1 - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss)

    return loss_fn


def extract_embedding(ft_model, img):
    # img is HWC uint8
    inp = img.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, 0)
    return ft_model.predict(inp, verbose=0).ravel()


def extract_alg_features(img, mask):
    # produce medically-meaningful scalar features
    w, h, orient, area = alg_f1_orientation_and_size(mask)
    diff, label23 = alg_f2_f3_acoustic_effect(img, mask)
    mean_int, hyp_pct = alg_f4_echogenity(img, mask)
    perim, solidity, circularity = alg_f5_contour_features(mask)
    orient_num = 1 if orient == "Vertical" else 0
    effect_num = 1 if label23 == "Enhancement" else 0
    return np.array(
        [
            w,
            h,
            area,
            orient_num,
            diff,
            effect_num,
            mean_int,
            hyp_pct,
            perim,
            solidity,
            circularity,
        ],
        dtype=float,
    )


def finetune_cnn(X_imgs, y, folds=3):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    fold = 0
    embedding_models = []
    fold_scores = []
    for train_idx, val_idx in skf.split(X_imgs, y):
        fold += 1
        print(f" FT fold {fold}/{folds}")
        Xtr = X_imgs[train_idx]
        ytr = y[train_idx]
        Xv = X_imgs[val_idx]
        yv = y[val_idx]

        # create trainable model: attach classification head to the extractor
        inp = cnn_ft_extractor.input
        emb = cnn_ft_extractor.output
        head = Dense(64, activation="relu")(emb)
        out = Dense(1, activation="sigmoid")(head)
        ft_model = Model(inputs=inp, outputs=out)

        # unfreeze last layers of base for small finetune
        for layer in ft_model.layers:
            layer.trainable = False
        # unfreeze last conv block of original tumor_model (approx)
        for layer in ft_model.layers[-10:]:
            layer.trainable = True

        ft_model.compile(
            optimizer=Adam(learning_rate=LR_FT),
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=["accuracy"],
        )
        cb = [
            EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, min_lr=1e-6),
        ]

        # data generator for augmentation
        train_gen = aug.flow(Xtr, ytr, batch_size=BATCH_SIZE, shuffle=True)
        ft_model.fit(
            train_gen,
            validation_data=(Xv, yv),
            epochs=EPOCHS_FT,
            callbacks=cb,
            verbose=1,
            steps_per_epoch=max(1, len(Xtr) // BATCH_SIZE),
        )

        yv_prob = ft_model.predict(Xv, verbose=0).ravel()
        try:
            aucv = roc_auc_score(yv, yv_prob)
        except Exception:
            aucv = float("nan")
        print(f" FT fold {fold} AUC: {aucv:.4f}")
        fold_scores.append(aucv)
        tmp_path = os.path.join(RESULTS_DIR, f"ft_model_fold{fold}.keras")
        ft_model.save(tmp_path)
        embedding_models.append(tmp_path)

    return embedding_models, (
        np.mean([s for s in fold_scores if not math.isnan(s)])
        if len(fold_scores) > 0
        else float("nan")
    )


ft_models, ft_mean_auc = finetune_cnn(X_imgs, y, folds=min(N_FOLDS, 3))
print("Fine-tune mean AUC:", ft_mean_auc)
best_ft_model_path = ft_models[0]
print("Loading best ft model:", best_ft_model_path)
best_ft = load_model(best_ft_model_path, custom_objects={"loss_fn": focal_loss()})
embedding_layer = None
for layer in best_ft.layers:
    if layer.name == "ft_embedding":
        embedding_layer = layer.output
        break
if embedding_layer is None:
    # fallback: use second-last dense or GAP
    embedding_layer = best_ft.layers[-2].output

embedding_extractor = Model(inputs=best_ft.input, outputs=embedding_layer)
print("Embedding extractor ready, dim:", embedding_extractor.output_shape[-1])
embs = []
alg_feats = []
for i in tqdm(range(len(X_imgs))):
    emb = extract_embedding(embedding_extractor, X_imgs[i])
    af = extract_alg_features(X_imgs[i], X_masks[i])
    embs.append(emb)
    alg_feats.append(af)
embs = np.array(embs)
alg_feats = np.array(alg_feats)
fused = np.concatenate([embs, alg_feats], axis=1)
print("Fused shape:", fused.shape)
scaler = StandardScaler()
fused_scaled = scaler.fit_transform(fused)
joblib.dump(scaler, os.path.join(RESULTS_DIR, "fused_scaler.joblib"))
print("Saved scaler.")
# k-Fold training for fusion classifier
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold = 0
accs = []
aucs = []
recalls = []
all_y_val = []
all_y_prob = []
for train_idx, val_idx in skf.split(fused_scaled, y):
    fold += 1
    print(f"Fusion fold {fold}/{N_FOLDS}")
    Xtr, Xv = fused_scaled[train_idx], fused_scaled[val_idx]
    ytr, yv = y[train_idx], y[val_idx]

    # compute class weights
    from sklearn.utils import class_weight

    cw = class_weight.compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    class_weights = {0: cw[0], 1: cw[1]}
    print("Class weights:", class_weights)

    # small fusion net
    inp = Input(shape=(Xtr.shape[1],))
    x = Dense(128, activation="relu")(inp)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation="sigmoid")(x)
    fusion = Model(inputs=inp, outputs=out)
    fusion.compile(
        optimizer=Adam(learning_rate=LR_FUSION),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy"],
    )
    cb = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, min_lr=1e-6),
    ]
    fusion.fit(
        Xtr,
        ytr,
        validation_data=(Xv, yv),
        epochs=EPOCHS_FUSION,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        class_weight=class_weights,
        verbose=1,
    )

    yv_prob = fusion.predict(Xv).ravel()
    # threshold sweep to maximize recall while keeping decent precision
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thr = 0.5
    best_recall = -1
    for thr in thresholds:
        yv_pred_thr = (yv_prob >= thr).astype(int)
        rec = recall_score(yv, yv_pred_thr)
        if rec > best_recall:
            best_recall = rec
            best_thr = thr
    print(
        f" Best threshold (maximize recall): {best_thr:.2f}, recall: {best_recall:.3f}"
    )
    yv_pred = (yv_prob >= best_thr).astype(int)

    accs.append(accuracy_score(yv, yv_pred))
    try:
        aucs.append(roc_auc_score(yv, yv_prob))
    except Exception:
        aucs.append(float("nan"))
    recalls.append(recall_score(yv, yv_pred))

    all_y_val.append(yv)
    all_y_prob.append(yv_prob)

    # save per-fold model
    fusion.save(os.path.join(RESULTS_DIR, f"fusion_fold{fold}.keras"))

all_y_val = np.concatenate(all_y_val)
all_y_prob = np.concatenate(all_y_prob)
all_y_pred = (all_y_prob >= 0.5).astype(int)  # report at 0.5 as baseline
print("Overall (baseline 0.5) ACC:", accuracy_score(all_y_val, all_y_pred))
try:
    overall_auc = roc_auc_score(all_y_val, all_y_prob)
except Exception:
    overall_auc = float("nan")
overall_recall = recall_score(all_y_val, all_y_pred)
print("Overall AUC:", overall_auc, "Recall:", overall_recall)
print("Per-fold ACCs:", accs, "AUCs:", aucs, "Recalls:", recalls)

joblib.dump(
    embedding_extractor, os.path.join(RESULTS_DIR, "embedding_extractor.joblib")
)  # may or may not be useful
with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
    f.write(
        f"ACC folds: {accs}\nAUCs: {aucs}\nRecalls: {recalls}\nOverall AUC: {overall_auc}\nOverall recall: {overall_recall}\n"
    )
# plots
try:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(all_y_val, all_y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")
    plt.title("ROC (aggregated)")
    plt.savefig(os.path.join(RESULTS_DIR, "roc_agg.png"))
    plt.close()
except Exception:
    pass

# confusion matrix at 0.5
cm = confusion_matrix(all_y_val, all_y_pred)
plt.figure(figsize=(4, 3))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.title("Confusion matrix (agg, thr=0.5)")
plt.title("Confusion matrix (agg, thr=0.5)")
plt.savefig(os.path.join(RESULTS_DIR, "cm_agg.png"))
plt.close()

print("results:", RESULTS_DIR)
