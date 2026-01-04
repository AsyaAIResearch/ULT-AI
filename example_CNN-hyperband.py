import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from kerastuner.tuners import Hyperband
from sklearn.metrics import classification_report, confusion_matrix

training_data = r"/content/drive/MyDrive/ultrasound breast classification/train"
val_data = r"/content/drive/MyDrive/ultrasound breast classification/val"
img_width, img_height = 224, 224
batch_size = 8
epochs = 25
learning_rate = 0.001


def data_augmentation():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    return train_datagen, val_datagen


def load_train_val(train_datagen, val_datagen):
    train_generator = train_datagen.flow_from_directory(
        training_data,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )
    validation_generator = val_datagen.flow_from_directory(
        val_data,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )
    return validation_generator, train_generator

def build_Deep_CNN(hp):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def compile_model(model):
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, train_generator, validation_generator):
    history = model.fit(train_generator,
                        steps_per_epoch=train_generator.samples // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.samples // batch_size)
    return model, history

def train_model_with_early_stop(model, train_generator, validation_generator):
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(train_generator,
                        steps_per_epoch=train_generator.samples // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.samples // batch_size,
                        callbacks=[early_stopping])

    for epoch in range(0, len(history.history['accuracy']), 5):
        print(f"Epoch {epoch + 1}:")
        train_acc = history.history['accuracy'][epoch]
        val_acc = history.history['val_accuracy'][epoch]
        print(f"Training Accuracy: {train_acc}, Validation Accuracy: {val_acc}")

    return model, history

def evaluate_model(model, validation_generator):
    scores = model.evaluate(validation_generator)
    print(f"Validation Loss: {scores[0]}, Validation Accuracy: {scores[1]}")
    return scores

def calculate(validation_generator, model):
    y_val_true = validation_generator.classes
    y_val_pred_probs = model.predict(validation_generator)
    y_val_pred = (y_val_pred_probs > 0.5).astype(int)

    total_accuracy = np.sum(y_val_true == y_val_pred.flatten()) / len(y_val_true)

    return y_val_pred, y_val_pred_probs, y_val_true, total_accuracy

def prints(y_val_pred, y_val_true, total_accuracy, train_generator):
    print(f"Total Accuracy (Validation Data): {total_accuracy}")
    print("Classification Report:")
    print(classification_report(y_val_true, y_val_pred, target_names=train_generator.class_indices))

    print("Confusion Matrix:")
    print(confusion_matrix(y_val_true, y_val_pred))

def model_save(model):
    model.save('CNN_5_bm.h5')

def visualize_training_results(history, y_val_true, y_val_pred, class_labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.show()

    cm = confusion_matrix(y_val_true, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    classification_rep = classification_report(y_val_true, y_val_pred, target_names=class_labels, output_dict=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(classification_rep).T.drop('support', axis=1), annot=True, cmap='Blues')
    plt.title('Classification Report')
    plt.show()


def main():
    train_datagen, val_datagen = data_augmentation()
    validation_generator, train_generator = load_train_val(train_datagen, val_datagen)
    tuner = Hyperband(
        build_Deep_CNN,
        objective='val_accuracy',
        max_epochs=epochs,
        factor=3,
        seed=42,
        directory='my_dir',
        project_name='breast_ultrasound_hyperparameters'
    )
    tuner.search(train_generator,
                 validation_data=validation_generator,
                 epochs=epochs,
                 steps_per_epoch=train_generator.samples // batch_size,
                 validation_steps=validation_generator.samples // batch_size)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    model.fit(train_generator,
              epochs=epochs,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples // batch_size)

    evaluate_model(model, validation_generator)
    model, history = train_model(model, train_generator, validation_generator)
    y_val_pred, y_val_pred_probs, y_val_true, total_accuracy = calculate(validation_generator, model)
    print(y_val_pred, y_val_true, total_accuracy, train_generator)
    model_save(model)
    class_labels = list(train_generator.class_indices.keys())
    visualize_training_results(history, y_val_true, y_val_pred, class_labels)


if __name__ == "__main__":
    main()
