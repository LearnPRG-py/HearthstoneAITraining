# DO NOT EDIT: TRAINING CODE
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
tf.random.set_seed(42)

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

MAX_FRAMES = 256

print("Loading model architecture from model.py...")
from model import model

print("Model loaded successfully!")


class CustomEpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n🚀 Starting epoch {epoch + 1}")

    def on_epoch_end(self, epoch, logs=None):
        print(
            f"✅ Epoch {epoch + 1} complete!"
            f"\nTraining:\nLoss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}"
            f"\nValidation:\nLoss: {logs['val_loss']:.4f}, Accuracy: {logs['val_accuracy']:.4f}"
        )


def process_sample(path, label):
    hands = np.load(path.numpy().decode())["hands"]

    T = hands.shape[0]

    if T > MAX_FRAMES:
        hands = hands[:MAX_FRAMES]

    elif T < MAX_FRAMES:
        padding_shape = (MAX_FRAMES - T, *hands.shape[1:])
        padding = np.zeros(padding_shape, dtype=np.float32)
        hands = np.concatenate([hands, padding], axis=0)

    return hands.astype(np.float32), np.int32(label)


def tf_process_sample(path, label):
    x, y = tf.py_function(process_sample, [path, label], [tf.float32, tf.int32])

    x.set_shape((MAX_FRAMES, 2, 20, 3))
    y.set_shape(())

    return x, y


def train_callback(epochs, batch_size, reduced_training=False):
    data = pd.read_csv("data.csv")
    data = data.dropna(subset=["videos"])

    videos = data["videos"].astype(str).to_numpy()

    encoder = LabelEncoder()
    labels = encoder.fit_transform(data["word"])

    if reduced_training:
        _, reduced_videos, _, reduced_labels = train_test_split(
            videos,
            labels,
            test_size=25600,  # 25% of full dataset
            stratify=labels,
            random_state=42,
        )
        videos = reduced_videos
        labels = reduced_labels

    X_run, X_test, y_run, y_test = train_test_split(
        videos, labels, test_size=0.01, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_run, y_run, test_size=0.03, random_state=42
    )

    import time
    import numpy as np

    t0 = time.time()
    for i in range(20):
        p = str(videos[i])
        print("loading:", p)
        _ = np.load(p)
    print("DONE loading test batch in:", time.time() - t0)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_ds = (
        train_ds.shuffle(4096)
        .map(tf_process_sample, num_parallel_calls=8)
        .batch(batch_size)
        .prefetch(1)
    )

    val_ds = (
        val_ds.map(tf_process_sample, num_parallel_calls=8)
        .batch(batch_size)
        .prefetch(1)
    )

    test_ds = test_ds.map(tf_process_sample, num_parallel_calls=8).batch(batch_size)

    checkpoint = ModelCheckpoint(
        filepath="best_model.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
    )

    callbacks = [checkpoint, lr_scheduler]
    if reduced_training:
        callbacks.append(CustomEpochCallback())

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks
    )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_plots.png")
    plt.close()

    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print("Test Results returned:")
    print("Loss: {:.4f}, Accuracy: {:.4f}".format(loss, accuracy))

    return loss, accuracy


a, b = train_callback(epochs=100, batch_size=256, reduced_training=False)
