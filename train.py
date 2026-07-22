# DO NOT EDIT: TRAINING CODE
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import os

print("Import done!")

print("Import done!")

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

MAX_FRAMES = 256
NEGATE_XZ_COORDS = True

print("Loading model architecture from model.py...")
from model import model

print("Model loaded successfully!")


class CustomEpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        msg = f"\n🚀 Starting epoch {epoch + 1}"
        print(msg)

        with open("training_log.txt", "a") as f:
            f.write(msg + "\n")

    def on_epoch_end(self, epoch, logs=None):
        msg = (
            f"✅ Epoch {epoch + 1} complete!"
            f"\nTraining:\nLoss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}"
            f"\nValidation:\nLoss: {logs['val_loss']:.4f}, Accuracy: {logs['val_accuracy']:.4f}\n"
        )

        print(msg)

        with open("training_log.txt", "a") as f:
            f.write(msg + "\n")


def process_sample(path, label):
    hands = np.load(path.numpy().decode())["hands"]

    T = hands.shape[0]

    if T > MAX_FRAMES:
        hands = hands[:MAX_FRAMES]

    elif T < MAX_FRAMES:
        padding_shape = (MAX_FRAMES - T, *hands.shape[1:])
        padding = np.zeros(padding_shape, dtype=np.float32)
        hands = np.concatenate([hands, padding], axis=0)

    if NEGATE_XZ_COORDS:
        hands[..., 0] *= -1
        hands[..., 2] *= -1

    return hands.astype(np.float32), np.int32(label)


def tf_process_sample(path, label):
    x, y = tf.py_function(process_sample, [path, label], [tf.float32, tf.int32])

    x.set_shape((MAX_FRAMES, 2, 20, 3))
    y.set_shape(())

    return x, y


def compute_topk_accuracy(model, dataset, ks=(1, 5, 10)):
    all_labels, all_logits = [], []
    for x_batch, y_batch in dataset:
        all_logits.append(model(x_batch, training=False).numpy())
        all_labels.append(y_batch.numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    sorted_preds = np.argsort(all_logits, axis=1)[:, ::-1]
    for k in ks:
        correct = np.any(sorted_preds[:, :k] == all_labels[:, None], axis=1).sum()
        print(f"Top-{k} Accuracy: {correct / len(all_labels) * 100:.2f}%")


def train_callback(
    epochs,
    batch_size,
    reduced_training=False,
    CI=False,
    disable_training=False,
    evaluate_model=False,
):
    from model import model

    data = pd.read_csv("ai_data.csv")
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

    multiplier = 4 if CI else 1
    X_run, X_test, y_run, y_test = train_test_split(
        videos, labels, test_size=0.01 * multiplier, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_run, y_run, test_size=0.03 * multiplier, random_state=42
    )

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_ds = (
        train_ds.shuffle(4096, seed=42)
        .map(
            tf_process_sample,
            num_parallel_calls=8,
            deterministic=True,
        )
        .batch(batch_size)
        .prefetch(8)
    )

    val_ds = (
        val_ds.map(
            tf_process_sample,
            num_parallel_calls=8,
            deterministic=True,
        )
        .batch(batch_size)
        .prefetch(8)
    )

    test_ds = test_ds.map(
        tf_process_sample,
        num_parallel_calls=8,
        deterministic=True,
    ).batch(batch_size)

    if (not CI) and (not reduced_training):
        checkpoint = ModelCheckpoint(
            filepath="best_model_v3.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
    )
    if CI or reduced_training:
        callbacks = [lr_scheduler, CustomEpochCallback()]
    else:
        callbacks = [checkpoint, lr_scheduler]

    if disable_training:
        print("Loading best_model_final.keras for evaluation only...")
        model = load_model("best_model_final.keras")
    elif (not CI) and (not reduced_training):
        print("Loading the best model to resume training")
        if os.path.exists("best_model_v3.keras"):
            model = load_model("best_model_v3.keras")
        else:
            print("No checkpoint found, starting fresh training.")

        # # Reduce dropout and recompile with a higher learning rate for fine-tuning
        # for layer in model.layers:
        #     if isinstance(layer, layers.Dropout):
        #         layer.rate = 0.2
        #     elif isinstance(layer, layers.Bidirectional):
        #         inner_layer = (
        #             layer.inner_layer
        #             if hasattr(layer, "inner_layer")
        #             else layer._layers[0]
        #         )
        #         # Bypass the property setter by modifying the underlying config dict directly
        #         if "dropout" in inner_layer.__dict__:
        #             inner_layer.__dict__["dropout"] = 0.2
        #         # Modify the public configuration property so it serializes correctly on save
        #         if hasattr(inner_layer, "get_config"):
        #             cfg = inner_layer.get_config()
        #             cfg["dropout"] = 0.2

    if not disable_training:
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks
        )

        if (not CI) and (not reduced_training):
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

    if evaluate_model or disable_training:
        compute_topk_accuracy(model, test_ds)

    return loss, accuracy


# Not for continuous integration runs, but useful for local testing and experimentation
a, b = train_callback(epochs=100, batch_size=32, reduced_training=False, disable_training=True, evaluate_model=True)
# bestmodel is based on the full 100 epoch training run while CI uses a reduced training run with only 10 epochs and runs on GH actions.
