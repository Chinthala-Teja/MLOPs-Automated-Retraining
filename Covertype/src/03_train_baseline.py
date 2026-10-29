import json

import numpy as np
from tensorflow import keras

from covtype_pipeline import (
    MODEL_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    TRAIN_BATCHES,
    VALIDATION_BATCHES,
    batch_bounds,
    classification_metrics,
    dense_rows,
    load_processed_data,
)


def main() -> None:
    features, labels, manifest, stream_metadata = load_processed_data()
    train_start, _ = batch_bounds(manifest, TRAIN_BATCHES.start)
    _, train_end = batch_bounds(manifest, TRAIN_BATCHES.stop - 1)
    validation_start, _ = batch_bounds(manifest, VALIDATION_BATCHES.start)
    _, validation_end = batch_bounds(manifest, VALIDATION_BATCHES.stop - 1)

    X_train = dense_rows(features, train_start, train_end)
    y_train = labels[train_start:train_end].astype(np.float32)
    X_validation = dense_rows(features, validation_start, validation_end)
    y_validation = labels[validation_start:validation_end].astype(np.int8)

    keras.utils.set_random_seed(RANDOM_SEED)
    model = keras.Sequential(
        [
            keras.Input(shape=(features.shape[1],)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation.astype(np.float32)),
        epochs=50,
        batch_size=1024,
        callbacks=[early_stopping],
        verbose=1,
    )

    probabilities = model.predict(X_validation, batch_size=2048, verbose=0).ravel()
    metrics = classification_metrics(y_validation, probabilities)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    metadata = {
        "model_version": "v1",
        "algorithm": "Keras binary classifier",
        "architecture": [
            f"{features.shape[1]} input features",
            "Dense(32, relu)",
            "Dropout(0.2)",
            "Dense(16, relu)",
            "Dense(1, sigmoid)",
        ],
        "features": stream_metadata["features"],
        "target": stream_metadata["target"],
        "training_batches": [TRAIN_BATCHES.start, TRAIN_BATCHES.stop - 1],
        "validation_batches": [
            VALIDATION_BATCHES.start,
            VALIDATION_BATCHES.stop - 1,
        ],
        "training_records": int(len(y_train)),
        "validation_records": int(len(y_validation)),
        "input_preprocessing": "No new scaler; supplied features are already scaled.",
        "random_seed": RANDOM_SEED,
        "epochs_completed": len(history.history["loss"]),
        "metrics": metrics,
        "note": "The dataset order is preserved but is not verified temporal order.",
    }
    with MODEL_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nModel V1 validation metrics")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {MODEL_METADATA_PATH}")


if __name__ == "__main__":
    main()
