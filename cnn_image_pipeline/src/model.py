from tensorflow.keras import layers, Sequential

def build_simple_cnn(input_shape=(512, 512, 3), num_classes=5, dropout=0.0):
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model
