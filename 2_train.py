import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def build_model(input_shape=(64, 64, 3)):
    """
    Xây dựng mô hình CNN cơ bản.
    
    ARCHITECTURE:
    ```
    Input(64,64,3)
         │
    Conv2D(32, 3x3, relu) ── MaxPool2D(2x2)
         │
    Conv2D(64, 3x3, relu) ── MaxPool2D(2x2)
         │
    Conv2D(128, 3x3, relu) ── MaxPool2D(2x2)
         │
       Flatten
         │
     Dense(128, relu) ── Dropout(0.5)
         │
     Dense(1, sigmoid) → Output: 0=Cat, 1=Dog
    ```
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Classifier
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    return model


def train_model(model, X_train, y_train, X_test, y_test):
    """
    Huấn luyện mô hình.
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('model.keras', monitor='val_accuracy', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        callbacks=callbacks
    )
    
    # Đánh giá trên tập test
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\n=== KET QUA ===")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return history


if __name__ == "__main__":
    print("=== LOAD DATA ===")
    data = np.load('data.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"Train: {X_train.shape}")
    print(f"Test: {X_test.shape}")
    
    print("\n=== BUILD MODEL ===")
    model = build_model()
    
    print("\n=== TRAIN MODEL ===")
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    print("\n=== XONG! Model da duoc luu ===")
