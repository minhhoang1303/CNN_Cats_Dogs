import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


IMG_SIZE = 64


def load_data():
    """
    Load ảnh mèo và chó từ thư mục.
    
    Returns:
        X: Mảng ảnh (N, 64, 64, 3)
        y: Mảng nhãn (0=cat, 1=dog)
    """
    X, y = [], []
    
    cats_dir = r'D:\AI\CNN_Cats_Dogs\cats'
    dogs_dir = r'D:\AI\CNN_Cats_Dogs\dogs'
    
    # Load ảnh mèo (label = 0)
    for filename in os.listdir(cats_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(cats_dir, filename)
            try:
                img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
                img = np.array(img) / 255.0
                X.append(img)
                y.append(0)
            except:
                print(f"Loi doc anh: {filename}")
    
    # Load ảnh chó (label = 1)
    for filename in os.listdir(dogs_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(dogs_dir, filename)
            try:
                img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
                img = np.array(img) / 255.0
                X.append(img)
                y.append(1)
            except:
                print(f"Loi doc anh: {filename}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Da load {len(X)} anh")
    print(f"  - Meo: {sum(y == 0)}")
    print(f"  - Cho: {sum(y == 1)}")
    print(f"Kich thuoc anh: {X.shape[1:]}") 
    
    return X, y


def split_data(X, y, test_size=0.2):
    """
    Chia dữ liệu thành train và test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    np.savez_compressed('data.npz', 
                        X_train=X_train, X_test=X_test, 
                        y_train=y_train, y_test=y_test)
    print("\nDa luu data vao data.npz")
