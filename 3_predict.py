import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


IMG_SIZE = 64


def load_model():
    """Load model đã train."""
    from tensorflow.keras.models import load_model
    model = load_model('model.keras')
    return model


def predict_image(model, img_path, show=True):
    """
    Dự đoán một ảnh và hiển thị kết quả.
    """
    img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_input = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    
    pred = model.predict(img_input, verbose=0)[0][0]
    
    if pred > 0.5:
        result = "CHO (Dog)"
        conf = pred
    else:
        result = "MEO (Cat)"
        conf = 1 - pred
    
    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        
        color = 'green' if conf > 0.5 else 'red'
        label = f"{result}\nDo chap nhan: {conf:.1%}"
        
        plt.title(label, fontsize=16, color=color, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    return result, conf


def predict_folder(model, folder_path, label):
    """Dự đoán tất cả ảnh trong thư mục."""
    correct = 0
    total = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            result, conf = predict_image(model, img_path, show=False)
            
            is_correct = (label == 0 and "MEO" in result) or (label == 1 and "CHO" in result)
            
            if is_correct:
                correct += 1
            total += 1
            
            print(f"{filename}: {result} ({conf:.2f}) - {'✓' if is_correct else 'X'}")
    
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{total})")
    
    return accuracy


if __name__ == "__main__":
    print("=== LOAD MODEL ===")
    model = load_model()
    
    test_folder = r'D:\AI\CNN_Cats_Dogs\img'
    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"\n=== TEST {len(image_files)} ANH TRONG FOLDER ===")
    print(f"Folder: {test_folder}")
    
    for i, filename in enumerate(image_files):
        img_path = os.path.join(test_folder, filename)
        print(f"\n--- Anh {i+1}/{len(image_files)}: {filename} ---")
        result, conf = predict_image(model, img_path, show=True)
        print(f"Ket qua: {result} (do chap nhan: {conf:.1%})")
