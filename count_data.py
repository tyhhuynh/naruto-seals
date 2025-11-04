import os
import tensorflow as ts

def count_images(folder_path):
    total = 0
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg'))])
            print(f"{class_folder}: {count} images")
            total += count
    return total


print("training images:")
train_count = count_images('data/train_data')
print(f"total training images: {train_count}")

print("\ntest images:")
test_count = count_images('data/test_data')
print(f"total test images: { test_count}")

