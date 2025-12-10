import os

train_dir = "data/AI-CA-Data/train"

classes = sorted(os.listdir(train_dir))
print("Detected class names:")
for c in classes:
    print(c)
