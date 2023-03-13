from ultralytics import YOLO

# Load model with trained weights from HW3_YOLO.ipynb
model = YOLO("bdd100k.pt")

results = model.predict(source="0", show=True)
print(results)