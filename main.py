import pathlib

from ultralytics import YOLO

PROJECT_NAME = "deslizamiento_deteccion"
MODEL = "yolov8n.pt"
N_EPOCHS = 50
IM_SIZE = 3000
BATCH_SIZE = 1
project_location = pathlib.Path().absolute() / f"data/{PROJECT_NAME}"
model = YOLO(MODEL)
res_train = model.train(
    data=f"{project_location}/dataset.yaml",
    epochs=N_EPOCHS,
    imgsz=IM_SIZE,
    batch=BATCH_SIZE,
)
res_val = model.val()
