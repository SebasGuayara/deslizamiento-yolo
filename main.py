import pathlib

from ultralytics import YOLO

PROJECT_NAME = "deslizamiento_deteccion"
MODEL = "yolov8n-cls.pt"
N_EPOCHS = 50
IM_SIZE = 3000
BATCH_SIZE = 1
# project_location = pathlib.Path().absolute() / f"data/{PROJECT_NAME}"
model = YOLO(MODEL)
res_train = model.train(
    data='/pscratch/sd/j/jarh1992/landslides-prj/data/deslizamiento_deteccion/',
    epochs=N_EPOCHS,
    imgsz=IM_SIZE,
    # batch=BATCH_SIZE,
)
# res_val = model.val()
