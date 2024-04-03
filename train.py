from ultralytics import settings
from ultralytics import YOLO

settings.update({'datasets_dir': './checkbox_state_v2/data/'})

# View all settings
print(settings)

# Return a specific setting
value = settings['runs_dir']

model = YOLO('runs/classify/train4/weights/best.pt')
model.train(cfg='args.yaml',data='./checkbox_state_v2/data/', epochs=500, imgsz=64)