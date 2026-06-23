from fastapi import FastAPI,File,UploadFile
from ultralytics import YOLO
from PIL import Image 
import io 
import numpy as np

app=FastAPI(title="Bee Detection API")

model=YOLO("best.pt")

@app.get("/")
def home():
    return {"message":"Welcome to the Bee Detection API"} 

@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    request_object_content = await file.read()
    image = Image.open(io.BytesIO(request_object_content))
    img_array = np.array(image)
    
    # 2. عمل التوقع باستخدام الموديل
    results = model.predict(source=img_array, conf=0.40)
    
    # 3. تجميع البيانات (Boxes) عشان نرجعها كـ JSON
    detections = []
    boxes = results[0].boxes
    for box in boxes:
        # الإحداثيات [xmin, ymin, xmax, ymax]
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = model.names[cls]
        
        detections.append({
            "box": xyxy,
            "confidence": conf,
            "class_id": cls,
            "class_name": name
        })
        
    return {"total_detected": len(detections), "detections": detections}