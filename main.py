from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import torch
import io
import base64


app = FastAPI(
    title="CCCD OCR API",
    description="API để nhận dạng và trích xuất thông tin từ ảnh CCCD/CMND",
    version="1.0.0"
)


class OCRResult(BaseModel):
    """Kết quả OCR từ ảnh CCCD"""
    data: Dict[str, str]
    confidence: float
    message: str

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    message: str


class ModelManager:
    def __init__(self):
        self.detector = None
        self.yolo_model = None
        self.device = None
        
    def initialize(self, model_path: str = "best.pt"):
        """Khởi tạo các models"""
        # Cấu hình VietOCR
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained'] = False
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        config['device'] = self.device
        
        # Fix cho PIL ANTIALIAS
        if not hasattr(Image, 'ANTIALIAS'):
            Image.ANTIALIAS = Image.LANCZOS
        
        # Khởi tạo VietOCR
        self.detector = Predictor(config)
        
        # Khởi tạo YOLO
        self.yolo_model = YOLO(model_path)
        
        print(f"Models initialized on device: {self.device}")


model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():

    try:

        model_path = "D:/Huewaco/OCR CCCD/model/best.pt"  # Hoặc đường dẫn tuyệt đối
        model_manager.initialize(model_path)
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        raise

@app.get("/", response_model=HealthCheck)
async def root():
    """Health check endpoint"""
    return HealthCheck(
        status="ok",
        message="CCCD OCR API is running"
    )

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Kiểm tra sức khỏe của API"""
    if model_manager.detector is None or model_manager.yolo_model is None:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    return HealthCheck(
        status="healthy",
        message=f"API is running on {model_manager.device}"
    )

def process_image(image_bytes: bytes) -> tuple:
    """Xử lý ảnh từ bytes"""

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Cannot decode image")
    
    return image

def extract_ocr_data(image: np.ndarray) -> Dict[str, str]:
    """Trích xuất dữ liệu OCR từ ảnh"""

    results = model_manager.yolo_model.predict(
        source=image, 
        conf=0.5, 
        save=False, 
        verbose=False
    )
    
    ocr_results = []
    total_confidence = 0
    count = 0
    

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model_manager.yolo_model.names[cls]
            
            # Crop vùng chứa text
            crop = image[y1:y2, x1:x2]
            
            # OCR bounding box
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            text = model_manager.detector.predict(img).strip()
            
            ocr_results.append({label: text})
            total_confidence += conf
            count += 1
    
   
    merged = {}
    for item in ocr_results:
        for key, value in item.items():
            if key not in merged:
                merged[key] = [value]
            else:
                merged[key].append(value)
    
    # Xử lý duplicate labels
    for key in merged:
        if len(merged[key]) > 1:
            merged[key] = ', '.join(sorted(merged[key], key=len))
        else:
            merged[key] = merged[key][0]
    
    avg_confidence = total_confidence / count if count > 0 else 0
    
    return merged, avg_confidence

@app.post("/ocr/predict", response_model=OCRResult)
async def predict_ocr(file: UploadFile = File(...)):

    try:
       
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        file_ext = '.' + file.filename.split('.')[-1] if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions and file.content_type not in ["image/jpeg", "image/jpg", "image/png", "application/octet-stream"]:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Please upload jpg, jpeg, or png. Got: {file.content_type}"
            )
        
        
        contents = await file.read()
        image = process_image(contents)  
        data, confidence = extract_ocr_data(image)
        
        return OCRResult(
            data=data,
            confidence=round(confidence, 3),
            message="OCR completed successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/ocr/predict-base64", response_model=OCRResult)
async def predict_ocr_base64(image_base64: str):
    try:
      
        image_bytes = base64.b64decode(image_base64)       
       
        image = process_image(image_bytes)
                
        data, confidence = extract_ocr_data(image)
        
        return OCRResult(
            data=data,
            confidence=round(confidence, 3),
            message="OCR completed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)