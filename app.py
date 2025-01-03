import torch
import uvicorn
import numpy as np
import cv2 as cv
import io
import logging

from fastapi import Depends, FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from requests import Session
from sqlalchemy import text
from PIL import Image

from database.database import get_db
from database.models import OCRText
from utils.database_operations import read_ocr_texts, save_db_log, save_to_db
from utils.image_processing import get_roi, warp_image
from utils.ocr_processing import save_image, save_text
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


#? Configure logging
logging.basicConfig(level=logging.INFO)


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load VietOCR model
try:
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = 'weights/transformerocr.pth' # path to the vietocr weights
    config['device'] = device # use the same device as the yolo models
    vietocr_model = Predictor(config)
    print("VietOCR model loaded successfully")
except Exception as e:
    logging.error(f"Error loading VietOCR model: {e}")
    exit()


# Initialize FastAPI app
app = FastAPI()
app.add_middleware( # avoid CORS errors
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin for development; limit this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

    
@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Performs object detection on an uploaded image with preprocessing.
    Args:
        file: Uploaded image file.
        debug (bool): debug mode.
    Returns:
        list: A list of detections with bounding box coordinates, confidence score, and class ID.
    """
    
    try:
        # Load the uploaded image
        image_bytes = await file.read()
        
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logging.error(f"Error opening image: {e}")
            save_db_log(db, file.filename, f"Error opening image: {e}", "Failure")
            raise HTTPException(status_code=400, detail="Invalid image file")

        image_np = np.array(image)[:, :, ::-1]  # Convert PIL image to BGR NumPy array
        file_name = file.filename

        try:
            # Perform warp image
            wrap_annotated_image, cropped_image, detections = warp_image(image_np, file_name)
        except Exception as e:
            logging.error(f"Error during warp image: {e}")
            save_db_log(db, file_name, f"Error during warp image: {e}", "Failure")
            raise HTTPException(status_code=500, detail="Error during warp image")

        try:
            # Perform ROI detection
            roi_annotated_image, roi_detections, preprocessed_roi_image = get_roi(cropped_image)
        except Exception as e:
            logging.error(f"Error during ROI detection: {e}")
            save_db_log(db, file_name, f"Error during ROI detection: {e}", "Failure")
            
            raise HTTPException(status_code=500, detail="Error during ROI detection")

        
        #? Save File to Folder
        # Save the annotated image
        save_path = save_image(wrap_annotated_image, "validation/detect", "processed", file_name)
        
        # Save the cropped image
        crop_save_path = save_image(cropped_image, "validation/cropped", "cropped", file_name)

        # Save the annotated ROI image
        roi_save_path = save_image(roi_annotated_image, "validation/roi", "roi_processed", file_name)


        #? original detections: {'botleft': (34, 417), 'botright': (424, 413), 'topleft': (42, 182), 'topright': (425, 184)}
        detections = [ [key, [int(coord) for coord in value]] for key, value in detections.items()]
        print("detections", detections)
        roi_detections = [[int(cls), float(conf), float(x1), float(y1), float(x2), float(y2)] for cls, conf, x1, y1, x2, y2 in roi_detections]
        
        # Apply VietOCR to each ROI
        recognized_texts = []
        for roi in roi_detections:
            cls, conf, x1, y1, x2, y2 = roi #? Must retrieve all values although we don't need conf
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            roi_image = preprocessed_roi_image[y1:y2, x1:x2] # Crop the ROI
            roi_image_pil = Image.fromarray(cv.cvtColor(roi_image, cv.COLOR_BGR2RGB)) # Convert to PIL image
            
            try:
                text = vietocr_model.predict(roi_image_pil)
                recognized_texts.append({
                    "class_id": cls,
                    "box": [x1, y1, x2, y2],
                    "text": text
                })
            except Exception as e:
                logging.error(f"Error during VietOCR prediction: {e}")
                recognized_texts.append({
                    "class_id": cls,
                    "box": [x1, y1, x2, y2],
                    "text": "Error"
                })

        ocr_text_path, ocr_text = save_text('validation/detect_text', recognized_texts, file_name)

        #? print for debug
        print('file_name:', file_name)
        print(f"recognized_texts: {recognized_texts}")
        print("OCR:", ocr_text) 

        content={
            "image_path": save_path,
            "crop_path": crop_save_path,
            "roi_image_path": roi_save_path,
            "ocr_text_path": ocr_text_path
        }

        # output: ocr_text = {
            # 'id': '001153023615',
            # 'name': 'NGUYỄN THỊ BÌNH',
            # 'dob': '28/09/1953',
            # 'gender': 'No',
            # 'nationality': 'Việt Nam',
            # 'origin_place': 'Gia Bình,Bắc Ninh',
            # 'current_place': '17B Hàng Đồng Hàng Bồ Hoàn Kiếm,Hà Nội',
            # 'expire_date': None
        # }
        
        #? Save OCR text data and log to the database
        save_to_db(db, ocr_text.copy())
        save_db_log(db, file_name, "OCR text data saved successfully", "Success")    
        
        #? can't return ocr_text because utf-8 (vietnamese) problems
        return JSONResponse(content=content)
        

    except HTTPException as http_exc:
        # Re-raise the HTTPException to be handled by FastAPI
        raise http_exc
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/read-ocr/") 
def get_ocr_texts(db: Session = Depends(get_db)):
    ocr_texts = read_ocr_texts(db)
    return ocr_texts


#? Clean Database cmd: curl -X POST "http://127.0.0.1:8000/clean-database/"
@app.post("/clean-database/")
def clean_database(db: Session = Depends(get_db)):
    try:
        # Run the queries to clean the database
        db.execute(text("TRUNCATE TABLE detection_logs RESTART IDENTITY CASCADE;")) # CASCADE for related tables (if any)
        db.execute(text("TRUNCATE TABLE ocr_texts RESTART IDENTITY CASCADE;"))
        
        
        # Commit the changes
        db.commit()

        return {"status": "Database cleaned successfully."}
        
    except Exception as e:
        # Rollback the transaction in case of an error
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred while cleaning the database: {str(e)}")



# Run the app using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
#TODO: read from .txt, improve app.py using OOP