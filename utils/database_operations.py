from fastapi import HTTPException
from requests import Session
from database.models import DetectionLog, OCRText
from sqlalchemy.orm import Session
from database.database import engine
from database import models
from database.models import OCRText, DetectionLog 

models.Base.metadata.create_all(bind=engine) # create database tables


def save_db_log(db: Session, file_name: str, message: str, status: str = "Failure"):
    try:
        # Create DetectionLog object
        detection_log = DetectionLog(
            status=status,
            message=message,
            file_name=file_name,
        )
        # Add DetectionLog to the session
        db.add(detection_log)
        db.commit()
        db.refresh(detection_log)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred while saving the log: {e}")


def save_to_db(db: Session, ocr_text_data: dict):
    try:
        # Create OCRText object
        ocr_text = OCRText(
            id_number=ocr_text_data.get("id"),
            name=ocr_text_data.get("name"),
            dob=ocr_text_data.get("dob"),
            gender=ocr_text_data.get("gender"),
            nationality=ocr_text_data.get("nationality"),
            origin_place=ocr_text_data.get("origin_place"),
            current_place=ocr_text_data.get("current_place"),
            expire_date=ocr_text_data.get("expire_date"),
        )
        
        # Add OCRText to the session
        db.add(ocr_text)
        db.commit()
        db.refresh(ocr_text)
        
        return ocr_text
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred while saving the OCR text: {e}")


#? READ ALL: curl -X POST "http://127.0.0.1:8000/read-ocr/"
# Function to read data from ocr_texts table
def read_ocr_texts(db: Session):
    try:
        ocr_texts = db.query(OCRText).all()
        return ocr_texts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Endpoint to read data from ocr_texts table
