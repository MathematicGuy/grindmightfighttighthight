from sqlalchemy import Column, Enum, Integer, String, Float, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime
from sqlalchemy import Text


# Users Table Model
class OCRText(Base):
    __tablename__ = "ocr_texts"
    id_text = Column(Integer, primary_key=True, index=True, autoincrement=True)
    id_number = Column(String, nullable=True) 
    name = Column(String, nullable=True)
    dob = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    nationality = Column(String, nullable=True)
    origin_place = Column(Text, nullable=True)
    current_place = Column(Text, nullable=True)
    expire_date = Column(String, nullable=True)


class DetectionLog(Base):
    __tablename__ = "detection_logs"
    log_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(TIMESTAMP(timezone=True), default=datetime.utcnow) # Store timezone information
    status = Column(Enum("Success", "Failure", "Partial", name="status_enum"), nullable=True)
    message = Column(Text, nullable=True)
    file_name = Column(String, nullable=True)
