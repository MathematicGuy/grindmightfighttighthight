import logging
import os
import cv2 as cv
import numpy as np
from torch import device
import torch
from ultralytics import YOLO
import yaml
from utils.NMS import non_max_suppression_roi
from utils.crop_image import crop_image
import supervision as sv



#? Load configuration 
try:
    roi_config_path = os.path.join(os.path.dirname(__file__), '../roi-config.yaml')
    with open(roi_config_path) as file:
        roi_config = yaml.safe_load(file)
except FileNotFoundError:
    logging.error("roi-config.yaml not found. Please make sure the config file is available")
    exit() # or handle the situation depending on your logic
except yaml.YAMLError as e:
    logging.error(f"Error loading roi-config.yaml: {e}")
    exit()
    
    
#? Load YOLO model and ensure it uses the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model_detect = YOLO("weights/yolov11s-4corners.pt").to(device)
    print(f"YOLO detect-4-corners model loaded successfully on {device}")
except Exception as e:
    logging.error(f"Error loading YOYLO model: {e}")
    exit()

try:
    model_roi = YOLO("weights/yolo11n-roi-30.pt").to(device)
    print(f"YOLO detect-roi model loaded successfully on {device}")
except Exception as e:
    logging.error(f"Error loading YOLO model: {e}")
    exit()
    
    
    
def resize_image(image: np.ndarray):
    # 1. Resize the image while keeping the aspect ratio
    height, width = image.shape[:2]
    
    max_size = 640
    if height > width:
        new_height = max_size
        new_width = int(width * (max_size/height))
    else:   
        new_width = max_size
        new_height = int(height * (max_size/width))
        
    return cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)


def automatic_brightness_and_contrast(gray, clip_hist_percent):
    """
    Automatically adjusts brightness and contrast of a grayscale image.
    Args:
        gray (np.ndarray): Input grayscale image.
        clip_hist_percent (float): Percentage of histogram to clip.

    Returns:
        tuple: A tuple containing the adjusted image, alpha, and beta values.
    """
    # Calculate grayscale histogram
    hist = cv.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    auto_result = cv.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    return (auto_result, alpha, beta)


def preprocess_frame(image: np.ndarray) -> np.ndarray:
    """
    Applies preprocessing steps to the input image for YOLO object detection.
    Args:
        image (np.ndarray): Input image as a NumPy array (BGR).

    Returns:
        np.ndarray: Preprocessed image as a NumPy array (BGR), ready for YOLO.
    """
    try:
        # 1. Resize and convert to grayscale
        resized_image = resize_image(image)
        gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)

        # 2. Apply CLAHE with adaptive disk size
        tile_grid_size = tuple(max(1, s // 16) for s in gray.shape[::-1])
        clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=tile_grid_size)
        gray_clahe = clahe.apply(gray)
        
        # 3. Convert back to BGR
        processed_image = cv.cvtColor(gray_clahe, cv.COLOR_GRAY2BGR)

        return processed_image

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise


def preprocess_text_region_detection(image: np.ndarray) -> np.ndarray:
    """
    Applies preprocessing steps to the cropped image for ROI detection.
    Args:
        image (np.ndarray): Input cropped image as a NumPy array (BGR).

    Returns:
        np.ndarray: Preprocessed image as a NumPy array (BGR), ready for ROI detection.
    """
    try:
        # Convert to grayscale and remove shadows
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        rgb_planes = cv.split(gray)
        result_planes = []
        kernel = np.ones((7,7), np.uint8)
        for plane in rgb_planes:
            dilated_img = cv.dilate(plane, kernel)
            bg_img = cv.medianBlur(dilated_img, 21)
            diff_img = 255 - cv.absdiff(plane, bg_img)
            result_planes.append(diff_img)
        shadow_removed = cv.merge(result_planes)
        # Apply morphological operations and blur
        kernel = np.ones((1, 1), np.uint8)
        morph = cv.morphologyEx(shadow_removed, cv.MORPH_OPEN, kernel)
        blurred = cv.GaussianBlur(morph, (3, 3), 0)

        # Adjust brightness and contrast
        auto_result, _, _ = automatic_brightness_and_contrast(blurred, 1)
        
        return cv.cvtColor(auto_result, cv.COLOR_GRAY2BGR)

    except Exception as e:
        logging.error(f"Error during text region preprocessing: {e}")
        raise


def warp_image(image: np.ndarray, file_name):
    """
    Detects 4 corners of the ID card, applies perspective transformation, and crops the image.
    Args:
        image (np.ndarray): Input image as a NumPy array (BGR).
        file_name (str): The original filename of the image.

    Returns:
        tuple: A tuple containing the annotated image, cropped image, and detections.
    """
    try:
        #? Preprocess the image (including resizing)
        preprocessed_image = preprocess_frame(image)
        print('size:', preprocessed_image.shape)
        
        # Perform object detection with YOLO
        results = model_detect(preprocessed_image, conf=0.35)[0]
        class_ids = [model_detect.names[int(cls)] for cls in results.boxes.cls]
        print('class:', class_ids)

        #? debug features (not errors)
        if len(class_ids) < 3:
            # Save the preprocessed image to the fail_to_detect folder
            fail_dir = "validation/fail_to_detect/less_than_2_corners"
            os.makedirs(fail_dir, exist_ok=True)
            fail_path = f'{fail_dir}/fail_{file_name}.jpg'
            cv.imwrite(fail_path, preprocessed_image)
            logging.error(f"Failed to reconstruct 4 corners, saving preprocessed image to {fail_path}")

            raise ValueError("Detection Error: ID card detected less than 2 corners")
            
        
        # Process results
        detections = []
        
        for result in results:
            # tensor list format
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates [x1, y1, x2, y2]
            confidence = result.boxes.conf.cpu().numpy()  # Confidence score
            class_id = result.boxes.cls.cpu().numpy()  # Class ID

            for box, conf, cls in zip(boxes, confidence, class_id):
                detections.append([int(cls), float(conf), *box])
        
        logging.info(f"Detections: {detections}")
        # print(f"Detections: {detections}")
        
        #? Crop Image & report if fail to reconstruct 4 corners base on 2 corners of the parallelogram 
        annotated_image, cropped_image, source_points = crop_image(preprocessed_image, detections, file_name)
        
        return annotated_image, cropped_image, source_points
    
    except Exception as e:
        logging.error(f"Error in warp_image: {e}")
        raise
    

def get_roi(image: np.ndarray):
    """
    Detects regions of interest (ROIs) in the cropped image.
    Args:
        image (np.ndarray): Input cropped image as a NumPy array (BGR).

    Returns:
        tuple: A tuple containing the annotated ROI image, roi detections, and preprocessed ROI image.
    """
    try:
        #! Fix bug: return 2 corners instead of 4 corners
        #? ROI Detection
        preprocessed_roi_image = preprocess_text_region_detection(image)
        roi_results = model_roi(preprocessed_roi_image, conf=0.35)[0]

        # Process results
        roi_detections = []
        for result in roi_results:
            # tensor list format
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates [x1, y1, x2, y2]
            confidence = result.boxes.conf.cpu().numpy()  # Confidence score
            class_id = result.boxes.cls.cpu().numpy()  # Class ID
            
            for box, conf, cls in zip(boxes, confidence, class_id):
                roi_detections.append([int(cls), float(conf), *box])

        # logging.info(f"ROI Detections before NMS: {roi_detections}")
        # print(f"ROI Detections before NMS: {roi_detections}")

        # Apply NMS to roi_detections
        nms_roi_detections = non_max_suppression_roi(roi_detections, iou_threshold=0.3, threshold=0.45)
        
        logging.info(f"ROI Detections after NMS: {nms_roi_detections}")
        print(f"ROI Detections after NMS: {nms_roi_detections}")

        #? Get roi bboxes annotations 
        if nms_roi_detections:
            roi_detections_sv = sv.Detections(
                xyxy=np.array([detection[2:] for detection in nms_roi_detections]),
                confidence=np.array([detection[1] for detection in nms_roi_detections]),
                class_id=np.array([detection[0] for detection in nms_roi_detections])
            )

            # Annotate the ROI image using supervision
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
            roi_annotated_image = preprocessed_roi_image.copy()
            roi_annotated_image = box_annotator.annotate(roi_annotated_image, detections=roi_detections_sv)
            roi_labels = [f"{roi_config['names'][class_id]} {confidence:.2f}" for class_id, confidence in zip(roi_detections_sv.class_id, roi_detections_sv.confidence)]
            roi_annotated_image = label_annotator.annotate(roi_annotated_image, detections=roi_detections_sv, labels=roi_labels)
        else:
            roi_detections_sv = sv.Detections.empty()
            roi_annotated_image = preprocessed_roi_image.copy()
            nms_roi_detections = [] # return empty list if no detection

        return roi_annotated_image, nms_roi_detections, preprocessed_roi_image

    except Exception as e:
        logging.error(f"Error in get_roi: {e}")
        raise

