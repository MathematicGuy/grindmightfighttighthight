import json
import logging
import os
import re
import cv2 as cv
import yaml 

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
    
    
def remove_duplicate_words_before_comma(text):
    """Removes duplicate words before commas in a string."""
    parts = text.split(',')
    cleaned_parts = []
    for part in parts:
        words = part.strip().split()
        seen = set()
        cleaned_words = []
        for word in words:
            if word not in seen:
                cleaned_words.append(word)
                seen.add(word)
        cleaned_parts.append(" ".join(cleaned_words))
    return ", ".join(cleaned_parts)


#? tự sửa lại các từ sai dễ sửa
def resub(s): 
    s = re.sub(r'Hà Nôi', 'Hà Nội', s)
    return s


def regex_ocr(recognized_texts):
    '''input
        recognized_texts = [
            {'class_id': 4, 'box': [256, 262, 482, 317], 'text': '040094018672'},
            {'class_id': 5, 'box': [180, 350, 419, 400], 'text': 'NGUYỂN HÔU SƠN'},
            {'class_id': 0, 'box': [184, 606, 480, 639], 'text': 'Năm Sai Đà Trước Nhân Đàng.'},
            {'class_id': 7, 'box': [183, 524, 482, 569], 'text': 'Nam Sơn. Đô Lương Nghệ An'},
            {'class_id': 2, 'box': [91, 590, 175, 625], 'text': '13 DS21 - 3 A P.'},
            {'class_id': 6, 'box': [526, 434, 626, 487], 'text': 'Việt Năm'},
            {'class_id': 1, 'box': [374, 401, 486, 439], 'text': '13031994'},
            {'class_id': 3, 'box': [302, 440, 354, 478], 'text': 'Nam'},
            {'class_id': 0, 'box': [440, 574, 494, 612], 'text': 'xóm xóm 1'}
        ]

        names: ['current_place', 'dob', 'expire_date', 'gender', 'id', 'name', 'nationality', 'origin_place']
    '''
    
    detect_text = {}
    for recognized_text in recognized_texts:
        text_class = roi_config['names'][recognized_text['class_id']]
        text = recognized_text['text']

        #? Reformat datatime (date to dd/mm/yyyy)
        if recognized_text['class_id'] in [1, 2]:
            if len(re.findall(r'\d', text)) < 8: # if detect less than 8 numbers
                text = "Không Thời Hạn"
                
            match = re.match(r"(\d{2})[./-]?(\d{2})[./-]?(\d{4})$", text)
            if match:
                day, month, year = match.groups()
                text = f"{day}/{month}/{year}"

    
        #? Remove duplicate words before each comma or period (for other classes)
        if recognized_text['class_id'] not in [1, 2]:
            text = resub(text) #? improve word accuracy
            parts = re.split(r'([.,])', text) # Split by comma or period and keep delimiters
            cleaned_parts = []
            
            for i, part in enumerate(parts):
                if i % 2 == 0: # process text segments only
                    words = part.split()
                    if not words:
                        cleaned_parts.append(part)
                        continue
                    cleaned_words = [words[0]]
                    for word in words[1:]:
                        if word != cleaned_words[-1]:
                            cleaned_words.append(word)
                    cleaned_parts.append(" ".join(cleaned_words).strip())
                else:
                    cleaned_parts.append(part) # keep the delimiter

            text = "".join(cleaned_parts).strip()
        
            
        if text_class in detect_text:
            #? if class is 'current_place' then append to the front bc ocr model always scan bottom text first
            if recognized_text['class_id'] == 0:
                detect_text[text_class].insert(0, text)
            else:
                detect_text[text_class].append(text)
        else:
            detect_text[text_class] = [text]
        
    #? Convert list values to strings for one-to-many key-value pai
    for key in detect_text:
        if isinstance(detect_text[key], list):
            detect_text[key] = " ".join(detect_text[key])
    
    #? Create the output dictionary because numbers of values can vary depend on detections
    ocr_text = {
        "id": detect_text.get("id"),
        "name": detect_text.get("name"),
        "dob": detect_text.get("dob"),
        "gender": detect_text.get("gender"),
        "nationality": detect_text.get("nationality"),
        "origin_place": detect_text.get("origin_place"),
        "current_place": detect_text.get("current_place"),
        "expire_date": detect_text.get("expire_date")
    }
    
    """output
    {
        "id": "001162013525",
        "name": "NGUYỄN THỊ LOAN",
        "dob": "03/12/1962",
        "origin_place": "Thanh Oai,Hà Nội",
        "current_place": "125 Bát Đành Hàng Bồ,Hoàn Kiếm,Hà Nội", 
        "expire_date": "Không Thời Hạn"
    }
    """
    return ocr_text


def save_text(folder, recognized_texts, file_name):
    # Extract the filename without extension
    base_name = os.path.splitext(file_name)[0]
    save_filename = f"ocr_{base_name}.json"
    save_path = os.path.join(folder, save_filename)

    os.makedirs(folder, exist_ok=True)
    ocr_text = regex_ocr(recognized_texts)

    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(ocr_text, file, ensure_ascii=False, indent=4)

    return save_path, ocr_text


def save_image(image, folder, prefix, filename):
    base_name = os.path.splitext(filename)[0]
    save_filename = f"{prefix}_{base_name}.jpg"
    save_path = os.path.join(folder, save_filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv.imwrite(save_path, image)
    
    return save_path
