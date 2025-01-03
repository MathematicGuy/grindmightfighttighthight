import os
import subprocess

# command = 'curl -X POST "http://127.0.0.1:8000/detect/" \
#     -H "Content-Type: multipart/form-data" \
#     -F "file=@D:\CODE\ML_2024_2025\Machine-Learning-2024\Project\id-card-extractor\app\\validation\images\\test3.jpg"'
    

def test_fastapi_app(folder_name):
    images_folder = f'D:\\CODE\\ML_2024_2025\\Machine-Learning-2024\\Project\\id-card-extractor\\app\\validation\\{folder_name}'
    for image_file in os.listdir(images_folder):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_folder, image_file)
            command = f'curl -X POST "http://127.0.0.1:8000/detect/" \
                -H "Content-Type: multipart/form-data" \
                -F "file=@{image_path}"'
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(f'Testing {image_file}:')
            print(result.stdout)
            print(result.stderr)



test_fastapi_app('test2')