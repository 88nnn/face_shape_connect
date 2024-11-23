# 필요한 라이브러리와 모듈 임포트
from flask import Flask, redirect, render_template, request, jsonify
import cv2
import numpy as np
import dlib
from PIL import Image as PILImage
import requests  # Flask와 Spring 서버 간 통신을 위한 requests 모듈 추가
from preprocess_image import preprocess_image
from classify_face_shape import classify_face_shape
from align_face import align_face
import os

# Flask 앱 초기화
app = Flask(__name__)

current_dir = os.path.dirname(__file__)
dat_path = os.path.join(
    current_dir, 'model', 'shape_predictor_68_face_landmarks.dat', 'shape_predictor_68_face_landmarks.dat'
)
# 파일이 존재하는지 확인
if not os.path.isfile(dat_path):
    raise FileNotFoundError(f"File not found at: {dat_path}")

face_detector = dlib.get_frontal_face_detector()
#landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#dat_path = r"C:\Users\ljcho\PycharmProjects\face_shape_detect\face_shape_detect\.venv\model\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"
landmark_predictor = dlib.shape_predictor(dat_path)

@app.before_request
def log_request_info():
    print(f"Headers: {request.headers}")
    print(f"Form Data: {request.form}")
    print(f"Files: {request.files}")
"""
# 웹 페이지 렌더링을 위한 라우트
@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', result="Upload an image to detect face shape.")
"""
"""
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def detect_face_shape():
    # 이미지와 userId가 요청에 포함되어 있는지 확인
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({"error": "No image file provided"}), 400
    
    try:
        # 이미지 읽기
        image_file = request.files['image'].read()
        img = preprocess_image(image_file)
        if img is None:
            return jsonify({"error": "Invalid or corrupt image"}), 400
"""
user_id = 2
@app.route('/', methods=["GET", "POST"])
# 얼굴형 판독 및 저장 API
@app.route('/upload', methods=['POST'])
@app.route('/faceshape/save', methods=['POST'])
def detect_face_shape():
    try:
    # 이미지와 userId가 요청에 포함되어 있는지 확인
        if 'image' not in request.files:
             image_file = request.files['image'].read()
             if image_file.filename == '':
                return jsonify({"error": "No image selected"}), 400
             # 이미지 처리 로직
             response = {"message": "Image received and saved successfully"}
             return jsonify(response)
        
        # 이미지 전처리 (크기 조정, 색상 변환 등)
        img = preprocess_image(image_file)
        if img is None:
            return jsonify({"error": "No image detected in the uploaded file"}), 400
        if img.filename == '':
            return jsonify({"error": "Empty file name"}), 400

        # 얼굴 감지
        faces = face_detector(img)
        if len(faces) == 0:
            return jsonify({"error": "No face detected"}), 400

        # 얼굴형 분류
        for face in faces:
            landmarks = landmark_predictor(img, face)
            face_shape = classify_face_shape(landmarks)
            app.logger.info(f"Face shape detected: {face_shape}")
            if face_shape and face_shape != "Unknown":
                response = requests.post(
                    'http://localhost:8080/faceshape/save/{userId}'.format(userId=user_id),
                    json={"faceShape": face_shape}
                )
                response.raise_for_status()
                app.logger.info(f"POST response: {response.status_code}, {response.text}")

                return jsonify({"face_shape": face_shape}), 200
            
        return jsonify({"error": "Unable to classify face shape"}), 400
                
    except Exception as e:
            app.logger.error(f"An error occurred: {e}")
            return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)



"""
cap = cv2.VideoCapture(0) # 노트북 웹캠을 카메라로 사용
cap.set(3,640) # 너비
cap.set(4,480) # 높이

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
    exit() # Exit the script if camera cannot be opened

ret, frame = cap.read() # 사진 촬영

# Check if frame is read correctly
if not ret:
    print("Error reading frame")
    exit() # Exit the script if frame cannot be read

frame = cv2.flip(frame, 1) # 좌우 대칭

cv2.imwrite('self camera test.jpg', frame) # 사진 저장

cap.release()
cv2.destroyAllWindows()

# Dlib 모델 로드: 얼굴 감지 및 랜드마크 추출
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(dat_path)

@app.route('/', methods=["GET", "POST"])
# 얼굴형 판독 및 저장 API
#@app.route('/upload', methods=['POST'])
#@app.route('/faceshape/save', methods=['POST'])
def detect_face_shape():
    user_id = 1 #테스트용
    # 이미지와 userId가 요청에 포함되어 있는지 확인
    if 'image' not in request.files: ##or 'userId' not in request.form:
    #image_path = os.path.join(current_dir, 'uploaded_image.png')
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        # 요청에서 이미지와 사용자 ID 가져오기
        image_file = request.files['image'] # .read()
    ##user_id = request.form['userId']
        # 이미지를 읽어서 저장하거나 처리
        img = image_file.read()
        #with open("received_image.jpg", "wb") as f:
         #   f.write(img)  # 저장 (테스트용)
        # 이미지 처리 로직
        response = {"message": "Image received and saved successfully"}
        return jsonify(response)
        # 이미지 파일 저장
        image_path = os.path.join(current_dir, 'uploaded_image.png')
        image_file.save(image_path)
        # 이미지 전처리 (크기 조정, 색상 변환 등)
        img = preprocess_image(image_file)
        if img is None:
            return jsonify({"error": "No image detected in the uploaded file"}), 400
        if img.filename == '':
            return jsonify({"error": "Empty file name"}), 400

    # 그레이스케일로 변환 후 얼굴 감지
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    # 얼굴이 감지되지 않으면 에러 반환
    if len(faces) == 0:
        return jsonify({"error": "No face detected"}), 400

    # 얼굴이 감지되면 각 얼굴에 대해 랜드마크를 추출하고 얼굴형 분류
    for face in faces:
        landmarks = landmark_predictor(gray, face)
        face_shape = classify_face_shape(landmarks)

        # 얼굴형 분류 결과 반환
        if face_shape != "Unknown":
            # 결과 반환
            return render_template('index.html', result=f"Detected Face Shape: {face_shape}")
            #break  # 유효한 얼굴형을 찾으면 루프 종료

        # 오류 발생 가능성이 높아 원본으로도 추출 가능하면 시도하지 않음
        # 얼굴형이 Unknown인 경우 얼굴 정렬 후 재시도
        aligned_img = align_face(image_file)
        if aligned_img is None:
            return jsonify({"error": "Aligned face shape could not be determined"}), 500

        # 정렬된 이미지에서 얼굴 감지 및 얼굴형 분류
        gray_aligned = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
        faces_aligned = face_detector(gray_aligned)
        if len(faces_aligned) == 0:
            return jsonify({"error": "No face detected in aligned image"}), 500

        for face_aligned in faces_aligned:
            landmarks_aligned = landmark_predictor(gray_aligned, face_aligned)
            face_shape = classify_face_shape(landmarks_aligned)
            # 결과 반환
            return render_template('index.html', result=f"Detected Face Shape: {face_shape}")
            break
            # 얼굴형 결과를 Spring 서버에 저장
    try:
        response = requests.post(
            'http://localhost:8080/faceshape/save/{userId}'.format(userId=user_id),
            json={"faceShape": face_shape}
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to save face shape in backend", "details": str(e)}), 500
            
    # 성공적으로 처리된 결과 반환
    return jsonify({"face_shape": face_shape})

# Flask 앱 실행
if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=5000)
    except SystemExit:
        print("Flask app exited with SystemExit")

"""
