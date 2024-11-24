# 필요한 라이브러리와 모듈 임포트
from flask import Flask, redirect, render_template, request, jsonify
import cv2
import numpy as np
import dlib
from PIL import Image as PILImage
import requests  # Flask와 Spring 서버 간 통신을 위한 requests 모듈 추가
from face_shape_classify.preprocess_image import preprocess_image
from face_shape_classify.classify_face_shape import classify_face_shape
from face_shape_classify.align_face import align_face

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

@app.route('/', methods=["GET", "POST"])
# 얼굴형 판독 및 저장 API
@app.route('/upload', methods=['POST'])
@app.route('/faceshape/save', methods=['POST'])
def detect_face_shape():
    try:
        # 이미지 파일 확인
        files = request.files.getlist('image')
        if not files or len(files) == 0:
            return jsonify({"error": "No image file found in request"}), 400

        file = files[0]
        if not file or file.filename == '':
            return jsonify({"error": "No image selected"}), 400

        img_bytes = file.read()  # 이미지 데이터를 읽음
        if not img_bytes:
            return jsonify({"error": "Empty file content"}), 400

        # 이미지 전처리
        img = preprocess_image(img_bytes)
        if img is None:
            return jsonify({"error": "Failed to preprocess image"}), 400

        # 얼굴 감지
        faces = face_detector(img)
        if len(faces) == 0:
            return jsonify({"error": "No face detected"}), 400

        # 얼굴형 분류
        face_shapes = []
        for face in faces:
            landmarks = landmark_predictor(img, face)
            face_shape = classify_face_shape(landmarks)
            app.logger.info(f"Face shape detected: {face_shape}")

            if face_shape == "Unknown":
                 app.logger.info("Face shape is unknown, aligning face...")
                 aligned_img = align_face(img, face)
                 if aligned_img is not None:
                # 재정렬된 얼굴로 다시 분류 시도
                     landmarks = landmark_predictor(aligned_img, face)
                     face_shape = classify_face_shape(landmarks)
                     app.logger.info(f"Face shape after alignment: {face_shape}")
                 else:
                     app.logger.error("Face alignment failed")
            face_shapes.append(face_shape)

        # 결과 반환
        if face_shapes and "Unknown" not in face_shapes:
            return jsonify({"face_shape": face_shapes}), 200
        else:
            return jsonify({"face_shape": "Unknown"}), 400

    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)