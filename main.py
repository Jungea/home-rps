from flask import Flask, render_template, Response, url_for, redirect, request
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image


app = Flask(__name__)
cors = CORS(app)
model = load_model('1ResNet50_model.h5')

label = 0


def gen_frames():
    camera = cv2.VideoCapture(0)  # 첫번째 영상 장치 불러오기

    if not camera.isOpened():
        raise RuntimeError("Could not open camera")

    while True:
        success, frame = camera.read()  # read the camera frame

        if not success:
            break

        # 카메라로 들어온 이미지 가공
        img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # 예측
        prediction = model.predict(x)
        predicted_class = np.argmax(prediction[0])  # 예측된 클래스 0, 1, 2
        print(prediction[0], predicted_class)

        global label
        label = predicted_class

        if predicted_class == 0:
            me = "바위"
        elif predicted_class == 1:
            me = "보"
        elif predicted_class == 2:
            me = "가위"

        # 학습 결과 camera 표시
        font1 = ImageFont.truetype("font/gulim.ttc", 100)
        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil)
        draw.text((50, 50), me, font=font1, fill=(0, 0, 255, 3))
        frame = np.array(frame_pil)
        cv2.imshow('RPS', frame)

        # HTML 파일의 Image 태그에 camera 내용 표시
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    camera.release()


@app.route('/page/')
def index_page():
    return render_template('index.html')


@app.route('/page/rps')
def rps_page():
    return render_template('rps.html')


# 웹캠으로 웹페이지에서 보여주기(실시간)
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# 사용자의 예측값을 가져오기 위해서(컴퓨터용)
@app.route('/get_label')
def getLabel():
    return {"result": int(label)}


if __name__ == "__main__":
    # app.debug = True
    app.run(host='0.0.0.0')
