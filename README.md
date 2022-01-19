
# 가위바위보 분류
[model(colab)](https://c11.kr/RPS_model)

[발표PPT]
[가위바위보분류_정은애_210725.pptx](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/aca82c47-e942-432a-a530-fa0acdcf2fdb/%EA%B0%80%EC%9C%84%EB%B0%94%EC%9C%84%EB%B3%B4%EB%B6%84%EB%A5%98_%EC%A0%95%EC%9D%80%EC%95%A0_210725.pptx)


## 1. 프로젝트 개요

-   복습겸 mnist와 같은 cnn사용 직접 데이터 셋을 만들어 볼만한 프로젝트 선정
-   CNN을 활용한 가위바위보 분류 모델
-   모델을 사용해 웹캠에 찍힌 손모양을 인식하는 웹페이지

## 2. 개발 환경 및 라이브러리

- [model]
	- ubuntu(colab)
	- python
	- pandas, numpy, tensorflow, keras, PIL, matplot

- [webpage]
	- windows
	- python, HTML, CSS, JS, Jinja2
	- flask, opencv, bootstrap, jquery, ajax,

## 3. 구조

-   Dataset 구조
    
    이미지
    
    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7d4585ba-8b7e-4108-8ce3-eeae95676111/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7d4585ba-8b7e-4108-8ce3-eeae95676111/Untitled.png)
    
    각 손모양당 210장
    
    (630, 224, 224, 3)
    
    정답
    
    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/09153044-5adb-49ba-816e-56f1d3d58ccf/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/09153044-5adb-49ba-816e-56f1d3d58ccf/Untitled.png)
    
    (630, 3)
    
    Trainning set(570) + Test set(60)
    
-   Model 구조
    
    이미지 인식 분야에서 좋은 성능을 보이는 CNN을 사용하였다.
    
    <aside> 💡 cnn 기본 설명 있으면 좋을듯
    
    </aside>
    
    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/11c9197a-80fb-4823-b615-686d85a92f5d/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/11c9197a-80fb-4823-b615-686d85a92f5d/Untitled.png)
    
    ```python
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu", input_shape=(224,224,3)))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Conv2D(128, kernel_size=(3,3), activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(64, activation="relu")) 
    model.add(keras.layers.Dense(3, activation="softmax"))
    
    ```
    
    <aside> 💡 dropout 써서 다시하면 괜찮지 않을까?
    
    </aside>
    
    컨볼루션 레이어에서 특징을 추출하는데
    
    input_shape=(224,224,3) ⇒ 이미지 dataset shape
    
    맥스풀링 레이어에서 레이어의 사이즈를 줄여주고
    
    Flatten을 이용해 1차원으로 바꿔준 다음
    
    Dense 레이어를 연결
    
    마지막 레이어 activation은 softmax(구분 결과를 확률로), unit 은 결과값 개수
    
    ```python
    model.summary()
    
    ```
    
    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/12d60911-47b8-4037-af56-9cd4a0eda94e/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/12d60911-47b8-4037-af56-9cd4a0eda94e/Untitled.png)
    
    ```python
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])
    model_history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)
    
    ```
    
    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/afb8fcc8-417c-4466-9b2a-1e7c3a7ceaa3/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/afb8fcc8-417c-4466-9b2a-1e7c3a7ceaa3/Untitled.png)
    
    마지막이 softmax일 경우 categorical_crossentropy 사용
    
    Model 평가
    
    loss and accuracy
    
    ```python
    model_loss = model_history.history["loss"]
    model_val_loss = model_history.history["val_loss"]
    
    model_acc = model_history.history["accuracy"]
    model_val_acc = model_history.history["val_accuracy"]
    
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_title('Loss according to Epoch')
    ax1.plot(range(1,11), model_loss, label="train")
    ax1.plot(range(1,11), model_val_loss, label="validation")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # fig = plt.figure(figsize=(10,5))
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title('Accuracy according to Epoch')
    ax2.plot(range(1,11), model_acc, label="train")
    ax2.plot(range(1,11), model_val_acc, label="validation")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    ```
    
    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/810669ec-e79f-46b7-93bd-631c96f8f9d5/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/810669ec-e79f-46b7-93bd-631c96f8f9d5/Untitled.png)
    
    Epoch 증가에 따라 Loss는 감소, 모델의 오차가 줄어든다.
    
    Epoch 증가에 따라 Accuarcy는 증가, 모델의 성능이 좋아진다.
    
    평가
    
    ```python
    model.evaluate(x_test, y_test)
    
    ```
    
    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a9288722-00c2-487a-8fcb-688ed5e64cab/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a9288722-00c2-487a-8fcb-688ed5e64cab/Untitled.png)
    
    모델 학습이 잘된거라 생각
    
-   프로젝트 구조
    

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9a3010ed-0be8-4814-ba7a-9e21cdcffffc/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9a3010ed-0be8-4814-ba7a-9e21cdcffffc/Untitled.png)

## 4. 프로젝트 기능

[요청 URL (1)](https://www.notion.so/e7fb24fda3d4453caf6013350828f1dd)

-   웹캠을 통한 얻은 데이터를 모델로 예측해
-   웹페이지에서 예측한 데이터를 가져와 표시해주고
-   예측 데이터에 반대되는 값을 표시해줍니다.

<aside> 💡 자세한 설명은 구현 코드 쪽에서 설명

</aside>

## 5. 구현 코드

-   Dataset 생성

```python
# 틀 생성
dataset_img = np.float32(np.zeros((630, 224, 224, 3))) # (sample_size, size1, size2, channels)
dataset_label = np.float64(np.zeros((630, 1)))

# 이미지 가공
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, 0)
x = preprocess_input(x)

dataset_img[num, :, :, :] = x
dataset_label[num] = i

```

```python
# [0] => [1, 0, 0]

y_test = keras.utils.to_categorical(y_test)
y_train = keras.utils.to_categorical(y_train)

```

-   Flask로 서버 만들기
    
    ```python
    from flask import Flask, render_template
    
    app = Flask(__name__)
    
    @app.route('/page/')
    def index_page():
        return render_template('index.html')
    
    @app.route('/get_label')
    def getLabel():
        return {"result": int(label)}
    
    if __name__ == "__main__":
        app.debug = True
        app.run(host='0.0.0.0')
    
    ```
    
-   웹캠 이미지를 불러와 예측
    
    rps_page.html
    
    ```python
    <img src="{{ url_for('video_feed') }}">
    
    ```
    
    [main.py](http://main.py)
    
    ```python
    import cv2
    
    def gen_frames():
        camera = cv2.VideoCapture(0)
    
        while True:
            success, frame = camera.read()
    				
    				... 웹캠으로 들어온 이미지 가공 코드...
            ... 예측 코드 ...
    
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
    
            yield (b'--frame\\r\\n'
                   b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame + b'\\r\\n')  # concat frame one by one and show result
    
        camera.release()
    
    @app.route('/video_feed')
    def video_feed():
        return Response(gen_frames(), 
    					mimetype='multipart/x-mixed-replace; boundary=frame')
    
    ```
    
-   AJAX를 이용해 예측값 가져오기
    
    <aside> 💡 .html
    
    </aside>
    
    ```python
    <h3 id="result"></h3>
    <img id="computer_hand" class="img-fluid" src="">
    
    ```
    
    rps_script.js
    
    ```python
    function getlabel() {
    
        $.ajax({
            url: "/get_label",
              method: "GET",
              dataType: "json",
    
          }).done(function(r) {
    
              var result = r.result;
              $('#result').text(result_string[result]);
              $('#computer_hand')
    							.attr('src', '/static/image/'+computer_hand[result]+'.png')
    
          console.log("ajax-getLabel-success", result);
    
        }).fail(function() {
              console.log("ajax-getLabel-fail");
        });
    
    ```
    
    main.js
    
    ```python
    @app.route('/get_label')
    def getLabel():
        return {"result": int(label)}
    
    ```
    

## 6. 동작

[](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d20fd380-54c8-4345-8e59-b20284103bc4/%EB%85%B9%ED%99%94_2021_07_23_13_38_15_772.mp4)[https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d20fd380-54c8-4345-8e59-b20284103bc4/녹화_2021_07_23_13_38_15_772.mp4](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d20fd380-54c8-4345-8e59-b20284103bc4/%EB%85%B9%ED%99%94_2021_07_23_13_38_15_772.mp4)
