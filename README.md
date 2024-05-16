# Atlantis engine

![1715102710368](example/docs/image/Atlantis.png)

## Thành quả phát triển

* [X]  Thiết kế API
* [X]  Công khai toàn bộ mã nguồn
* [X]  Tạo các code mẫu để áp dụng
* [ ]  đang cập nhật tiếp...

https://github.com/iotran207/Atlantis/assets/112244548/6781001b-ceaa-44ec-bb2f-b848aefccb5a

## Thử nghiệm

Thầy cô có thể truy cập đường link https://pealadau.org/atlantis để dùng thử dự án lần này.

Ngoài ra dự án sẽ sớm công khai link API để áp dụng vào dự án thực tế.

## Lời mở đầu

Giao tiếp có vai trò vô cùng quan trọng đối với mỗi chúng ta, nó vừa là điều kiện tồn tại của xã hội, vừa là cách giúp con người gia nhập vào các mối quan hệ, lĩnh hội nền văn hóa, đạo đức chuẩn mực xã hội. Tuy nhiên đối với người khiếm thính, họ chỉ có thể sử dụng ngôn ngữ kí hiệu để giao tiếp và điều khó khăn nhất là không phải nhiều người biết hoặc hiểu loại ngôn ngữ này! Mô hình máy học (Machine Learning) là công nghệ khá mới, chưa được ứng dụng rộng rãi ở đất nước chúng ta. Do đó chúng em đã sử dụng mô hình này để phát triển nên **“Dự án mã nguồn mở Atlantis engine” **với mục đích có thể giúp đỡ những người thiếu may mắn ấy dễ dàng hơn giao tiếp cũng như giúp họ kết nối với cộng đồng.

## Giới thiệu sản phẩm

**“Dự án mã nguồn mở Atlantis engine”** là một dự án mã nguồn mở được phát triển trên công nghệ thị giác máy tính và mô hình máy học. Mã nguồn mở hoàn toàn miễn phí dành cho tất cả mọi người với sứ mệnh không ai bị bỏ lại phía sau. Ngoài ra dự án còn cung cấp một số giải pháp hỗ trợ cho lập trình viên và người dùng có thể khai thác hết sức mạnh của dự án như là tài liệu API chính thức, hệ thống thử nghiệm mô hình, …

### Công nghệ sử dụng

* **Thị giác máy tính**

Thị giác máy tính là gì? Thị giác máy tính là một công nghệ mà máy sử dụng để tự động nhận biết và mô tả hình ảnh một cách chính xác và hiệu quả. Ngày nay, các hệ thống máy tính có quyền truy cập vào khối lượng lớn hình ảnh và dữ liệu video bắt nguồn từ hoặc được tạo bằng điện thoại thông minh, camera giao thông, hệ thống bảo mật và các thiết bị khác. Ứng dụng thị giác máy tính sử dụng trí tuệ nhân tạo và máy học (AI/ML) để xử lý những dữ liệu này một cách chính xác nhằm xác định đối tượng và nhận diện khuôn mặt, cũng như phân loại, đề xuất, giám sát và phát hiện.

* **Máy học**

Máy học (machine learning) là một lĩnh vực của trí tuệ nhân tạo (AI) mà trong đó máy tính được lập trình để tự động học và cải thiện từ dữ liệu mà nó nhận được. Thay vì chỉ dựa trên các quy tắc cụ thể được lập trình trước, máy học cho phép máy tính "học" thông qua việc phân tích dữ liệu và tìm ra các mẫu, xu hướng hoặc quy luật ẩn trong dữ liệu mà không cần được lập trình trực tiếp.

* **Giao diện chương trình ứng dụng**

Giao diện chương trình là gì? Giao diện chương trình – Application Programming Interface viết tắt là API là một trung gian phần mềm cho phép hai ứng dụng giao tiếp với nhau, có thể sử dụng cho web-based system, operating system, database system, computer hardware, hoặc software library.

## Hướng dẫn cài đặt và chạy

### Hướng dẫn train models cho cá nhân

#### Dưới đây là hướng dẫn train model cơ bản

```python
# collect_imgs.py
import os
import cv2
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
number_of_classes = 5
dataset_size = 100
cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))
    print('Collecting data for class {}'.format(j))
    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1
cap.release()
cv2.destroyAllWindows()
```

```python
#create_dataset.py
import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
```

```python
#train_classifier.py
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from os import remove
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

remove('./data.pickle')
```

```python
#inference_classifier.py
import pickle

import cv2
import mediapipe as mp
import numpy as np
import time

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

OUTPUT = []

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0:'ok',1:'xin chao',2:'tam biet'}
CHECK_FRAME = 0
predicted_character = ''

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
  
    if results.multi_hand_landmarks:

        CHECK_FRAME+=1

        if CHECK_FRAME == 1:
            CHECK_FRAME = 0

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            print(prediction)

            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character  , (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    if predicted_character != '':
        if len(OUTPUT) == 0 or OUTPUT[-1] != predicted_character:
            OUTPUT.append(predicted_character)


    cv2.imshow('frame', frame)
    if cv2.waitKey(25)==ord('q'):
        break

print(OUTPUT)

cap.release()
cv2.destroyAllWindows()
```

## Giấy phép

```
MIT License

Copyright (c) 2024 iotran207

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION 
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Phần cuối xin cảm ơn team MEDIAPIPE của google vì đã phát triển một framework thật tuyệt vời và [computervisioneng](https://github.com/computervisioneng) đã tạo nên một repo thật tuyệt vời để học hỏi.
