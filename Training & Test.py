from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 데이터 경로
DATA_PATH = os.path.join('MP_Data') 

# Actions
actions = np.array([])

# 캡처 Sequence
no_sequences = 30

# 각 Sequence당 길이
sequence_length = 30

# 라벨 맵
label_map = {label:num for num, label in enumerate(actions)}

# 캡처된 데이터파일, 경로 속 .npy 파일을 모두 불러움
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# 데이터의 X, Y 값
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# 트레이닝을 위한 Tensorflow import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# 학습 모델 설정
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

res = [.7, 0.2, 0.1]

actions[np.argmax(res)]

# 학습 옵티마이저, 로스, 매트릭스 설정
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 학습의 Epochs 설정
model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])

model.summary()

res = model.predict(X_test)

# 학습된 결과를 action.h5로 저장
model.save('action.h5')

# 파파고 오픈 번역 툴을 불러오기 위한 import
import os
import sys
import urllib.request
import json
from pprint import pprint

# 오픈 API 클라이언트 접속 ID, PW
client_id = "dClDzLhTicynMYdfirZF"
client_secret = "ZAB8esUKDy"

# 한국어 > 영어 번역 함수
def ToEn (koText) :
    encText = urllib.parse.quote(koText)
    data = "source=ko&target=en&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        result = response_body.decode('utf-8')
        d = json.loads(result)
        print('--- Korean to English ---')

        print('번역 후 : ' , d['message']['result']['translatedText'])
    else:
        print("Error Code:" + rescode)


# 영어 > 한국어 번역 함수        
def ToKo (egText) :
    kocText = urllib.parse.quote(egText)
    data = "source=en&target=ko&text=" + kocText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        result = response_body.decode('utf-8')
        d = json.loads(result)
        print('--- English to Korean ---')
        print('번역 후 : ' , d['message']['result']['translatedText'])
    else:
        print("Error Code:" + rescode)

        
# --------------------------------------------------------
        
# 모델 불러오기
model.load_weights('action.h5')

# 모델 인식 Tensorflow
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# 불러올 모델의 x, y값
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

# 새로운 detection 값
sequence = []
sentence = []
threshold = 0.7

# 웹캠 실행
cap = cv2.VideoCapture(0)

# Mediapipe 모델 적용 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # feed 읽기
        ret, frame = cap.read()

        # detections 만들기
        image, results = mediapipe_detection(frame, holistic)
        
        # Landmarks 만들기
        draw_styled_landmarks(image, results)
        
        # 예측 Logic
        keypoints = extract_keypoints(results)
        sequence.insert(0,keypoints)
        sequence = sequence[:10]
         
        if len(sequence) == 10:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            
            
        # Viz Logic
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 화면 
        cv2.imshow('OpenCV Feed', image)

        # 중지 코드
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    
# 번역퇸 Text 를 출력 (웹캠 정지 필수)
str_sentence = str(sentence)

ToKo(str_sentence)
