import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0) # 노트북 웹캠 불러오기
cap.set(3, 1200)  # 화면 크기 설정
cap.set(4, 1000)  # 화면 크기 설정

detector = dlib.get_frontal_face_detector() # 정면 얼굴 점출
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# dlib을 이용해 얼굴에서 점 68개 찾기

xml = 'C:\\Users\YEJINGGU\PycharmProjects\pythonProject\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(xml) #하르 특징 분류기 사용하기 위해 경로 저장
low,high=10,10

def lowc(value): # 캐니 엣지 low값
    global low
    low=value

def highc(value): # 캐니 엣지 high 값
    global high
    high=value

while True:
    ret, frame = cap.read() # 프레임 읽어옴
    frame = cv2.flip(frame, 1)  # 좌우 대칭
    cv2.createTrackbar("low", "face Canny Edge", low, 100, lowc)
    cv2.createTrackbar("high", "face Canny Edge", high, 300, highc)
    for i in range(4):
        text = ["Press 'c' = Canny edge filtern (Please keep pressing)",
       "Press 'm' (And close your mouth) = Mosaic filter",
       "Press 'h' (And open your mouth) = Heart filter",
       "Press 'b' = bird filter"]

        org = [(10, 30),(10, 50),(10, 70),(10, 90)]
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, text[i], org[i], font, 0.7, (255, 255, 255), 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 흑백 영상으로 바꿈

    keycode = cv2.waitKey(1)

    if keycode == ord('c'): # 'c' 를 누르면 canny filter 실행
        ret, frame_canny = cap.read()  # 프레임 읽어옴
        frame_canny = cv2.flip(frame_canny, 1)  # 좌우 대칭
        frame_canny = cv2.GaussianBlur(frame_canny, (5, 5), 1.23)
        # frame 읽고, 5x5 가우시안 블러 적용해서 노이즈 제거
        frame_canny=cv2.Canny(frame_canny,low,high)
        # cv2.Canny를 이용해 캐니 엣지 적용
        cv2.imshow('face Canny Edge', frame_canny)

        if cv2.waitKey(1) == ord('q'):  # 'q'를 1초간 누르면 나가짐
            break

    elif keycode == ord('m'): # 'm' 을 누르면 mosaic filter 실행
        while True:
            ret, frame_mosaic = cap.read() # 프레임 읽어옴
            frame_mosaic = cv2.flip(frame_mosaic, 1)  # 좌우 대칭
            frame_mosaic_gray = cv2.cvtColor(frame_mosaic, cv2.COLOR_BGR2GRAY) # 흑백 영상으로 바꿈

            faces = face_cascade.detectMultiScale(frame_mosaic_gray, 1.05, 5)
            # detectMultiScale 함수에 흑백 영상 넣어주고 얼굴 검출
            faces2 = detector(frame_mosaic_gray)  # 얼굴 검출
            for face in faces2:
                land = predictor(frame_mosaic, face)  # 랜드마크 검출
                close_mouse = land.part(67).y - land.part(63).y
                if (close_mouse < 5):
                    for (x, y, w, h) in faces:
                        face_img = frame_mosaic[y:y + h, x:x + w]  # 인식된 얼굴 이미지 영역
                        face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.05, fy=0.05)  # 축소
                        face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA)
                        # 축소한 이미지를 원래 크기로 만들어줌, interpolation 보간
                        frame_mosaic[y:y + h, x:x + w] = face_img  # 인식된 얼굴 영역 모자이크
            cv2.imshow('face mosaic', frame_mosaic)

            if cv2.waitKey(1) == ord('q'):  # 'q'를 1초간 누르면 나가짐
                break

    elif keycode == ord('b'): # 'z' 를 누르면 bird filter 실행
        while True:
            bird_mask = cv2.imread('../images/bird.jpg') # 마스크 만들 사진 불러옴
            ret, frame_bird = cap.read()  # frame 읽어옴
            frame_bird = cv2.flip(frame_bird, 1)  # 좌우 대칭
            frame_bird_gray = cv2.cvtColor(frame_bird, cv2.COLOR_BGR2GRAY) # 흑백 영상으로 바꿈

            face_rects = face_cascade.detectMultiScale(frame_bird_gray, 1.3, 5)
            # detectMultiScale 함수에 흑백 영상 넣어주고 얼굴 검출

            try: # 예외처리
                 for (x, y, w, h) in (face_rects):
                    y = y - 140
                    frame_mask = frame_bird[y:y + h, x:x + w]
                    # 얼굴 영역 추출
                    bird_mask = cv2.resize(bird_mask, (w, h))
                    # 필터 마스크 영상 크기 조절
                    bird_gray = cv2.cvtColor(bird_mask, cv2.COLOR_BGR2GRAY)
                    # 필터 마스크 흑백 영상으로 바꿔줌
                    ret, bm = cv2.threshold(bird_gray, 240, 255, cv2.THRESH_BINARY_INV)
                    # 이미지 흑백으로 변환해서 쓰레시 홀딩 이미지 만듬
                    bm_in = cv2.bitwise_not(bm)
                    # 비트 와이즈로 역 이미지 만듬
                    mask_bird = cv2.bitwise_and(bird_mask, bird_mask, mask=bm)
                    mask_frame = cv2.bitwise_and(frame_mask, frame_mask, mask=bm_in)
                    # 비트와이즈로 새 마스크와 얼굴 마스크 만듬
                    frame_bird[y:y + h, x:x + w] = cv2.add(mask_bird, mask_frame)
                    # 새와 얼굴을 합침

                    cv2.imshow('face mask', frame_bird)
            except cv2.error:
                print("카메라와 거리를 조절해주세요..")

            if cv2.waitKey(1) == ord('q'):  # 'q'를 1초간 누르면 나가짐
                break

    elif keycode == ord('h'): # 'h' 를 누르면 heart filter 실행
        while (True):
            heart = cv2.imread('../images/heart.png')  # 하트 이미지 읽음
            ret, frame_heart = cap.read()  # 프레임 읽음
            frame_heart = cv2.flip(frame_heart, 1) # 좌우 대칭
            gray = cv2.cvtColor(frame_heart, cv2.COLOR_BGR2GRAY)  # frame_heart를 받아 흑백 영상으로 바꿈

            heart_gray = cv2.cvtColor(heart, cv2.COLOR_BGR2GRAY) # heart 흑백 영상으로 만듬
            ret, hm = cv2.threshold(heart_gray, 255, 1, cv2.THRESH_BINARY_INV)
            # 이미지 흑백으로 변환해서 쓰레시 홀드 함수로 임계값 기준으로 이진화함

            hm_in = cv2.bitwise_not(hm)  # 비트 와이즈로 흑,백 바꾸어줌
            heart_mask = cv2.bitwise_and(heart, heart, mask=hm)
            # 비트와이즈 and 연산으로 하트 합

            faces = detector(gray)  # frame_heart에서 얼굴 검출

            for face in faces:
                land = predictor(frame_heart, face)  # 랜드마크 검출

                open_mouth = land.part(67).y - land.part(63).y

                if(open_mouth > 5):
                    heart_mask_resize = cv2.resize(heart_mask, (80, 80))
                    # 마스크 영상 크기 조절
                    hm_in_resize = cv2.resize(hm_in, (80, 80))
                    # 이미지 영상 크기 조절
                    x = land.part(18).x - 10  # 왼쪽 눈 x 좌표 (눈썹 아래 부터)
                    y = land.part(18).y - 10  # 왼쪽 눈 y 좌표
                    x1 = land.part(22).x - 10   # 오른쪽 눈 x 좌표
                    y1 = land.part(22).y - 10   # 오른쪽 눈 y 좌표
                    # 검출한 랜드마크를 이용해 눈 좌표 추출

                    r_eye = frame_heart[y:y + 80, x:x + 80]  # 왼쪽 눈 영역
                    l_eye = frame_heart[y1:y1 + 80, x1:x1 + 80]  # 오른쪽 눈 영역

                    l = cv2.bitwise_and(l_eye, l_eye, mask=hm_in_resize)
                    r = cv2.bitwise_and(r_eye, r_eye, mask=hm_in_resize)
                    # 비트와이즈로 하트와 합칠 마스크 영상

                    frame_heart[y:y + 80, x:x + 80] = cv2.add(heart_mask_resize, r)
                    frame_heart[y1:y1 + 80, x1:x1 + 80] = cv2.add(heart_mask_resize, l)
                    # 마스크와 하트를 합침

            cv2.imshow('eyes heart', frame_heart)  # 출력

            if cv2.waitKey(1) == ord('q'): # 'q'를 1초간 누르면 나가짐
                break

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()