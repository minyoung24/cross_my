import cv2
from ultralytics import YOLO

# 학습된 모델 불러오기
model = YOLO("C:\\Users\\lmy79\\safeT\\w_h\\best.pt")

# 동영상 파일 열기
cap = cv2.VideoCapture("C:/Users/lmy79/safeT/video1.mp4")

# 동영상 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0  # 프레임 카운터 초기화

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 객체 탐지
    results = model(frame)

    # 탐지 결과 그리기
    for result in results:
        boxes = result.boxes.xyxy.numpy()
        scores = result.boxes.conf.numpy()
        classes = result.boxes.cls.numpy()

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box

            if score >= 0.5:  # 신뢰도가 0.5 이상일 때만 라벨 표시
                if cls == 0:  # 가로 횡단보도
                    label = f'Length {score:.2f}'
                    color = (0, 255, 0)  # Green
                elif cls == 1:  # 세로 횡단보도
                    label = f'Width {score:.2f}'
                    color = (0, 0, 255)  # Red
                else:
                    continue  # 기타 클래스는 무시

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 결과 저장
    out.write(frame)

    # 결과 화면에 표시
    cv2.imshow('Crosswalk Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 프레임 카운터 증가
    frame_count += 1
    if frame_count % 100 == 0:  # 매 100프레임마다 진행상황 출력
        print(f'Processed {frame_count} frames')

cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete. Results saved to output1.avi")
