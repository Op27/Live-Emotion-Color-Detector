import cv2
from rmn import RMN
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
m = RMN()

emotion_colors = {
    "angry": (0, 0, 255),  # Red
    "disgust": (0, 128, 0),  # Dark Green
    "fear": (255, 0, 255),  # Magenta
    "happy": (147, 20, 255),  # Pink
    "sad": (255, 0, 0),  # Blue
    "surprise": (255, 255, 0),  # Yellow
    "neutral": (255, 255, 255),  # White
}

process_every_n_frames = 5
frame_count = 0
last_processed_time = time.time()
processing_interval = 0.25 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % process_every_n_frames == 0 or (time.time() - last_processed_time) > processing_interval:
        last_processed_time = time.time()
        results = m.detect_emotion_for_single_frame(frame)
        for result in results:
            x, y, x2, y2 = result['xmin'], result['ymin'], result['xmax'], result['ymax']
            emotion = result['emo_label'].lower()  
            print(emotion)  # Debugging line to check the emotion labels
            color = emotion_colors.get(emotion, (0, 255, 255))  # Default to yellow if emotion not found
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
