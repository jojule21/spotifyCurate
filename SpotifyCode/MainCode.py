import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)


img_count = 0


while True:
    ret, img = cap.read()

    if not ret:
        print("Error")
        break

    if img_count % 10 == 0:
        userEmotion = DeepFace.analyze(img, actions = ['emotion'], enforce_detection = False)
        emotion = userEmotion[0]['dominant_emotion'] 
        
    cv2.putText(img, f'Emotion: {emotion}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
    
    img_count += 1
    
    cv2.imshow("Video", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    

cap.release()
cv2.destroyAllWindows()
