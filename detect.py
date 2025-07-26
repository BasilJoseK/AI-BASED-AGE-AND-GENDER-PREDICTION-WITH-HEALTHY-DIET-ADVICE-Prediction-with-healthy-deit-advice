import cv2
import numpy as np

# Initialize the models and other constants
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Initialize webcam
video = cv2.VideoCapture(0)

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

def capture_age_and_gender():
    padding = 20
    hasFrame, frame = video.read()
    if not hasFrame:
        return None, None

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        return None, None

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                     :min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        return age[1:-1], gender
    return None, None

def release_video():
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Create OpenCV window and set it to fullscreen
        window_name = "Age and Gender Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Run the video capture and detection loop
        while cv2.waitKey(1) < 0:
            hasFrame, frame = video.read()
            if not hasFrame:
                break

            resultImg, faceBoxes = highlightFace(faceNet, frame)
            if not faceBoxes:
                print("No face detected")
                continue

            # Process each face detected in the frame
            for faceBox in faceBoxes:
                face = frame[max(0, faceBox[1] - 20):
                             min(faceBox[3] + 20, frame.shape[0] - 1), max(0, faceBox[0] - 20)
                             :min(faceBox[2] + 20, frame.shape[1] - 1)]

                # Get age and gender predictions
                age, gender = capture_age_and_gender()
                
                if age and gender:
                    print(f"Gender: {gender}, Age: {age} years")
                    cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # Display the resulting image with predictions
            cv2.imshow(window_name, resultImg)
    except KeyboardInterrupt:
        print("Process interrupted")
    finally:
        release_video()  # Release the webcam and close the OpenCV windows
