import cv2
import os
import face_recognition
import pickle
import numpy as np

print("Loading Encode File ...")
with open('EncodeFile.p', 'rb') as file:
    existing_list = pickle.load(file)

given_id="Rinkal"

existing_encoding, existing_id = existing_list
print("Encode File Loaded")

if given_id in existing_id:
    index=existing_id.index(given_id)
    encoding=[existing_encoding[index]]

path_to_images = "C:\\Users\\Ria B. Kangariwala\\Capstone_project\\Photos_1"
myList = os.listdir(path_to_images)
images=[]
for img_name in myList:
    img = cv2.imread(f'{path_to_images}/{img_name}')
    imgS = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceLoc = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceLoc)

    RECOGNITION_THRESHOLD = 0.50  # Adjust this value as needed

    # Check if any face is detected
    if faceLoc:
        for encodeFace, facePosition in zip(encodeCurFrame, faceLoc):
            # Compare the detected face with known faces
            matches = face_recognition.compare_faces(encoding, encodeFace)
            faceDis = face_recognition.face_distance(encoding, encodeFace)
            
            # Find the best match
            matchIndex = np.argmin(faceDis)
            print(faceDis[matchIndex])
            
            if matches[matchIndex] and faceDis[matchIndex] < RECOGNITION_THRESHOLD:
                # If a match is found and below the threshold, retrieve the ID and display
                matchedId = given_id
                images.append(img_name)
                print(f"Face recognized: {matchedId}")
                '''
                # Draw bounding box and display ID on the image
                y1, x2, y2, x1 = facePosition
                y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, matchedId, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # Display the image in full screen
                cv2.namedWindow("Face Recognition", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("Face Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                # Show the image with the recognized face(s)
                cv2.imshow("Face Recognition", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                        '''
            else:
                continue
              
    else:
        continue


print(images)

    