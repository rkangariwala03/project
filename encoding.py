import cv2
import face_recognition
import os
import pickle

path_to_images = "C:\\Users\\Ria B. Kangariwala\\Capstone_project\\Photos"  
images = []
Ids = []
myList = os.listdir(path_to_images)

try:
    with open('EncodeFile.p', 'rb') as file:
        existing_list = pickle.load(file)
        existing_encodings, existing_ids = existing_list
        print("Existing encodings loaded.")
except FileNotFoundError:
    existing_encodings = []
    existing_ids = []
    print("No existing encodings found, starting from scratch.")

for img_name in myList:
    img = cv2.imread(f'{path_to_images}/{img_name}')
    id = os.path.splitext(img_name)[0]  

    if id in existing_ids:
        print(f"Skipping {id}, already encoded.")
        continue 

    images.append(img)
    Ids.append(id)

print("New Image IDs:", Ids)

def findEncodings(images):
    encodeList = []
    for img in images:
        # Convert the image to RGB (required for face_recognition)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            print("No face found in image, skipping encoding.")
    return encodeList

if images:
    print("Generating encodings for new images ...")
    encodeListNew = findEncodings(images)
    
    # Add the new encodings to the existing ones
    existing_encodings.extend(encodeListNew)
    existing_ids.extend(Ids)
    
    # Save the updated encodings and IDs back to the file
    print("Saving updated encodings to a file ...")
    with open('EncodeFile.p', 'wb') as file:
        existing_list = [existing_encodings, existing_ids]
        pickle.dump(existing_list, file)
    
    print("Encodings saved successfully!")
else:
    print("No new images to encode.")
