# importing the packages
import cv2                  # "cv2", computer vision thus for accessing the webcam
import numpy as np          # for creating the arrays
import os                   # for reading and writing the files in the operating system thus "os"
import pickle               # for saving the dataset



# how to import the dataset for out project
video = cv2.VideoCapture(0) # the value "0" means the web camera 
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # uses the cascadeclassifier to detect the faces in the camera
                                                                          # the xml file was downloaded from github to help classify just the "frontal face" so i used the search "haarcascade_frontalface_default.xml download"

face_data = [] # this array is created to store the face data
i = 0 # creating a varaible "i" and assigning its value ot "0"
name = input("Enter your name: ") # asking the user for his name


# how to use the web camera to detect our faces
while True:
    ret, frame = video.read() #reading data form the webcam and storing the values in the "ret" and "frame" variables
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converting the colorful images to black and white
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) # for reading the webcam video

    # using the for loop to find the coordinates in the faces
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :] # cropping our image
        resized_img = cv2.resize(crop_img, (50,50)) # resize the crop img to a size of 50:50

        if len(face_data)<= 100 and i % 10 == 0:
            face_data.append(resized_img)
        i = i+1

        cv2.putText(frame, str(len(face_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(50, 50, 255), thickness=1) # writing out text in the frame, the len(face_data) tells how many faces are detected in the webcam
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,255), 1) # creating a rectangle to detect our face

    # showing the output
    cv2.imshow("frame", frame) # the "frame" will be the name of the webcam frame
    k = cv2.waitKey(1) # the wait key will be "1"

    if len(face_data) == 50: # if the faces are directed 50 times the output will be break
        break 

video.release()  # release the video after the completion of our output
cv2.destroyAllWindows() # remove all the output 


# how to save the detected faces with pickle and convert that faces into the dataset
face_data = np.array(face_data) # converting it in array format
face_data = face_data.reshape(50,-1) #reshaping all the face detected in the array

if "names.pkl" not in os.listdir("data/"): # writing in the pickle file
    names = [name]*100 # saving the names
    with open("data/names.pkl", "wb") as f: # writing in the names.pkl file and renaming it to "f"
        pickle.dump(names, f) # putting the names in the file

else: # if the pickle file exists then updating it or create a new one
    with open("data/names.pkl", "rb") as f:
        names = pickle.load(f) # loading the file for the data
    names = names + [name]*100


    with open("data/names.pkl", "wb") as f:
        pickle.dumb(names, f)

        
if "face_data.pkl" not in os.listdir("data/"): # if the face data.pkl file doesn't exist then we create another one
    with open("data/face_data.pkl", "wb") as f:
        pickle.dump(face_data, f)
else:
    with open("data/face_data.pkl", "rb") as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_data, axis=0)
    with open("data/face_data.pkl", "wb") as f:
        pickle.dump(faces, f)

print("✅ Face data saved successfully!")










# import cv2
# import numpy as np
# import os
# import pickle

# # Create "data/" directory if it doesn't exist
# if not os.path.exists("data"):
#     os.makedirs("data")

# # Initialize camera and face detector
# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# face_data = []  # List to store face data
# name = input("Enter your name: ")

# # Capture face data
# while True:
#     ret, frame = video.read()
#     if not ret:
#         print("Error accessing webcam")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#     faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w]
#         resized_img = cv2.resize(crop_img, (50, 50))  # Resize to 50x50
#         face_data.append(resized_img)  # Store the face

#         # Display the face count
#         cv2.putText(frame, f"Faces Collected: {len(face_data)}/100", (10, 50), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

#     cv2.imshow("Face Capture", frame)

#     if len(face_data) >= 100:  # Stop when 100 faces are captured
#         break

#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
#         break

# video.release()
# cv2.destroyAllWindows()

# # Convert to numpy array and reshape
# face_data = np.array(face_data)
# face_data = face_data.reshape(100, -1)  # Flatten images

# # Save name data using pickle
# names_file = "data/names.pkl"
# faces_file = "data/face_data.pkl"

# if not os.path.exists(names_file):
#     names = [name] * 100
#     with open(names_file, "wb") as f:
#         pickle.dump(names, f)
# else:
#     with open(names_file, "rb") as f:
#         names = pickle.load(f)
#     names.extend([name] * 100)
#     with open(names_file, "wb") as f:
#         pickle.dump(names, f)

# # Save face data using pickle
# if not os.path.exists(faces_file):
#     with open(faces_file, "wb") as f:
#         pickle.dump(face_data, f)
# else:
#     with open(faces_file, "rb") as f:
#         faces = pickle.load(f)
#     faces = np.append(faces, face_data, axis=0)
#     with open(faces_file, "wb") as f:
#         pickle.dump(faces, f)

# print("✅ Face data saved successfully!")
