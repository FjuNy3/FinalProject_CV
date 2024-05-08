import cv2

def generate_dataset(img, id, img_id):
    # write image in data dir
    cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpg", img)

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    num_faces = len(features)  # Count the number of detected faces
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        cv2.putText(img, "Total faces: " + str(num_faces), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords, num_faces

def detect(img, faceCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords, num_faces = draw_boundary(img, faceCascade, 1.1, 3, color['red'], "Face")

    if len(coords)==4:
        # Updating region of interest by cropping image
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        # Assign unique id to each user
        user_id = 1
        # img_id to make the name of each image unique
        generate_dataset(roi_img, user_id, img_id)
    # If feature is detected, the draw_boundary method will return the x,y coordinates and width and height of rectangle else the length of coords will be 0
    # if len(coords)==4:
    #     # Updating region of interest by cropping image
    #     roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
    #     # Passing roi, classifier, scaling factor, Minimum neighbours, color, label text
    #     coords = draw_boundary(roi_img, eyeCascade, 1.1, 12, color['red'], "Eye")
    #     coords = draw_boundary(roi_img, noseCascade, 1.1, 4, color['green'], "Nose")
    #     coords = draw_boundary(roi_img, mouthCascade, 1.1, 20, color['white'], "Mouth")
    return img

model = ['haarcascade_frontalface_alt.xml',          
         'haarcascade_frontalface_alt2.xml',       
         'haarcascade_frontalface_default.xml']
  
faceCascade = cv2.CascadeClassifier(model[0])

# Giả sử index camera 0 là chính xác
cap = cv2.VideoCapture(0)

img_id = 0

while True:
    if img_id % 50 == 0:
        print("Collected ", img_id," images")
    # Chụp ảnh từng khung
    ret, img = cap.read()

    # Kiểm tra xem có chụp được ảnh thành công hay không
    if ret:
        img = detect(img, faceCascade)
        cv2.imshow("Face detection", img)
        img_id += 1
        if cv2.waitKey(1) == ord('q'):
            break  # Thoát khi nhấn 'q'
    else:
        print("Error: No camera was found.")
        break

# Giải phóng bộ nhớ capture và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()