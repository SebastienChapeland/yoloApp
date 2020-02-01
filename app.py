import streamlit as st
import cv2
import pandas as pd
import numpy as np
# import requests
# from PIL import Image

# Add a title and sidebar
st.title("Object Detection")
st.sidebar.markdown("# Model")
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)

# Get the necessary files
# @st.cache(show_spinner=False)
# def get_files(urls):
#     try:
#         for fn, url in urls.items():
#             r = requests.get(url)
#             open(fn, 'wb').write(r.content)
#     except Exception as e:
#         st.write(e)
        
# a function to download the image from the selected file
@st.cache(show_spinner=False)
def read_img(img):
    image = cv2.imread(im, cv2.IMREAD_COLORg)
    image = image[:, :, [2, 1, 0]] # BGR -> RGB
    return image

# Yolo function
def yolo_v3(image, confidence_threshold=0.5, overlap_threshold=0.3):

	# Load model architecture
    net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
    output_layer_names = net.getLayerNames()
    output_layer_names = [output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Set input and get output
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer_names)

    boxes, confidences, class_IDs = [], [], []
    H, W = image.shape[:2]

    # For each detected object, compute the box, find the score, ignore if below
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box.astype("int")
                x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_IDs.append(classID)

    # Write the name of detected objects above image
    f = open("classes.txt", "r")
    f = f.readlines()
    f = [line.rstrip('\n') for line in list(f)]
    try:
    	st.subheader("Detected objects: " + ', '.join(list(set([f[obj] for obj in class_IDs]))))
    except IndexError:
    	st.write("Nothing detected")

    # Apply non-max suppression to identify best bounding box
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlap_threshold)
    xmin, xmax, ymin, ymax, labels = [], [], [], [], []
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            xmin.append(x)
            ymin.append(y)
            xmax.append(x+w)
            ymax.append(y+h)
    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

    # Add a layer on top on a detected object 
    LABEL_COLORS = [0, 255, 0]
    image_with_boxes = image.astype(np.float64)
    for _, (xmin, ymin, xmax, ymax) in boxes.iterrows():
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += LABEL_COLORS
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2

    # Display the final image
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

img_type = st.sidebar.selectbox("Select image type?", ['Surf'])

if img_type == 'Surf':
    image_url = "images/surf.jpg"


image = read_img(image_url)

# urls = {'yolov3.weights':"https://onedrive.live.com/download?cid=2FC9D36DB856FA39&resid=2FC9D36DB856FA39%2134980&authkey=ADwI9Y5h5HCc5kU",
#         'yolov3.cfg':"https://onedrive.live.com/download?cid=2FC9D36DB856FA39&resid=2FC9D36DB856FA39%2134979&authkey=AGCGt7UDRRx4_L8"}

# get_files(urls)

# uploaded_file = st.file_uploader("Choose an image", type="jpg")
# if uploaded_file is not None:
#     with Image.open(uploaded_file) as img:
#         image = np.array(img)
        # yolo_v3(image, confidence_threshold)
    
# Get the boxes for the objects detected by YOLO by running the YOLO model.
yolo_v3(image, confidence_threshold)
    