
# End-to-End Sign Language Detection Project

This repository contains the code for an **End-to-End Sign Language Detection System**, which leverages the power of a **YOLO (You Only Look Once)** pretrained model that has been fine-tuned using a custom dataset. The system is designed to detect hand gestures and classify them as different signs in **sign language**. The dataset used is custom, which was collected by myself and enhanced with **Roboflow** for annotation.

---

## 📑 **Table of Contents**  
1. Project Overview  
2. Key Features  
3. Dataset  
4. YOLO Model Architecture  
5. Training and Fine-Tuning  
6. Deployment  
7. How to Use  
8. Acknowledgements  

---

## 🛠 **Project Overview**  
This project demonstrates an end-to-end pipeline for sign language detection:
1. **Custom Dataset**: A custom dataset of hand gestures in sign language was collected and annotated using **Roboflow**.
2. **YOLO Architecture**: Fine-tuning a **YOLO** model for detecting and classifying the gestures.
3. **Training**: Training the YOLO model on the custom dataset to improve accuracy.
4. **Detection and Classification**: Detecting hand gestures in real-time using a camera or images, and classifying them into different sign language symbols.
5. **Deployment**: Deploying the model into a web or application interface for real-time use.

---

## 🔑 **Key Features**  
- **Pretrained YOLO Model**: The project uses a pretrained YOLO model for object detection, which has been fine-tuned with custom data for improved accuracy in sign language gesture recognition.
- **Real-Time Detection**: Capable of detecting hand gestures in real-time via a webcam or image.
- **Custom Dataset**: A dataset of hand gestures in sign language collected and annotated using **Roboflow**.
- **Accuracy and Performance**: The model delivers efficient performance with high accuracy in detecting sign language gestures.
- **Deployment**: The model can be deployed into an application interface for real-time sign language translation.

---

## 📂 **Dataset**  
The custom dataset used for this project was collected by me and annotated using **Roboflow**, a tool that helps in labeling images for object detection tasks. The dataset contains hand gesture images representing different signs in sign language. It includes:

- **Hand Gesture Images**: Captures of different hand gestures representing different sign language symbols.
- **Annotations**: Each image is annotated with bounding boxes and labels to train the model effectively.

The dataset is divided into training, validation, and test sets, ensuring that the model is evaluated correctly on unseen data.

---

## 🧑‍💻 **YOLO Model Architecture**  

YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system. The model is designed to perform object detection tasks efficiently by framing it as a **single regression problem**. Here's an in-depth look at the YOLO architecture and how it's applied to sign language detection:

### **Architecture Breakdown**:

1. **Input Image**:  
   The YOLO model takes an input image and divides it into an SxS grid. Each grid cell is responsible for predicting bounding boxes and classifying the object within that grid.

2. **Grid Cell Predictions**:  
   For each grid cell, YOLO predicts the following:
   - **Bounding Box Coordinates**: The `x`, `y`, `w`, and `h` values which represent the center, width, and height of the bounding box.
   - **Confidence Score**: The confidence score predicts how likely the model is to correctly predict the object within that bounding box.
   - **Class Probabilities**: The model also predicts the probabilities of different classes (in this case, the sign language gestures).

3. **Anchor Boxes**:  
   YOLO uses predefined anchor boxes to help predict bounding boxes more effectively. These anchors are designed to cover various shapes and sizes of objects.

4. **Output Layer**:  
   The output of the YOLO network is a tensor containing predictions for each grid cell. Each prediction contains bounding box coordinates, confidence score, and class probabilities for the detected object (sign language gestures).

5. **Non-Maximum Suppression (NMS)**:  
   After predictions are made, YOLO uses NMS to filter out redundant boxes, keeping only the ones with the highest confidence score.

### **How YOLO Works for Sign Language Detection**:  
In the case of sign language detection, the model's goal is to detect the hand gesture, classify it into the correct sign language symbol, and provide real-time feedback. Fine-tuning YOLO for this task involves:
- **Dataset Customization**: The YOLO model was retrained on the custom sign language dataset to recognize gestures.
- **Real-Time Object Detection**: The trained YOLO model is used to detect the hand gestures and classify them into one of the predefined categories.

---

## 🏋️‍♂️ **Training and Fine-Tuning**  
- **Pretrained YOLO Model**: The project starts by loading a pretrained YOLO model ( YOLOv5) from a model zoo. This provides a good base for fine-tuning on the custom dataset.
- **Data Augmentation**: Techniques like rotation, flipping, and scaling are applied to augment the dataset and improve generalization.
- **Training Process**: 
   - The dataset is loaded and split into training, validation, and test sets.
   - The model is fine-tuned on the custom dataset using **transfer learning**.
   - Hyperparameters like learning rate, batch size, and number of epochs are tuned for optimal results.
   - Loss functions like **objectness loss** and **classification loss** are used to optimize the model's performance.
  
---

## 🌐 **Deployment**  
Once the model is trained and fine-tuned, it is deployed in an application. The deployment can be done via:

- **Real-Time Gesture Recognition**: The model can be deployed into an application where users can use their webcam to detect sign language gestures.
- **Web Interface**: A Flask or Streamlit app can be used to create a simple web interface where users can input images or use a webcam to detect gestures.

### **Running Locally**:  
1. Clone the repository:
   ```bash
   git clone https://github.com/HelpRam/End-to-End-Sign-Language-Detection.git
   cd End-to-End-Sign-Language-Detection  
   ```

2. Install dependencies:
   ```bash  
   pip install -r requirements.txt  
   ```

3. Run the detection script:
   ```bash  
   python detect_sign_language.py  
   ```

4. Access the web interface or use your webcam for real-time sign language detection.

---

## 💡 **How to Use**  
1. **Webcam Input**: Use your webcam to perform real-time sign language detection.
2. **Upload Images**: Upload an image with a hand gesture, and the model will classify it into the correct sign language symbol.
3. **View Results**: The detected sign language symbol and bounding box will be displayed.

---

## 🙌 **Acknowledgements**  
- The **YOLO** model for real-time object detection.
- **Roboflow** for helping with dataset annotation and preparation.
- Libraries such as **OpenCV**, **PyTorch**, and **TensorFlow** for object detection and model training.
- Open-source community for making object detection techniques accessible.

Feel free to explore, contribute, or provide feedback! 😊

---

This README provides a complete overview of your project, including the YOLO architecture and fine-tuning process for sign language detection. It also outlines how the system is deployed and how others can use it.
