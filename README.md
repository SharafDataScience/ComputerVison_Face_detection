# Face and Eye Detection using Haar Cascade Classifiers (OpenCV)

This project demonstrates how to detect human **faces** and **eyes** in images using **Haar Cascade Classifiers** with **OpenCV** in Python. It utilizes pre-trained XML classifiers for both face and eye detection.

---

## Project Structure

```
face-eye-detection/
│
├── Face Detection Haar Cascade Classifiers _OpenCV.ipynb  # Jupyter notebook with implementation
├── haarcascade_frontalface_default.xml                    # Haar cascade for face detection
├── haarcascade_eye.xml                                   # Haar cascade for eye detection
├── README.md                                              # Project documentation
```

---

## Requirements

Make sure you have the following installed:

- Python 3.x
- OpenCV
- Jupyter Notebook (optional, for viewing the `.ipynb`)

Install dependencies using:

```bash
pip install opencv-python
```

---

## How It Works

The notebook performs the following steps:

1. **Load Haar Cascade Classifiers**:
   - `haarcascade_frontalface_default.xml` – for frontal face detection
   - `haarcascade_eye.xml` – for detecting eyes within detected faces

2. **Read Image or Use Webcam**:
   - Load a static image or activate your webcam for real-time detection.

3. **Convert to Grayscale**:
   - Haar cascades require grayscale images.

4. **Detect Faces**:
   - Use `face_cascade.detectMultiScale(...)` to find faces.

5. **Detect Eyes in Detected Faces**:
   - Use `eye_cascade.detectMultiScale(...)` inside each detected face region.

6. **Display Results**:
   - Draw rectangles around faces and eyes.

---


## Parameters You Can Tune

- `scaleFactor`: Specifies how much the image size is reduced at each image scale. Default is `1.3`.
- `minNeighbors`: Specifies how many neighbors each rectangle should have to retain it. Default is `5`.
- `minSize`: Minimum possible object size. Objects smaller than that are ignored.

Example:

```python
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
```

---

## Output

- The output will show the original image with:
  - Blue rectangles around detected faces
  - Green rectangles around detected eyes

For webcam-based real-time detection, just loop over frames from `cv2.VideoCapture(0)` and apply the same detection steps per frame.

---




