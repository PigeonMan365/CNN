import sys
import os
import glob
import torch
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

MODEL_PATH = "cnn_resize_0_1.ts.pt"
IMAGE_FOLDER = "images"

# -------------------------
# Model Load
# -------------------------
model = torch.jit.load(MODEL_PATH)
model.eval()

def get_prob(out):
    if isinstance(out, (tuple, list)):
        out = out[0]
    return float(out.reshape(-1)[0])

# -------------------------
# CNN Stage Visual Approximations
# -------------------------
def conv_stage_maps(img):
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    lap = cv2.Laplacian(img, cv2.CV_32F)

    outs = []
    for m in [sobelx, sobely, lap]:
        m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX)
        outs.append(m.astype(np.uint8))
    return outs

def pool_sim(img, factor):
    small = cv2.resize(img, (img.shape[1]//factor, img.shape[0]//factor))
    return cv2.resize(small, img.shape[::-1])

# -------------------------
# Qt Helpers
# -------------------------
def to_pixmap(gray):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

# -------------------------
# Main Window
# -------------------------
class Demo(QMainWindow):

    def __init__(self):
        super().__init__()

        self.files = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.png")))
        if not self.files:
            raise RuntimeError("No PNG images found")

        self.idx = 0

        self.setWindowTitle("CNN Live Technical Visualization Demo")
        self.showFullScreen()

        root = QWidget()
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)

        # -------------------------
        # Title
        # -------------------------
        self.title = QLabel("Convolutional Neural Network — Live Inference Pipeline")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size:32px;font-weight:bold;padding:20px;")
        layout.addWidget(self.title)

        # -------------------------
        # Grid of stages
        # -------------------------
        self.grid = QGridLayout()
        layout.addLayout(self.grid)

        self.labels = []
        self.captions = []

        stage_names = [
            "Input Image",
            "Convolution Response X",
            "Convolution Response Y",
            "Edge / Feature Map",
            "Pooling Stage 1",
            "Pooling Stage 2"
        ]

        stage_desc = [
            "Raw grayscale tensor fed to CNN",
            "Simulated conv filter detecting vertical gradients",
            "Simulated conv filter detecting horizontal gradients",
            "Edge intensity aggregation",
            "Spatial downsampling (feature compaction)",
            "Deep spatial abstraction"
        ]

        for i in range(6):
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setMinimumSize(320, 320)

            caption = QLabel(f"{stage_names[i]}\n{stage_desc[i]}")
            caption.setAlignment(Qt.AlignCenter)
            caption.setStyleSheet("font-size:14px;padding:8px;")

            self.grid.addWidget(img_label, i//3*2, i%3)
            self.grid.addWidget(caption, i//3*2+1, i%3)

            self.labels.append(img_label)
            self.captions.append(caption)

        # -------------------------
        # Bottom Panel
        # -------------------------
        bottom = QHBoxLayout()
        layout.addLayout(bottom)

        self.result = QLabel("Press button to process first image")
        self.result.setStyleSheet("font-size:22px;padding:20px;")
        bottom.addWidget(self.result)

        self.button = QPushButton("Process Next Image")
        self.button.setStyleSheet("font-size:20px;padding:15px;")
        self.button.clicked.connect(self.process_next)
        bottom.addWidget(self.button)

    # -------------------------
    # Processing Step
    # -------------------------
    def process_next(self):

        path = self.files[self.idx]
        self.idx = (self.idx + 1) % len(self.files)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

        t0 = cv2.getTickCount()
        with torch.no_grad():
            out = model(x)
        t1 = cv2.getTickCount()

        ms = (t1 - t0) / cv2.getTickFrequency() * 1000
        prob = get_prob(out)
        label = "MALWARE" if prob >= 0.5 else "BENIGN"

        # Stage visuals
        maps = conv_stage_maps(img)
        p1 = pool_sim(img, 2)
        p2 = pool_sim(img, 4)

        visuals = [img, maps[0], maps[1], maps[2], p1, p2]

        for lab, im in zip(self.labels, visuals):
            lab.setPixmap(to_pixmap(im).scaled(
                420, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.result.setText(
            f"File: {os.path.basename(path)}     "
            f"Prediction: {label}     "
            f"Confidence: {prob:.3f}     "
            f"Inference Time: {ms:.1f} ms"
        )

# -------------------------
# Run
# -------------------------
app = QApplication(sys.argv)
demo = Demo()
demo.show()
sys.exit(app.exec_())
