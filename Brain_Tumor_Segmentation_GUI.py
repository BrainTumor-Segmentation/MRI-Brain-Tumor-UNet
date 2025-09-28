import sys
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QGraphicsScene

# -------------------------------
# Replace with your paths
# -------------------------------
UI_PATH = "MRI.ui"              # put your .ui file path here
MODEL_PATH = "best_model.pth"   # put your trained model path here


# -------------------------------
# U-Net model definition
# -------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        bn = self.bottleneck(p4)

        up4 = self.up4(bn)
        merge4 = torch.cat([up4, d4], dim=1)
        c4 = self.conv4(merge4)
        up3 = self.up3(c4)
        merge3 = torch.cat([up3, d3], dim=1)
        c3 = self.conv3(merge3)
        up2 = self.up2(c3)
        merge2 = torch.cat([up2, d2], dim=1)
        c2 = self.conv2(merge2)
        up1 = self.up1(c2)
        merge1 = torch.cat([up1, d1], dim=1)
        c1 = self.conv1(merge1)

        out = self.final_conv(c1)
        return out


# -------------------------------
# Main Application
# -------------------------------
class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        uic.loadUi(UI_PATH, self)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = UNet(in_channels=1, out_channels=1).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

        # Initialize image vars
        self.current_image = None
        self.mask_image = None

        # Connect buttons to actions
        self.pushButton.clicked.connect(self.load_image)    # Load Image
        self.pushButton_2.clicked.connect(self.reset_image) # Reset
        self.pushButton_3.clicked.connect(self.save_image)  # Save

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.tif)")
        if file_path:
            # Load colored image for display
            img_color = cv2.imread(file_path)  
            img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

            # Keep grayscale for model input
            self.current_image = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

            # Show original image in graphicsView
            scene = QGraphicsScene()
            qimg = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0],
                        img_rgb.strides[0], QImage.Format_RGB888)
            scene.addPixmap(QPixmap.fromImage(qimg))
            self.graphicsView.setScene(scene)

            # Predict mask automatically
            self.predict_mask()


    def predict_mask(self):
        if self.current_image is None:
            return

        img = cv2.resize(self.current_image, (256, 256))
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0

        with torch.no_grad():
            output = self.model(tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred = (pred > 0.5).astype(np.uint8) * 255

        # Show mask
        self.mask_image = pred
        mask_scene = QGraphicsScene()
        qimg = QImage(pred.data, pred.shape[1], pred.shape[0],
                      pred.strides[0], QImage.Format_Grayscale8)
        mask_scene.addPixmap(QPixmap.fromImage(qimg))
        self.graphicsView_2.setScene(mask_scene)

    def reset_image(self):
        if self.current_image is None and self.mask_image is None:
            QMessageBox.information(self, "Info", "No image to reset.")
            return

        reply = QMessageBox.question(self, "Confirm Reset", "Are you sure you want to reset?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.graphicsView.setScene(None)
            self.graphicsView_2.setScene(None)
            self.current_image = None
            self.mask_image = None

    def save_image(self):
        if self.mask_image is None:
            QMessageBox.information(self, "Info", "No mask image to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPG Files (*.jpg)")
        if file_path:
            cv2.imwrite(file_path, self.mask_image)
            QMessageBox.information(self, "Saved", f"Image saved to {file_path}")


# -------------------------------
# Run the app
# -------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
