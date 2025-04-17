import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QGraphicsScene, QGraphicsPixmapItem
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtCore import Qt, QFile
from PySide2.QtUiTools import QUiLoader

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        file = QFile("GUI.ui")
        file.open(QFile.ReadOnly)
        self.ui = QUiLoader().load(file)
        file.close()
        self.image = None
        self.processed_image = None

        # 连接按钮事件
        self.ui.loadButton.clicked.connect(self.load_image)
        self.ui.saveButton.clicked.connect(self.save_image)
        self.ui.grayButton.clicked.connect(self.to_gray)
        self.ui.binaryButton.clicked.connect(self.to_binary)
        self.ui.addNoiseButton.clicked.connect(self.add_noise)
        self.ui.faceDetectButton.clicked.connect(self.face_detect)
        self.ui.openButton.clicked.connect(self.open_operation)
        self.ui.closeButton.clicked.connect(self.close_operation)
        self.ui.erosionButton.clicked.connect(self.erosion_operation)
        self.ui.dilationButton.clicked.connect(self.dilation_operation)
        self.ui.edgeDetectionComboBox.currentIndexChanged.connect(self.edge_detection)
        self.ui.segmentationComboBox.currentIndexChanged.connect(self.segment_image)

    def load_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "打开图像文件", "",
                                                  "Images (*.png *.xpm *.jpg *.tif);;All Files (*)", options=options)
        if fileName:
            self.image = cv2.imread(fileName)
            if self.image is not None:
                self.display_image(self.image, self.ui.inputImageLabel)
                self.display_histogram(self.image, self.ui.inputHistogram)
                self.processed_image = self.image.copy()
            else:
                QMessageBox.critical(self, "错误", "加载图像失败。")

    def display_image(self, image, view):
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        out_image = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        out_image = out_image.rgbSwapped()

        pixmap = QPixmap.fromImage(out_image)
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(pixmap))
        view.setScene(scene)
        view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def display_histogram(self, image, view):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        histogram = histogram.flatten()

        plt.figure(figsize=(6, 4))
        plt.plot(histogram, color='b')
        plt.title("Histogram")
        plt.xlabel("Pixel value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig("histogram.png")
        plt.close()

        hist_image = cv2.imread("histogram.png")
        self.display_image(hist_image, view)

    def to_gray(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
            self.display_image(gray_image, self.ui.outputImageLabel)
            self.display_histogram(gray_image, self.ui.outputHistogram)
            self.processed_image = gray_image

    def to_binary(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
            self.display_image(binary_image, self.ui.outputImageLabel)
            self.display_histogram(binary_image, self.ui.outputHistogram)
            self.processed_image = binary_image

    def add_noise(self):
        if self.image is not None:
            noisy_image = self.image.copy()
            mean = 0
            var = 1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, self.image.shape).astype('uint8')
            noisy_image = cv2.add(self.image.copy(), gauss)
            self.display_image(noisy_image, self.ui.outputImageLabel)
            self.display_histogram(noisy_image, self.ui.outputHistogram)
            self.processed_image = noisy_image

    def face_detect(self):
        if self.image is not None:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray_image = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
            original_image = self.image.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self.display_image(original_image, self.ui.outputImageLabel)
            self.display_histogram(original_image, self.ui.outputHistogram)
            self.processed_image = original_image

    def open_operation(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
            kernel = np.ones((5, 5), np.uint8)
            opened_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
            self.display_image(opened_image, self.ui.outputImageLabel)
            self.display_histogram(opened_image, self.ui.outputHistogram)
            self.processed_image = opened_image

    def close_operation(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
            kernel = np.ones((5, 5), np.uint8)
            closed_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
            self.display_image(closed_image, self.ui.outputImageLabel)
            self.display_histogram(closed_image, self.ui.outputHistogram)
            self.processed_image = closed_image

    def erosion_operation(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
            kernel = np.ones((5, 5), np.uint8)
            eroded_image = cv2.erode(gray_image, kernel, iterations=1)
            self.display_image(eroded_image, self.ui.outputImageLabel)
            self.display_histogram(eroded_image, self.ui.outputHistogram)
            self.processed_image = eroded_image

    def dilation_operation(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
            kernel = np.ones((5, 5), np.uint8)
            dilated_image = cv2.dilate(gray_image, kernel, iterations=1)
            self.display_image(dilated_image, self.ui.outputImageLabel)
            self.display_histogram(dilated_image, self.ui.outputHistogram)
            self.processed_image = dilated_image

    def edge_detection(self):
        if self.image is not None:
            method_index = self.ui.edgeDetectionComboBox.currentIndex()
            gray_image = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)

            if method_index == 0:  # Canny 边缘检测
                edges = cv2.Canny(gray_image, 100, 200)
                self.display_image(edges, self.ui.outputImageLabel)
                self.display_histogram(edges, self.ui.outputHistogram)
                self.processed_image = edges
            elif method_index == 1:  # Sobel 边缘检测
                sobel_edges = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=3)
                sobel_edges = cv2.convertScaleAbs(sobel_edges)
                self.display_image(sobel_edges, self.ui.outputImageLabel)
                self.display_histogram(sobel_edges, self.ui.outputHistogram)
                self.processed_image = sobel_edges
            elif method_index == 2:  # Prewitt 边缘检测
                # 定义 Prewitt 滤波器
                kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
                kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
                prewittx = cv2.filter2D(gray_image, -1, kernelx)
                prewitty = cv2.filter2D(gray_image, -1, kernely)
                prewitt_edges = cv2.addWeighted(prewittx, 0.5, prewitty, 0.5, 0)
                self.display_image(prewitt_edges, self.ui.outputImageLabel)
                self.display_histogram(prewitt_edges, self.ui.outputHistogram)
                self.processed_image = prewitt_edges
            elif method_index == 3:  # Roberts 边缘检测
                # 定义 Roberts 滤波器
                kernelx = np.array([[1, 0], [0, -1]], dtype=int)
                kernely = np.array([[0, 1], [-1, 0]], dtype=int)
                robertsx = cv2.filter2D(gray_image, -1, kernelx)
                robertsy = cv2.filter2D(gray_image, -1, kernely)
                roberts_edges = cv2.addWeighted(robertsx, 0.5, robertsy, 0.5, 0)
                self.display_image(roberts_edges, self.ui.outputImageLabel)
                self.display_histogram(roberts_edges, self.ui.outputHistogram)
                self.processed_image = roberts_edges

    def segment_image(self):
        if self.image is not None:
            method_index = self.ui.segmentationComboBox.currentIndex()
            gray_image = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)

            if method_index == 0:  # Otsu 阈值分割
                _, segmented_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.display_image(segmented_image, self.ui.outputImageLabel)
                self.display_histogram(segmented_image, self.ui.outputHistogram)
                self.processed_image = segmented_image
            elif method_index == 1:  # K-means 聚类分割
                Z = gray_image.reshape((-1, 1))
                Z = np.float32(Z)

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                K = 2
                _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                centers = np.uint8(centers)
                segmented_image = centers[labels.flatten()]
                segmented_image = segmented_image.reshape((gray_image.shape))
                self.display_image(segmented_image, self.ui.outputImageLabel)
                self.display_histogram(segmented_image, self.ui.outputHistogram)
                self.processed_image = segmented_image
            elif method_index == 2:  # GrabCut 算法
                mask = np.zeros(self.image.shape[:2], np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                rect = (50, 50, self.image.shape[1] - 100, self.image.shape[0] - 100)
                cv2.grabCut(self.image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                segmented_image = self.image * mask2[:, :, np.newaxis]
                self.display_image(segmented_image, self.ui.outputImageLabel)
                self.display_histogram(segmented_image, self.ui.outputHistogram)
                self.processed_image = segmented_image

    def save_image(self):
        if self.processed_image is not None:
            options = QFileDialog.Options()
            fileName, _ = QFileDialog.getSaveFileName(self, "保存图像", "", "Images (*.jpg )", options=options)
            if fileName:
                cv2.imwrite(fileName, self.processed_image)
                QMessageBox.information(self, "图像已保存", "图像保存成功！")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.ui.show()
    sys.exit(app.exec_())
