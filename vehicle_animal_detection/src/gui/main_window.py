import sys
import cv2
import numpy as np
import yaml
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QSlider, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap

from detection.yolo_detector import YOLOTinyDetector
from classification.classifier import Classifier

class ProcessingThread(QThread):
    progress_signal = pyqtSignal(int)
    frame_processed_signal = pyqtSignal(np.ndarray)
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)
    alert_signal = pyqtSignal(str)  

    def __init__(self, config, video_path, config_path):
        super().__init__()
        self.config = config
        self.video_path = video_path
        self.config_path = config_path
        self.detector = YOLOTinyDetector(self.config)
        self.classifier = Classifier(self.config_path)

    def process_frame(self, frame, detections):
        animal_detected = False
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            class_name = detection['class']
            confidence = detection['confidence']
            
            animals = ['dog', 'cat', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
            vehicles = ['car', 'truck', 'bus', 'motorcycle']
            
            if class_name in animals or class_name in vehicles:
                object_img = frame[y1:y2, x1:x2]
                
                try:
                    classification = self.classifier.classify(object_img)
                    
                    if classification and classification['confidence'] >= self.config['models']['classifier']['confidence_threshold']:
                        if classification['class'] == 'animal':
                            color = (0, 255, 0)  
                            label = f"Animal: {class_name} ({confidence:.2f})"
                            animal_detected = True
                        else:
                            color = (0, 0, 255)  
                            label = f"Vehicle: {class_name} ({confidence:.2f})"
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        
                        if classification['class'] == 'animal':
                            self.alert_signal.emit(self.config['alerts']['animal_detected'])
                except Exception as e:
                    if not hasattr(self, 'last_error') or str(e) != self.last_error:
                        print(f"Error in classification: {str(e)}")
                        self.last_error = str(e)
        
        if animal_detected:
            self.alert_signal.emit(self.config['alerts']['animal_detected'])
        
        return frame

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = []

        for i in range(0, total_frames, self.config['performance']['frame_skip']):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, tuple(self.config['performance']['target_resolution']))

            detections = self.detector.detect(frame)
            processed_frame = self.process_frame(frame, detections)
            processed_frames.append(processed_frame)

            self.frame_processed_signal.emit(processed_frame)
            self.progress_signal.emit(int((i + 1) / total_frames * 100))

            if self.isInterruptionRequested():
                break

        cap.release()
        self.finished_signal.emit(processed_frames)


class PlaybackThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    finished_signal = pyqtSignal()

    def __init__(self, processed_frames):
        super().__init__()
        self.processed_frames = processed_frames
        self.current_frame = 0
        self._run_flag = True
        self.delay = 0

    def run(self):
        while self._run_flag and self.current_frame < len(self.processed_frames):
            self.change_pixmap_signal.emit(self.processed_frames[self.current_frame])
            self.current_frame += 1
            if self.delay > 0:
                self.msleep(self.delay)
        self.finished_signal.emit()

    def stop(self):
        self._run_flag = False
        self.wait()

    def set_delay(self, delay):
        self.delay = delay

class MainWindow(QMainWindow):
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path  
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.setWindowTitle(self.config['gui']['window_title'])
        self.setGeometry(100, 100, self.config['gui']['window_size']['width'], self.config['gui']['window_size']['height'])

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        self.load_button = QPushButton("Upload Video")
        self.load_button.clicked.connect(self.load_video)
        self.layout.addWidget(self.load_button)

        self.process_button = QPushButton("Process Video")
        self.process_button.clicked.connect(self.process_video)
        self.process_button.setEnabled(False)
        self.layout.addWidget(self.process_button)

        self.play_pause_button = QPushButton("Play/Pause")
        self.play_pause_button.clicked.connect(self.play_pause_video)
        self.play_pause_button.setEnabled(False)
        self.layout.addWidget(self.play_pause_button)

        self.progress_bar = QProgressBar(self)
        self.layout.addWidget(self.progress_bar)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(0)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(10)
        self.speed_slider.valueChanged.connect(self.change_speed)
        self.layout.addWidget(self.speed_slider)

        self.alert_label = QLabel(self)
        self.alert_label.setAlignment(Qt.AlignCenter)
        self.alert_label.setStyleSheet("background-color: red; color: white; font-size: 18px;")
        self.alert_label.hide()
        self.layout.addWidget(self.alert_label)

        self.video_path = None
        self.processed_frames = None
        self.processing_thread = None
        self.playback_thread = None
        self.video_playing = False

    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.video_path = file_name
            self.process_button.setEnabled(True)
            self.show_alert("Video uploaded. Press 'Edit Video' to start processing.")

    def process_video(self):
        if self.video_path:
            self.processing_thread = ProcessingThread(self.config, self.video_path, self.config_path)
            self.processing_thread.progress_signal.connect(self.update_progress)
            self.processing_thread.frame_processed_signal.connect(self.update_image)
            self.processing_thread.finished_signal.connect(self.processing_finished)
            self.processing_thread.error_signal.connect(self.show_error)
            self.processing_thread.alert_signal.connect(self.show_alert) 
            self.processing_thread.start()
            self.process_button.setEnabled(False)
            self.load_button.setEnabled(False)
            self.show_alert("Processing...")

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        if value % 10 == 0: 
            self.show_alert(f"Processing... {value}% completed")

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def processing_finished(self, processed_frames):
        self.processed_frames = processed_frames
        self.play_pause_button.setEnabled(True)
        self.show_alert("Processing completed. Press 'Play' to start playback.")

    def play_pause_video(self):
        if not self.video_playing:
            if self.processed_frames:
                if not self.playback_thread:
                    self.playback_thread = PlaybackThread(self.processed_frames)
                    self.playback_thread.change_pixmap_signal.connect(self.update_image)
                    self.playback_thread.finished_signal.connect(self.playback_finished)
                self.playback_thread.start()
                self.video_playing = True
                self.play_pause_button.setText("Pause")
        else:
            if self.playback_thread:
                self.playback_thread.stop()
            self.video_playing = False
            self.play_pause_button.setText("Play")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def show_alert(self, message):
        self.alert_label.setText(message)
        self.alert_label.show()
        QTimer.singleShot(3000, self.alert_label.hide)

    def show_error(self, error_message):
        print(f"Error: {error_message}")
        self.show_alert(f"Error: {error_message}")

    def change_speed(self, value):
        delay = value 
        if self.playback_thread:
            self.playback_thread.set_delay(delay)

    def playback_finished(self):
        self.video_playing = False
        self.play_pause_button.setText("Play")
        self.show_alert("Reproduction completed.")

    def closeEvent(self, event):
        if self.processing_thread:
            self.processing_thread.wait()
        if self.playback_thread:
            self.playback_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow('config/config.yaml')
    main_window.show()
    sys.exit(app.exec_())