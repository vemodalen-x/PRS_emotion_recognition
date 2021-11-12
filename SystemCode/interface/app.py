from FaceDetectionModule import FaceDetector
from ui.ui_detect_ui import Ui_MainWindow

from PyQt5.Qt import *
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import sys
import time
import paddlex as pdx
# -----------------------------



# main windows
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.timer_video = QtCore.QTimer()  # time count
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.num_stop = 1 
        self.output_folder = 'output/'
        self.vid_writer = None
        self.detector = FaceDetector(minDetectionCon=0.5)
        # initilize filename
        self.openfile_name_model = None

    # slot
    def init_slots(self):
        self.ui.pushButton_img.clicked.connect(self.button_image_open)
        self.ui.pushButton_video.clicked.connect(self.button_video_open)
        self.ui.pushButton_camer.clicked.connect(self.button_camera_open)
        self.ui.pushButton_stop.clicked.connect(self.button_video_stop)
        self.ui.pushButton_finish.clicked.connect(self.finish_detect)

        self.timer_video.timeout.connect(self.show_video_frame)  

    def detect(self, name_list, img):
        '''
        :param name_list: file_list
        :param img: img
        :return: info_show:detect the text
        '''
        # showimg = img
        img, bboxs,emotions = self.detector.findFaces(img)
        print(type(img))
        print(img.shape)
        info_show='bboxs"'+str(bboxs)+'\nemotions:'+str(emotions)

        return info_show

    # open image 
    def button_image_open(self):
        print('button_image_open')
        name_list = []
        try:
            img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "open image", "data/images",
                                                                "*.jpg;;*.png;;All Files(*)")
        except OSError as reason:
            print('please check the filepath' + str(reason))
        else:
            # if image is none
            if not img_name:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"open error", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                img = cv2.imread(img_name)
                print("img_name:", img_name)
                info_show = self.detect(name_list, img)
                print(info_show)
                info_show = str(info_show)

                # get sys time as filename
                now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
                file_extension = img_name.split('.')[-1]
                new_filename = now + '.' + file_extension  
                file_path = self.output_folder + 'img_output/' + new_filename
                cv2.imwrite(file_path, img)

                # show info
                self.ui.textBrowser.setText(info_show)


                # show info
                self.result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
          
                self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                          QtGui.QImage.Format_RGB32)
                self.ui.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                self.ui.label.setScaledContents(True)  # adaptive

    def set_video_name_and_path(self):
        # get sys time 
        now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        # if vid_cap:  # video
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # save filepath
        save_path = self.output_folder + 'video_output/' + now + '.mp4'
        return fps, w, h, save_path

    # opeb video and detect
    def button_video_open(self):
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "open video", "data/videos", "*.mp4;;*.avi;;All Files(*)")
        flag = self.cap.open(video_name)
        if not flag:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"open video error", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            # -------------------------write----------------------------------#
            fps, w, h, save_path = self.set_video_name_and_path()
            self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            self.timer_video.start(30)  # every 30 sec reset
            # off other buttons
            self.ui.pushButton_video.setDisabled(True)
            self.ui.pushButton_img.setDisabled(True)
            self.ui.pushButton_camer.setDisabled(True)

    # open camera
    def button_camera_open(self):
        print("Open camera to detect")
        # set id
        camera_num = 0
        # open
        self.cap = cv2.VideoCapture(camera_num)
        # check
        bool_open = self.cap.isOpened()
        if not bool_open:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"open camer error", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            fps, w, h, save_path = self.set_video_name_and_path()
            fps = 5  # set FPS
            self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.timer_video.start(30)
            self.ui.pushButton_video.setDisabled(True)
            self.ui.pushButton_img.setDisabled(True)
            self.ui.pushButton_camer.setDisabled(True)

    # video frame
    def show_video_frame(self):
        name_list = []
        # self.detector = FaceDetector()
        flag, img = self.cap.read()
        if img is not None:
            info_show = self.detect(name_list, img)  # write to img

            self.vid_writer.write(img)  # write to video
            print(info_show)
            info_show = str(info_show)
            # show
            self.ui.textBrowser.setText(info_show)


            show = cv2.resize(img, (640, 480))  # show result
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.ui.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.ui.label.setScaledContents(True)  # adaptive

        else:
            print("img is None")
            self.timer_video.stop()
            # release
            self.cap.release()  # video_capture
            self.vid_writer.release()  # video_writer
            self.ui.label.clear()
            # off other button
            self.ui.pushButton_video.setDisabled(False)
            self.ui.pushButton_img.setDisabled(False)
            self.ui.pushButton_camer.setDisabled(False)

    # stop and resume
    def button_video_stop(self):
        self.timer_video.blockSignals(False)
        # stop
        if self.timer_video.isActive() == True and self.num_stop % 2 == 1:
            self.ui.pushButton_stop.setText(u'stop detection')  # stop
            self.num_stop = self.num_stop + 1  
            self.timer_video.blockSignals(True)
        # resume
        else:
            self.num_stop = self.num_stop + 1
            self.ui.pushButton_stop.setText(u'resume detection')

    # stop video detection
    def finish_detect(self):
        # self.timer_video.stop()
        self.cap.release()  # release video_capture
        self.vid_writer.release()  # release video_writer
        self.ui.label.clear()  # clean label
        # enable other button
        self.ui.pushButton_video.setDisabled(False)
        self.ui.pushButton_img.setDisabled(False)
        self.ui.pushButton_camer.setDisabled(False)

        # check
        if self.num_stop % 2 == 0:
            print("Reset stop/begin!")
            self.ui.pushButton_stop.setText(u'stop/resume')
            self.num_stop = self.num_stop + 1
            self.timer_video.blockSignals(False)


if __name__ == '__main__':
    # initialize
    app = QApplication(sys.argv)
    # create
    win = MainWindow()
    # show
    win.show()
    # loop
    sys.exit(app.exec_())
