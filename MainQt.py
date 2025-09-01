# -*- coding: utf-8 -*-
import logging
import cv2
import os
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QMessageBox,
    QStatusBar, QLabel,QMenuBar,QPlainTextEdit,
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot,QObject
from PySide6.QtGui import QImage, QIcon, QPixmap

from ui import Ui_MainWindow
from yolo5_model_5 import YOLOv5Model

# 自定义一个 Qt 线程安全的日志 Handler
# --------------------------------------------------
class QtSignaler(QObject):
    """在日志线程里 emit，主线程里接收，保证线程安全"""
    log_record = Signal(str)     # 发送整行格式化后的日志


class QPlainTextEditHandler(logging.Handler):
    """
    把 logging 产生的每一条记录送到 QPlainTextEdit。
    通过 Qt 信号槽机制保证跨线程安全。
    """
    def __init__(self, parent_widget: QPlainTextEdit):
        super().__init__()
        self.widget = parent_widget
        self.signaler = QtSignaler()
        # 把信号连接到真正插入文本的槽函数
        self.signaler.log_record.connect(self._append_plain_text)
        # 设置样式：不同级别不同颜色（可选）
        '''
        self.widget.setStyleSheet("""
            QPlainTextEdit {
                font-family: Consolas, "Courier New", monospace;
                font-size: 9pt;
            }
        """)
        
        '''

    def emit(self, record: logging.LogRecord):
        """logging.Handler 的接口，运行在产生日志的线程"""
        msg = self.format(record)
        self.signaler.log_record.emit(msg)   # 异步发信号

    # -------------- 槽函数，运行在主线程 --------------
    def _append_plain_text(self, msg: str):
        self.widget.appendPlainText(msg)
        # 自动滚动到最底部
        # bar = self.widget.verticalScrollBar()
        # bar.setValue(bar.maximum())

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.resize(1200, 700)
        # 主窗口


        # ---------- 常量 ----------
        self.btn_disenable_stylesheet = """
        QPushButton {
            text-align:middle;
            background-color: gray;
            color: white;
            border-bottom: 1px solid #34495e;
            font-size: 13px;
        }
        QPushButton:hover {
            background-color: #34495e;
        }
        QPushButton:pressed {
            background-color: #b0b0b0;
        } 
        """
        self.btn_enable_stylesheet = """
        QPushButton {
            text-align:middle;
            background-color: rgb(49, 50, 44);
            color: white;
            border-bottom: 1px solid #34495e;
            font-size: 13px;
        }
        QPushButton:hover {
            background-color: #34495e;
        }
        QPushButton:pressed {
            background-color: #b0b0b0;
        }                          
        """

        # ---------- 变量 ----------
        self.camera_index = 0
        self.cap   = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.iou_timer = QTimer()
        self.iou_timer.setSingleShot(True)
        self.iou_timer.timeout.connect(self._really_log_iou)
        self.conf_timer = QTimer()
        self.conf_timer.setSingleShot(True)
        self.conf_timer.timeout.connect(self._really_log_conf)
        self.delay_time = 500

        self.video_play = None           # 导入即开始播放, 但要兼容图片，所以是None
        self.pt_loaded  = True           # 是否加载了权重
        self.is_inputed = False          # 是否输入了图像or视频 
        self.detection_running = False   # 是否正在检测
        self.detected   = False          # 是否已经检测
        self.now = datetime.now()
        self.image_path = None

        # 默认权重
        default_weight = "weights/best.pt"
        self.model = YOLOv5Model(default_weight)
        self.model.iou_thres = self.iou_slider.value()/100.0
        self.model.conf_thres = self.conf_slider.value()/100.0
        # self.model.conf_thres = self.conf_slider.value() 会导致没有结果
        # 可以用 self.iou_slider.value() 注意value本身是 0-100  因为只能是整数
        # 可以用 doublespinbox 是小数，在ui.py里的设置好了，

        # ---------- 信号连接槽函数 ----------
        # 菜单栏信号与槽函数的连接
        self.control_menu.triggered.connect(self.action_triggered)
        self.swift_lang.triggered.connect(self.swift_lang_def)  
        # 输入图片视频
        self.btn_image.clicked.connect(self.select_image)
        self.btn_video.clicked.connect(self.select_video)
        self.btn_camera.clicked.connect(self.open_camera)
        # 相机索引
        self.combobox_cam.currentIndexChanged.connect(self.cam_change)
        self.btn_video_end.clicked.connect(self.stop_play)
        # 加载权重与结束
        self.btn_pause_video.clicked.connect(self.pause_play)
        self.btn_load_pt.clicked.connect(self.load_pt)
        # 阈值设置 slider 和 text 数据互变同步
        self.iou_slider.valueChanged.connect(self.iou_slider_changed)
        self.iou_spinbox.valueChanged.connect(self.iou_spinbox_changed)
        self.conf_slider.valueChanged.connect(self.conf_slider_changed)
        self.conf_spinbox.valueChanged.connect(self.conf_spinbox_changed)
        
        # 新增的检测按钮信号
        self.btn_start_detect.clicked.connect(self.start_detection)
        self.btn_pause_detect.clicked.connect(self.pause_detection)
        self.btn_save.clicked.connect(self.save_result)

        # 必须在控件创建好之后再初始化 logging
        self.init_logging(self.plaintext)

        logging.info("日志初始化完成")
    
    # ---------- 菜单栏方法实现 ----------
    # 菜单栏的槽函数
    def action_triggered(self, action):  
        if action == self.clear_terminal: 
            ... 
            return
        elif action == self.clear_loaded:
            ...
            return
        elif action == self.logs_dir:
            '''
            
            filename, filter = QFileDialog.getOpenFileName(self, "打开", ".", "文本文件(*.txt)")

            try:
                fp = open(filename, encoding="UTF-8")
                string = fp.readlines()
                self.plainText.clear()
                for i in string:
                    self.plainText.appendPlainText(i)
                fp.close()

            except:
                QMessageBox.critical(self, "打开文件失败", "请选择适合的文件!")
            '''  
            return
        elif action == self.results_dir:
            ...
            return
        elif action == self.quit:  # 退出动作
            self.close()

    def swift_lang_def(self):
        print("swift not yet")
        ...
        return
        
    # ---------- 界面终端显示 ----------
    # 初始化日志
    def init_logging(self, plaintext: QPlainTextEdit):
        # 不写入参数，则 name 为 root, 写入"detect"则不写入, 获取指定名称的日志记录器,不存在"detect"
        root = logging.getLogger()  
        root.setLevel(logging.DEBUG)

        # 统一格式
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"            
        )

        # 1) 控制台
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(fmt)
        root.addHandler(console)

        # 2) 文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 单独定义日志目录路径
        logs_dir = os.path.join(current_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)     
        log_filepath = os.path.join(logs_dir,datetime.now().strftime("detect_%Y%m%d.log"))
        file_handler = logging.FileHandler(
            #datetime.now().strftime("detect_%Y%m%d.log"),  # 日志文件名 detect_ 随着日期变化
            log_filepath,
            encoding="utf-8"
        )  

        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

        # 3) GUI 控件
        gui_handler = QPlainTextEditHandler(plaintext)
        gui_handler.setLevel(logging.INFO)
        gui_handler.setFormatter(fmt)
        root.addHandler(gui_handler)
    '''
    def show_results(self, results_str):
        if results_str is not None:
            self.plaintext.appendPlainText(results_str) # PlainText自带滑条，默认按需显示    
    '''
    
    # ---------- 选择文件/摄像头 ----------
    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片 (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)")
        if path:
            self.image_path = path  # 用于predict
            self.video_play = None  # 这是为了兼容图片，图片则暂停播放不响应
            self.is_inputed = True
            logging.info(f"输入文件 {path}")
            #self.show_results(f"输入文件{path}" + f" ---{self.now:%Y/%m/%d %H:%M}---")
            #print(f'输入文件{path}')
            self.path_line.setText(path)
            self.btn_video_end.setEnabled(True)
            self.btn_video_end.setStyleSheet(self.btn_enable_stylesheet)
            if self.pt_loaded == True:
                self.btn_start_detect.setEnabled(True)
                self.btn_start_detect.setStyleSheet(self.btn_enable_stylesheet)

            
            # 用于展示图片
            image = QImage(path)
            self.label_img.setPixmap(QPixmap.fromImage(image).scaled(640,480))
            self.scaleFactor = 1.0
            
            #img = cv2.imread(path)
            #self.show_cv_img(img)

    # 效果： 加载视频会自动播放 用的Qtimer 
    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv)")
        if path:
            logging.info(f"输入文件 {path}")
            self.video_play = True
            self.is_inputed = True 
            self.open_source(path)
            #self.show_results(f"输入文件{path}" + f" ---{self.now:%Y/%m/%d %H:%M}---")
            #print(f'输入文件{path}')
            self.path_line.setText(path)
            self.btn_pause_video.setEnabled(True)  # 启用 播放/暂停 键
            self.btn_pause_video.setStyleSheet(self.btn_enable_stylesheet)
            self.btn_video_end.setEnabled(True)
            self.btn_video_end.setStyleSheet(self.btn_enable_stylesheet)
            if self.pt_loaded == True:
                self.btn_start_detect.setEnabled(True)
                self.btn_start_detect.setStyleSheet(self.btn_enable_stylesheet)

    def cam_change(self,index):
        self.camera_index = index

    def open_camera(self):
        #self.stop_play()
        # 先清缓存
        if self.cap:
            self.cap.release(); self.cap = None

        if self.camera_index == 0:
            self.open_source(self.camera_index)
            logging.warning("默认启动电脑内置相机，相机索引0，无实际检测价值")
            #self.show_results("电脑内置相机0，无实际检测价值" + f" ---{self.now:%Y/%m/%d %H:%M}---")

        else:
            self.open_source(self.camera_index)
        
        self.path_line.setText(str(self.camera_index))


    def open_source(self, src):
        self.timer.stop()
        #self.stop_play()

        # 先清除缓存
        if self.cap:
            self.cap.release(); self.cap = None

        self.detection_running = False

        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开视频/摄像头")
            logging.warning(f"无法打开相机，相机索引{self.camera_index}，相机是否已连接")
            return
        
        self.btn_video_end.setEnabled(True)
        self.btn_video_end.setStyleSheet(self.btn_enable_stylesheet)
        self.path_line.setText(str(src))
        self.timer.start(30)
    
    # 实现暂停播放：注意对状态 video_play 进行改变 共几次？ 是每次
    def pause_play(self):
        if self.video_play is None:  # 这是为了兼容图片，图片则暂停播放不响应
            return
        if self.video_play == True:
            self.timer.stop()     # 实现暂停
            self.video_play = False
            logging.info("暂停")
            #self.show_results("暂停"+ f" ---{self.now:%Y/%m/%d %H:%M}---")
        else:
            self.timer.start(30)  # 实现播放
            self.video_play = True
            logging.info("播放")
            #self.show_results("播放"+ f" ---{self.now:%Y/%m/%d %H:%M}---")


    def stop_play(self):
        self.timer.stop()
        if self.cap:
            self.cap.release(); self.cap = None
        # 注意对video_play 状态改变
        self.video_play = None
        self.detection_running = False
        self.label_img.setText("加载图像或视频后显示")
        # 按钮变化
        self.btn_pause_video.setEnabled(False)
        self.btn_pause_video.setStyleSheet(self.btn_disenable_stylesheet)
        self.btn_video_end.setEnabled(False)
        self.btn_video_end.setStyleSheet(self.btn_disenable_stylesheet)
        self.is_inputed = False
        self.btn_start_detect.setEnabled(False)
        self.btn_start_detect.setStyleSheet(self.btn_disenable_stylesheet)
        self.btn_pause_detect.setEnabled(False)
        self.btn_pause_detect.setStyleSheet(self.btn_disenable_stylesheet)
        # 输入路径清除
        self.path_line.setText("")
        logging.info("结束, 图像或视频已经退出")
        #self.show_results("结束"+ f" ---{self.now:%Y/%m/%d %H:%M}---")

    def closeEvent(self, event):
        if self.cap is not None:  # 先检查是否读取视频，否则退出时报错
            self.cap.release()
        super().closeEvent(event)


    # ---------- 权重切换与参数设置 ----------
    def load_pt(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择权重文件", "", "pt(*.pt)")
        if path:
            self.model.weights_path = path 
            self.pt_loaded = True
            #self.show_results(f'载入权重{path}'+f' ---{self.now:%Y/%m/%d %H:%M}---')
            #print(f'载入权重{path}')
            self.pt_line.setText(path)
            self.pt_line.setStyleSheet("color:black")
            if self.is_inputed == True:
                self.btn_start_detect.setEnabled(True)
                self.btn_start_detect.setStyleSheet(self.btn_enable_stylesheet) 
            logging.info(f"切换权重 {path}")
    # iou 滑块值与spinbox 互变统一

    def iou_slider_changed(self, value):
        iou = value/100.0
        self.iou_spinbox.setValue(iou)
        #logging.info(f"IOU阈值已变更为{iou}") 
        #self.iou_threshold = iou 没必要的 self.iou_spinbox.value()可以获得iou值, conf同理
        self.iou_timer.start(self.delay_time)  # 每次改变都重启计时器，实现延时logging
    def iou_spinbox_changed(self, value):
        iou = int(value*100)
        self.iou_slider.setValue(iou)
        #logging.info(f"IOU阈值已变更为{iou}")
        #self.iou_threshold = iou
        self.iou_timer.start(self.delay_time)
    # 日志延迟记录调整过程中最后一个值，delay_time = 500ms
    def _really_log_iou(self):  
        self.model.ciou_thres = self.iou_spinbox.value()
        logging.info(f" I o U 阈值已变更为{self.model.iou_thres:.2f}")
        if self.video_play is  None and self.is_inputed:
            self.btn_start_detect.setEnabled(True)
            self.btn_start_detect.setStyleSheet(self.btn_enable_stylesheet)

    # conf 置信度 滑块值与spinbox 互变统一
    def conf_slider_changed(self, value):
        conf = value/100.0
        self.conf_spinbox.setValue(conf)    
        #self.conf_threshold = conf
        self.conf_timer.start(self.delay_time)

    def conf_spinbox_changed(self, value):
        conf = int(value*100)
        self.conf_slider.setValue(conf)
        #logging.info(f"置信度阈值已变更为{conf}")
        #self.conf_threshold = conf
        self.conf_timer.start(self.delay_time)

    def _really_log_conf(self):  
        # conf = self.conf_spinbox.value()  这不一定是实际置信度
        self.model.conf_thres = self.conf_spinbox.value()
        logging.info(f"置信度阈值已变更为{self.model.conf_thres:.2f}")
        if self.video_play is  None and self.is_inputed:
            self.btn_start_detect.setEnabled(True)
            self.btn_start_detect.setStyleSheet(self.btn_enable_stylesheet)

    ''' logging更新太快，有冗余，弃用
    # iou 滑块值与spinbox 互变统一

    def iou_slider_changed(self, value):
        iou = value/100.0
        self.iou_spinbox.setValue(iou)
        logging.info(f"IOU阈值已变更为{iou}")
        self.iou_threshold = iou
    def iou_spinbox_changed(self, value):
        iou = int(value*100)
        self.iou_slider.setValue(iou)
        logging.info(f"IOU阈值已变更为{iou}")
        self.iou_threshold = iou

    # conf 置信度 滑块值与spinbox 互变统一
    def conf_slider_changed(self, value):
        conf = value/100.0
        self.conf_spinbox.setValue(conf)    
        logging.info(f"置信度阈值已变更为{conf}")
        self.conf_threshold = conf
    def conf_spinbox_changed(self, value):
        conf = int(value*100)
        self.conf_slider.setValue(conf)
        logging.info(f"置信度阈值已变更为{conf}")
        self.conf_threshold = conf
    '''        
    

    # ---------- 检测控制 ----------
    @Slot()
    def start_detection(self):
        """开始/继续检测"""
        #if not self.cap or not self.cap.isOpened():
        '''
        # 通过 disenabled btn 已经规避，所以弃用
        if self.is_inputed == False:
            QMessageBox.warning(self, "提示", "请先选择视频/摄像头")
            return
        '''
        if self.video_play is None and self.is_inputed:
            img = cv2.imread(self.image_path)  # 注意读取图片 而不是输入path给model
            img_out = self.model.predict(img)  # .predict 就是能否视频推理的关键
            self.show_cv_img(img_out)


        else:
            ret, frame = self.cap.read()
            if not ret:
                self.stop_play(); return            
            if self.detection_running:
                # 调用 YOLOv5 模型进行推理
                self.model.conf_thres= self.conf_spinbox.value()
                self.model.iou_thres= self.iou_spinbox.value()
                frame = self.model.predict(frame) 
                self.show_cv_img(frame) 

                pass
        self.detection_running = True
        self.detected   = True
        self.btn_save.setEnabled(True)
        self.btn_save.setStyleSheet(self.btn_enable_stylesheet)
        self.btn_start_detect.setEnabled(False)
        self.btn_start_detect.setStyleSheet(self.btn_disenable_stylesheet)
        if self.video_play is not None:
            self.btn_pause_detect.setEnabled(True)
            self.btn_pause_detect.setStyleSheet(self.btn_enable_stylesheet)
        logging.info("启动检测")
        #self.show_results("[INFO] 开始检测（YOLOv5 逻辑待接入）")
        #print("[INFO] 开始检测（YOLOv5 逻辑待接入）")


    @Slot()
    def pause_detection(self):
        """暂停检测（视频继续播放，但推理暂停）"""
        self.detection_running = False
        logging.info( "暂停检测(逻辑待接入)")
        #self.show_results("[INFO] 暂停检测")
        #print("[INFO] 暂停检测")
        self.btn_start_detect.setEnabled(True)
        self.btn_start_detect.setStyleSheet(self.btn_enable_stylesheet)
        self.btn_pause_detect.setEnabled(False)
        self.btn_pause_detect.setStyleSheet(self.btn_disenable_stylesheet)

    @Slot()
    def save_result(self):
        """保存当前帧或整个结果（占位）"""
        logging.info("保存结果（逻辑待实现）")
        #self.show_results("[INFO] 保存结果（逻辑待实现）")
        #print("[INFO] 保存结果（逻辑待实现）")
        #QMessageBox.information(self, "提示", "保存结果功能待接入")

    # ---------- 显示 ----------
    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_play(); return
        if self.detection_running:
            # 调用 YOLOv5 模型进行推理
            self.model.conf_thres= self.conf_spinbox.value()
            self.model.iou_thres= self.iou_spinbox.value()
            frame = self.model.predict(frame)
        self.show_cv_img(frame)


    def show_cv_img(self, cv_img):
        if cv_img is None: 
            return
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.label_img.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.label_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.label_img.pixmap():
            self.label_img.setPixmap(self.label_img.pixmap().scaled(
                self.label_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        
        ...


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
