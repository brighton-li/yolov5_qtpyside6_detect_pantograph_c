from datetime import datetime
from PySide6.QtWidgets import (QMainWindow,QWidget, QPushButton, QLabel, QLineEdit,
    QSlider,QPlainTextEdit, QVBoxLayout, QHBoxLayout,QFrame,QFormLayout,QDoubleSpinBox,
    QComboBox,QTabWidget,QSplitter,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator 


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # 设置窗口标题和大小
        MainWindow.setWindowTitle("弓网目标检测 by brighton-li")
        MainWindow.setGeometry(150, 50, 1100, 500)  # 初始左上角位置&大小 x，y,w,h

        # 菜单栏，状态栏 直接用Qmainwindows 的，不要放入布局
        menubar = self.menuBar()          # 直接拿 QMainWindow 的菜单栏
        self.control_menu = menubar.addMenu("控制")
        self.clear_loaded = self.control_menu.addAction("清空加载")
        self.clear_terminal = self.control_menu.addAction("清空终端")
        self.logs_dir = self.control_menu.addAction("日志文件夹")
        self.results_dir = self.control_menu.addAction("结果文件夹")
        self.control_menu.addSeparator()
        self.quit = self.control_menu.addAction("退出")
        self.swift_lang = menubar.addAction("切换语言")
        menubar.setFixedHeight(25)


        statusbar = self.statusBar()
        self.date = datetime.now().strftime("%m")
        statusbar.addPermanentWidget(QLabel("V1.0." + self.date))
        statusbar.showMessage("已就绪",5000)
        statusbar.setFixedHeight(15)

        # 中心
        central = QWidget(); 
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # 左侧 选择文件 Frame 容器
        self.left_menu = QFrame()
        self.left_menu.setFixedSize(150,270)
        self.left_menu.setStyleSheet("""
        QFrame {
            background-color: rgb(230, 206, 140);
            border-right: 1px solid #528279;
            border: 1px solid white;
            border-radius: 5px;
        }
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
        """)
        
        # 左侧 设置 frame 容器
        self.setting_menu = QFrame()
        self.setting_menu.setMinimumSize(150,220)
        self.setting_menu.setStyleSheet("""
        QFrame {
            background-color: rgb(75, 196, 188);
            border-right: 1px solid #528279;
            border: 1px solid white;
            border-radius: 2px;
        }
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

        QLabel {
            color: #333333;
            font-size: 13px;
            padding: 4px 8px;
            background-color: transparent;
            border: none; /* 完全移除边框 */
            margin: 0px; /* 可选：移除外边距 */
        }                        
        """)

        # 右侧检测控制 frame 容器
        self.detect_menu = QFrame()
        self.detect_menu.setFixedSize(1000,35)
        self.detect_menu.setStyleSheet("""
        QFrame {
            background-color: rgb(230, 206, 140);
            border-right: 1px solid #528279;
            border: 1px solid white;
            border-radius: 2px;                           
        }
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
        """)

        # 文件/摄像头
        self.btn_image  = QPushButton("输入图片")
        self.btn_image.setFixedSize(100, 40)
        self.btn_video  = QPushButton("输入视频")
        self.btn_video.setFixedSize(100, 40)
        self.btn_camera = QPushButton("连接相机")
        self.btn_camera.setToolTip("默认打开电脑相机")
        self.btn_camera.setFixedSize(100, 40)

        self.combobox_cam = QComboBox(self,fixedWidth = 40)
        self.combobox_cam.setEditable(True)
        self.combobox_cam.addItems(["0","1","2"])  # 下拉显示
        int_validator = QIntValidator(self)
        int_validator.setRange(0,9)                # 允许填入的
        self.combobox_cam.setValidator(int_validator)
        self.cam_label = QLabel("相机索引",maximumWidth = 50)
        self.cam_label.setStyleSheet("border: none;")

        self.btn_pause_video = QPushButton("播放/暂停")
        self.btn_pause_video.setFixedSize(100, 40)
        self.btn_pause_video.setEnabled(False)  # 默认禁用
        self.btn_pause_video.setStyleSheet("background-color:gray; color:white;")

        self.btn_video_end  = QPushButton("结束stop")
        self.btn_video_end.setFixedSize(100, 40)
        self.btn_video_end.setEnabled(False)  # 默认禁用
        self.btn_video_end.setStyleSheet("background-color:gray; color:white;")
        # 输入路径
        self.path_line  = QLineEdit(); 
        self.path_line.setReadOnly(True)
        self.path_line.setToolTip('只读')

        # 一个图标 弓网俯视图
        self.my_icon = QLabel("to be continue")
        self.my_icon.setAlignment(Qt.AlignCenter)
        self.my_icon.setMinimumSize(150,120)
        self.my_icon.setMaximumSize(150,150)
        self.my_icon.setStyleSheet("background-color:white; color:grey;")

        '''
        
        # 语言切换
        self.btn_switch_lang = QPushButton('中/En')
        self.btn_switch_lang.setToolTip('中英切换')
        self.btn_switch_lang.setMinimumSize(80,30)
        self.btn_switch_lang.setStyleSheet("""
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
        }"""
        )
        
        '''
        # 权重加载
        self.btn_load_pt = QPushButton('切换权重')
        self.btn_load_pt.setMinimumSize(80,30)
        self.btn_load_pt.setStyleSheet("""
        QPushButton {
            text-align:middle;
            background-color: rgb(90, 200, 190);
            color: black;
            font-size: 13px;
            border-bottom: 1px solid #34495e;
        }
        QPushButton:hover {
            background-color: rgb(142, 240, 224);
        }
        QPushButton:pressed {
            background-color: black;
        }"""
        )
        # 权重路径
        self.pt_line  = QLineEdit("已载入默认权重, 可直接测试")
        self.pt_line.setReadOnly(True)
        self.pt_line.setToolTip('只读')
        self.pt_line.setStyleSheet("color: gray")


        # 参数设置
        self.label_setting = QLabel("阈值设置")
        self.label_setting.setAlignment(Qt.AlignCenter)  # 文本居中对齐
        self.label_setting.setFixedSize(80, 40)
        self.label_setting.setStyleSheet("color: black; font : 12px; ")
        
        # 置信度阈值
        self.conf_label = QLabel("置信度")       
        # 添加滑动块
        self.conf_slider = QSlider(Qt.Vertical)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(25) # 默认值
        self.conf_slider.setTickInterval(1)
        self.conf_slider.setContentsMargins(0, 0, 0, 0)        
        self.conf_slider.setToolTip("推荐值0.25")
        # 添加可调整数值输入框
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setDecimals(2)
        self.conf_spinbox.setRange(0.0, 1.0)
        self.conf_spinbox.setSingleStep(0.01)  # 设置步长为0.01
        self.conf_spinbox.setMaximumWidth(50)  # 设置外形最大宽度
        self.conf_spinbox.setValue(0.25)        # 默认值
        self.conf_spinbox.setToolTip("推荐值0.25")
        # iou 阈值
        self.iou_label = QLabel("IoU")
        # self.iou_label.setStyleShet("QLabel { color: white; }")        
        # 添加滑动块
        # QSlider 的刻度单位是 整数
        self.iou_slider = QSlider(Qt.Vertical)
        self.iou_slider.setMinimum(0)
        self.iou_slider.setMaximum(100)
        self.iou_slider.setValue(45) # 默认值
        self.iou_slider.setTickInterval(1)
        self.iou_slider.setToolTip("推荐值0.45")        
        # 添加可调整数值输入框
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setDecimals(2)
        self.iou_spinbox.setRange(0.0, 1.0)
        self.iou_spinbox.setSingleStep(0.01)  # 设置步长为0.01
        self.iou_spinbox.setMaximumWidth(50)  # 设置外形最大宽度
        self.iou_spinbox.setValue(0.45)          # 默认值
        self.iou_spinbox.setToolTip("推荐值0.45")

        # 图像显示
        self.label_img = QLabel("加载图像或视频后显示")
        self.label_img.setAlignment(Qt.AlignCenter)
        self.label_img.setMinimumSize(640,480)
        self.label_img.setStyleSheet("background-color:black; color:white;")

        # 曲线所在
        self.tabWidget = QTabWidget(self)
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tabWidget.addTab(self.tab1,"输出曲线")
        self.tabWidget.addTab(self.tab2,"位置列表")
        self.tabWidget.setMinimumSize(200,480)
        self.tabWidget.setStyleSheet("""
        QTabBar {
            text-align:middle;
            background-color: gray;
            color: black;                              
        }
        QTabBar:hover {
            background-color: #86919b;                       
        }
        QTabBar:pressed {
            background-color: gray;                                     
        }                                   
        """)
        
        # 检测控制+保存
        self.btn_start_detect = QPushButton("启动检测")
        self.btn_start_detect.setFixedSize(100,20)
        self.btn_start_detect.setEnabled(False)     # 默认禁用
        self.btn_pause_detect = QPushButton("暂停检测")
        self.btn_pause_detect.setEnabled(False)     # 默认禁用
        self.btn_pause_detect.setFixedSize(100,20)
        self.btn_save  = QPushButton("保存结果")
        self.btn_save.setFixedSize(100,20)
        self.btn_save.setEnabled(False)             # 默认禁用

        # 创建文本框-终端执行情况
        self.plaintext = QPlainTextEdit("执行反馈日志-Results show")
        self.plaintext.setMinimumSize(1100, 40)
        self.plaintext.setReadOnly(True)  # 设置文本框为只读
        # 设置文本框-终端的CSS样式
        self.plaintext.setStyleSheet("""
            QPlainTextEdit {
                background-color: white; /* 浅蓝色 */
                border: 1px solid white; /* 边框颜色白色 */
                border-radius: 10px; /* 设置边框圆角 */
                padding: 5px; /* 添加内边距 */
                font-size: 13px;
                color : gray
            }
        """)
        # 滑条自带 QPlainTextEdit / QTextEdit 这里始终显示, 默认按需出现
        self.plaintext.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)   

        # ---------- 布局与添加控件 ----------
        # 创建布局
        left_layout   = QVBoxLayout()
        #right_layout  = QVBoxLayout()
        right_layout  = QSplitter(Qt.Vertical)
        combob_layout = QHBoxLayout() # 实现 combobox 加标签 居中对齐
        #input_layout  = QFormLayout() # 快捷实现 标签+文本
        iou_layout    = QVBoxLayout()
        conf_layout   = QVBoxLayout()

        #right_upper_layout = QHBoxLayout()
        right_upper_layout = QSplitter(Qt.Horizontal)


        # 设置frame容器加布局
        left_menu_layout = QVBoxLayout(self.left_menu)
        setting_menu_layout = QHBoxLayout(self.setting_menu)

        # --- 左侧布局控件 ---
        # 添加 控件到 选择文件 frame容器
        left_menu_layout.setSpacing(12)  # 设置布局中部件之间的间距为 9 像素
        left_menu_layout.addWidget(self.btn_image,alignment=Qt.AlignCenter)
        left_menu_layout.addWidget(self.btn_video,alignment=Qt.AlignCenter)
        left_menu_layout.addWidget(self.btn_camera,alignment=Qt.AlignCenter)
        left_menu_layout.addLayout(combob_layout)
        combob_layout.addWidget(self.cam_label, )
        combob_layout.addWidget(self.combobox_cam)
        combob_layout.setAlignment(Qt.AlignCenter)
        left_menu_layout.addWidget(self.btn_pause_video,alignment=Qt.AlignCenter)
        left_menu_layout.addWidget(self.btn_video_end,alignment=Qt.AlignCenter)
        left_menu_layout.addStretch(5)  # 添加伸缩项，将按钮推到顶部
        # 添加 选择文件 frame容器
        left_layout.addWidget(self.left_menu)


        # 添加 自定义图标
        left_layout.addWidget(self.my_icon)
        left_layout.addStretch(1)  # 布局添加伸缩项，位置不同效果不同


        # 添加 控件到 设置菜单 frame容器
        # 添加 置信度 阈值 到 conf布局 
        conf_layout.addWidget(self.conf_label, alignment= Qt.AlignCenter)
        conf_layout.addWidget(self.conf_slider, alignment= Qt.AlignCenter)
        conf_layout.addWidget(self.conf_spinbox, alignment= Qt.AlignCenter)
        # 添加 iou 阈值 到 iou布局 
        iou_layout.addWidget(self.iou_label, alignment= Qt.AlignCenter)
        iou_layout.addWidget(self.iou_slider, alignment= Qt.AlignCenter)
        iou_layout.addWidget(self.iou_spinbox, alignment= Qt.AlignCenter)
        # 添加 设置菜单 frame容器
        setting_menu_layout.addLayout(conf_layout)
        setting_menu_layout.addLayout(iou_layout)

        # 所有元素加入左布局
        #left_layout.addWidget(self.btn_switch_lang)
        left_layout.addWidget(self.btn_load_pt)
        left_layout.addWidget(self.label_setting)
        left_layout.addWidget(self.setting_menu,alignment=Qt.AlignHCenter)


        # --- 右侧布局控件 ---
        right_upper_layout.addWidget(self.label_img)
        right_upper_layout.addWidget(self.tabWidget)
        right_layout.addWidget(right_upper_layout)
        #right_layout.addLayout(right_upper_layout)
        #right_layout.addWidget(self.label_img)

        #form布局+容器
        input_widget = QWidget()
        input_layout = QFormLayout(input_widget)
        # 输入文件路径
        input_layout.addRow("输入路径：",self.path_line)
        # 权重文件路径
        input_layout.addRow("权重路径：",self.pt_line)
        right_layout.addWidget(input_widget)

        # 检测控制按钮
        detect_layout = QHBoxLayout(self.detect_menu)
        detect_layout.addWidget(self.btn_start_detect)
        detect_layout.addWidget(self.btn_pause_detect)
        detect_layout.addWidget(self.btn_save)
        # 添加 检测控制 frame容器
        right_layout.addWidget(self.detect_menu)

        # 终端执行文本
        right_layout.addWidget(self.plaintext)
        #right_layout.setSpacing(5)  # 设置布局中部件之间的间距为 5 像素
        
        # 添加布局到主布局
        main_layout.addLayout(left_layout)
        main_layout.addWidget(right_layout)

