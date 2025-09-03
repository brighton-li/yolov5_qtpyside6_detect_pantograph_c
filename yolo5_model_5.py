import torch
import cv2
import numpy as np
import sys
import logging
from pathlib import Path

'''
SRC_ROOT = Path(__file__).parent / 'yolov5_cut'
sys.path.insert(0, str(SRC_ROOT))
'''

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

class YOLOv5Model:
    def __init__(self,
                 weights_path: str,
                 device: str = None,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45):
        # 自动检测设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # 打印设备信息
        if torch.cuda.is_available():
            logging.info(f"使用GPU设备: {self.device}")
            logging.info(f"GPU名称: {torch.cuda.get_device_name(self.device)}")
            logging.info(f"CUDA版本: {torch.version.cuda}")
            logging.info(f"cuDNN版本: {torch.backends.cudnn.version()}")
            logging.info(f"GPU内存: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.2f} GB")
        else:
            logging.info("未检测到GPU，使用CPU设备")

        # 1. 加载模型并移至指定设备
        self.model = attempt_load(weights_path, map_location=self.device)
        self.model.to(self.device)
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') \
                     else self.model.names
                      
        # 验证模型是否正确加载到指定设备
        model_device = next(self.model.parameters()).device
        logging.info(f"模型成功加载到设备: {model_device}")

        # 2. warmup
        self.model(torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(self.model.parameters())))
        logging.info("模型预热完成，准备进行推理")

    @torch.no_grad()
    def predict(self, img_bgr):
        """输入 OpenCV BGR，返回画好框的 BGR 和检测结果信息"""
        import time
        
        # 记录开始时间
        start_time = time.time()
        
        # 1. 前处理
        img = letterbox(img_bgr, 640, stride=self.stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB, HWC → CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # 验证输入张量是否在正确的设备上
        tensor_device = img.device
        # logging.debug(f"输入张量位于设备: {tensor_device}")

        # 2. 推理
        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        
        # 计算推理时间
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 显示GPU内存使用情况（仅在GPU上运行时）
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # 转换为MB
            gpu_memory_cached = torch.cuda.memory_reserved(self.device) / 1024**2  # 转换为MB
            # 只在某些帧打印详细信息，避免日志过多
            if hasattr(self, 'frame_count'):
                self.frame_count += 1
            else:
                self.frame_count = 1
            
            if self.frame_count % 10 == 0:
                logging.info(f"推理时间: {inference_time:.2f} ms, GPU内存已分配: {gpu_memory_allocated:.2f} MB, GPU内存已缓存: {gpu_memory_cached:.2f} MB")

        # 3. 后处理并画框
        det = pred[0]
        contact_points = []
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_bgr.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{self.names[int(cls)]} {conf:.2f}'
                self._plot_one_box(xyxy, img_bgr, label=label,
                                   color=(100, 160, 0), line_thickness=2)
                
                # 提取contact point的中心点坐标
                if self.names[int(cls)] == 'contact point':
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    y_center = (xyxy[1] + xyxy[3]) / 2
                    contact_points.append((float(x_center), float(y_center)))
        
        # 返回画好框的图像和contact point信息
        return img_bgr, contact_points

    @staticmethod
    def _plot_one_box(xyxy, img, color, label=None, line_thickness=3):
        tl = line_thickness or max(round(sum(img.shape[:2]) / 2 * 0.003), 2)
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                        (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)