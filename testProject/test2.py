import torch
import cv2
import mediapipe as mp
from torch.utils.data import Dataset, DataLoader
from skeletton_LSTM import skeleton_LSTM
import numpy as np
import time  # 타이머를 위해 추가
import warnings


class MyDataset(Dataset):
    def __init__(self, seq_list):
        self.X = []
        self.y = []
        for dic in seq_list:
            self.y.append(dic['key'])
            self.X.append(dic['value'])

    def __getitem__(self, index):
        data = self.X[index]
        label = self.y[index]
        return torch.Tensor(np.array(data)), torch.tensor(np.array(int(label)))

    def __len__(self):
        return len(self.X)


warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
# YOLOv5 모델 불러오기
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                            device='cuda:0' if torch.cuda.is_available() else 'cpu')
yolo_model.classes = [0]  # 사람만 감지

# 설정 변수
n_CONFIDENCE = 0.3
y_CONFIDENCE = 0.3
start_dot = 11
attention_dot = [n for n in range(start_dot, 29)]
draw_line = [[11, 13], [13, 15], [15, 21], [15, 19], [15, 17], [17, 19],
             [12, 14], [14, 16], [16, 22], [16, 20], [16, 18], [18, 20],
             [23, 25], [25, 27], [24, 26], [26, 28], [11, 12], [11, 23],
             [23, 24], [12, 24]]

# LSTM 모델 불러오기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = skeleton_LSTM().to(device)
model_name = 'LSTM57.pt'
net.load_state_dict(torch.load(model_name))
net.eval()

# 절도 감지 상태 변수
theft_detected = False
theft_detected_time = 0


def process_frame_sequence(frames, abnormal_dir):
    global theft_detected, theft_detected_time
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False,
                                  min_detection_confidence=n_CONFIDENCE)
    xy_list_list = []
    status = 'Normal'

    # # 절도 감지 후 10초가 지나지 않았으면 모든 프레임을 절도로 처리
    # if theft_detected and (time.time() - theft_detected_time) < 10:
    #     status = 'Theft'
    #     return status, frames

    for img in frames:
        # YOLOv5로 사람 감지
        res = yolo_model(img)
        res_refine = res.pandas().xyxy[0].values
        nms_human = len(res_refine)
        if nms_human > 0:
            for bbox in res_refine:
                xx1, yy1, xx2, yy2 = int(bbox[0]) - 10, int(bbox[1]), int(bbox[2]) + 10, int(bbox[3])
                if xx1 < 0: xx1 = 0
                if xx2 > 639: xx2 = 639
                if yy1 < 0: yy1 = 0
                if yy2 > 639: yy2 = 639

                # 사람을 인식한 후 히트박스 그리기
                cv2.rectangle(img, (xx1, yy1), (xx2, yy2), (0, 0, 255), 2)
                cv2.putText(img, 'AbNormal', (xx1, yy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                c_img = img[yy1:yy2, xx1:xx2]
                results = pose.process(cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB))
                if not results.pose_landmarks: continue
                xy_list = []
                draw_line_dic = {}
                for idx, x_and_y in enumerate(results.pose_landmarks.landmark):
                    if idx in attention_dot:
                        xy_list.append(x_and_y.x)
                        xy_list.append(x_and_y.y)
                        draw_line_dic[idx] = [int(x_and_y.x * (xx2 - xx1)), int(x_and_y.y * (yy2 - yy1))]

                xy_list_list.append(xy_list)
                for line in draw_line:
                    x1, y1 = draw_line_dic[line[0]][0], draw_line_dic[line[0]][1]
                    x2, y2 = draw_line_dic[line[1]][0], draw_line_dic[line[1]][1]
                    c_img = cv2.line(c_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    if len(xy_list_list) == 30:
        dataset = [{'key': 0, 'value': xy_list_list}]
        dataset = MyDataset(dataset)
        dataset = DataLoader(dataset)
        xy_list_list = []

        for data, label in dataset:
            data = data.to(device)
            with torch.no_grad():
                result = net(data)
                _, out = torch.max(result, 1)
                if out.item() == 1:  # 'Theft'가 감지된 경우
                    status = 'Theft'
                    theft_detected = True
                    theft_detected_time = time.time()
                    break  # 상태를 "Theft"로 설정
    print(status)
    return status, frames
