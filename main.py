#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import datetime
import math
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend

backend.clear_session()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default="filmage001.avi")
ap.add_argument("-c", "--class", help="name of class", default="car")
args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
haveCountedCar = []  # 已经被计数过的车辆
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
                           dtype="uint8")


def main(yolo):
    start = time.time()
    # Definition of the parameters
    max_cosine_distance = 0.5  # 余弦距离的控制阈值
    nn_budget = None
    nms_max_overlap = 0.3  # 非极大抑制的阈值
    count = 0
    class_name = ''
    counter = []

    # 使用km/s还是m/s
    kms = 1
    # deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    # video_path = "./output/output.avi"
    video_capture = cv2.VideoCapture(args["input"])

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # 将视频和检测目标输出
        out = cv2.VideoWriter('./output/' + args["input"][43:57] + "_" + args["class"] + '_output.avi', fourcc, 15,
                              (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0

    inCount = 0
    outCount = 0

    pixelPerReal = 53.126  # 像素：现实（m）
    frameIndex = 0
    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()
        virtualLine = int(0.5 * frame.shape[1])  # 虚拟线位置
        cv2.line(frame, (virtualLine, 0), (virtualLine, frame.shape[0]), (255, 0, 0), 3)

        frameIndex += 1  # frame的帧数

        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, class_names = yolo.detect_image(image)  # 获取当前帧上的bounding box和种类

        features = encoder(frame, boxs)
        # score to 1.0 here).

        # 1.获取每帧的Detections
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        # 2.NMS
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        # 3.对每一个track进行预测
        tracker.predict()
        # 4.update：进行匹配match，根据两种不同的tracks选择不同的处理方式，confirmed track进行级联匹配，其他的使用IOU匹配；
        #           更新track，根据关联的detection更新track（hit、update since等等），未关联的track也更新（可能deleted），根据detection创建新的track
        #           特征集更新，distance metric
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        c = []
        boxes = []
        for det in detections:
            bbox = det.to_tlbr()
            #   绘制detections的方框
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)

        for track in tracker.tracks:  # 每一个跟踪到的检测框
            # 只绘制已经确定的或者上一帧有出现的track
            # 开始可视化
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 3)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 150, (color), 2)
            if len(class_names) > 0:
                class_name = class_names[0]
                cv2.putText(frame, str(class_names[0]), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150, (color), 2)

            i += 1
            # bbox_center_point(x,y)

            # center的x坐标，center的y坐标，center所在的帧数index
            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2), frameIndex)
            # track_id[center]

            pts[track.track_id].append(center)

            print(center[0],center[1])
            thickness = 5
            # center point
            cv2.circle(frame, (center[0], center[1]), 1, color, thickness)

            # 做了一个车辆行驶方向的判断，还有是否经过虚拟线
            # 当前追踪的这个车辆id至少有超过8帧了
            # center:x,y,frameIndex
            # trackIndex = len(pts[track.track_id]) - 1
            if len(pts[track.track_id]) >= 8:
                lastIndex = len(pts[track.track_id]) - 8
                # while True:
                #     # 如果上一帧相差的帧数大于8
                #     if pts[track.track_id][lastIndex][2] <= center[2] - 8:
                #         break
                #     # 没有超过8帧的话就继续往前
                #     lastIndex -= 1
                #     if lastIndex < 0:
                #         break
                # # 如果始终没有8帧以外的 就直接结束 不计算了

                if lastIndex < 0:
                    break
                # 找到上一个center点了
                lastCenter = pts[track.track_id][lastIndex]

                # 行驶方向
                if center[1] > lastCenter[1]:
                    cv2.putText(frame, 'down', (int(bbox[0] + 40), int(bbox[1] - 40)), 0, 5e-3 * 150, (color), 2)
                else:
                    cv2.putText(frame, 'up', (int(bbox[0] + 40), int(bbox[1] - 40)), 0, 5e-3 * 150, (color), 2)

                # 车流量计算
                if track.track_id not in haveCountedCar:
                    if center[0] > virtualLine > lastCenter[0]:
                        inCount += 1
                        haveCountedCar.append(track.track_id)
                    else:
                        if center[0] < virtualLine < lastCenter[0]:
                            outCount += 1
                            haveCountedCar.append(track.track_id)

                # 车速
                dPixels = math.sqrt(pow(abs(center[0] - lastCenter[0]), 2) + pow(abs(center[1] - lastCenter[1]), 2))
                dFrame = center[2] - lastCenter[2]
                vCar = 1.0 * 24 * dPixels / pixelPerReal / dFrame

                if kms == 1:
                    cv2.putText(frame, str(int(vCar * 3.6)) + 'km/h', (int(bbox[0] + 100), int(bbox[1] - 40)), 0,
                                5e-3 * 250, (color),
                                2)
                else:
                    cv2.putText(frame, str(int(vCar)) + 'm/s', (int(bbox[0] + 100), int(bbox[1] - 40)), 0,
                                5e-3 * 250, (color),
                                2)

            # draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame, (pts[track.track_id][j - 1][0], pts[track.track_id][j - 1][1]),
                         (pts[track.track_id][j][0], pts[track.track_id][j][1]), (color), thickness)
                # cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

        count = len(set(counter))
        cv2.putText(frame, "Total Object Counter: " + str(count), (int(20), int(120)), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "Current Object Counter: " + str(i), (int(20), int(80)), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "In Car Counter: " + str(inCount), (int(20), int(200)), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "Out Car Counter: " + str(outCount), (int(20), int(250)), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "FPS: %f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)
        cv2.namedWindow("YOLO3_Deep_SORT", 0);
        cv2.resizeWindow('YOLO3_Deep_SORT', 2574, 1440);
        cv2.imshow('YOLO3_Deep_SORT', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(
                        str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
            list_file.write('\n')
        fps = (fps + (1. / (time.time() - t1))) / 2
        # print(set(counter))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(" ")
    print("[Finish]")
    end = time.time()

    if len(pts) != None:
        print(args["input"][43:57] + ": " + str(count) + " " + str(class_name) + ' Found')
    else:
        print("[No Found]")

    video_capture.release()

    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
