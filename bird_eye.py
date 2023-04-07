import argparse
import cv2
import os

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import math

from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from track import create_polyline
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'


def transform_axes(my_array, ax, axes_size):
    # matrix axes scaller
    MIN = my_array[:, ax].min()
    MAX = my_array[:, ax].max()
    return (axes_size * ((my_array[:, ax] - MIN) / (MAX - MIN))).astype(int)
@torch.no_grad()
def run(
        source='0',
        imgsz=(640, 640),  # inference size (height, width)

        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        # project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
        track_lines_len=0,
        homography_source=False,
        source_homorgafhy=False,
        track_source=False,
        homography_output_file=False,
):

    with open(source_homorgafhy, 'r') as f:
        homography = f.read().replace('\n', '').split(',')
        homography_matrix = np.array(homography, dtype=float).reshape((3, 3))

    track_df = np.loadtxt(track_source, dtype=int) # frame_idx + 1, id, bbox_left, bbox_top, bbox_w, bbox_h, -1, -1, -1, i  MOT format
    frames_n_ids = np.array(track_df[:, :2], dtype=int)
    track_df_transform = np.ones_like(track_df[:, 2:5])
    track_df_transform[:, 0] = track_df[:, 2] + track_df[:, 4] / 2
    track_df_transform[:, 1] = track_df[:, 3] + track_df[:, 5] / 2

    track_df_transform = np.array([np.dot(homography_matrix, points, ) for points in track_df_transform])
    # track_df_transform = np.dot(track_df_transform, homography_matrix) # wrong trasformation
    track_df_transform[:, 0] = transform_axes(track_df_transform, 0, imgsz[0]) # здесь я могу путать ширину и высоту
    track_df_transform[:, 1] = transform_axes(track_df_transform, 1, imgsz[1]) # здесь я могу путать ширину и высоту

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Создаем новый видеофайл с помощью cv.VideoWriter
    out = cv2.VideoWriter(homography_output_file, fourcc, 30, imgsz)  # собираем видос

    unique_people_ids = set(frames_n_ids[:, 1])  # собираем уникальные идентификаторы
    # проходим по фреймам. Обрабите внимание, максимальный размер фрейма не равен размеру массива,
    # за счет нескольких объектов в кадре
    for i in range(1, frames_n_ids[-1, 0] - 1):
        img = np.zeros((imgsz[0], imgsz[1], 3), dtype=np.uint8)  # Создаем пустой кадр
        img.fill(255) # по умолчанию заполняем фон белым
        for my_unique_people_ids in unique_people_ids:  # проходим по унимальным персонам
            # применяем условия о текущем фрейме и применяем условие по уникальной персоне
            current_ind = (frames_n_ids[:, 0] <= i) & (frames_n_ids[:, 1] == my_unique_people_ids)

            # применяем условия для наличия персоны в кадре
            if my_unique_people_ids in frames_n_ids[frames_n_ids[:, 0] == i][:, 1]:
                points = [list(i.astype(int)) for i in track_df_transform[current_ind, :2]]  #

                if len(points) > 3:
                    polyline = create_polyline(points, track_lines_len)  #

                    for i1 in range(len(polyline) - 1, 0, -1):  #
                        cv2.line(img, polyline[i1], polyline[i1 - 1], (0, 0, 255), thickness=2)  # values by default
        out.write(img)

    # Закрываем видеофайл
    out.release()
    print('Results saved to {}'.format(homography_output_file))
    return 0

def parse_opt():
    ######################## add values for default for current code
    parser = argparse.ArgumentParser()
    # parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--source', type=str, default='./track/exp/tracks/001.avi', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--homography-source', type=str, help='sourse of homorraphy matrix')
    # parser.add_argument('--homography-source', type=str, default='./HallWayTracking/HallWayTracking/homography/001.txt', help='sourse of homorraphy matrix')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    # parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    # parser.add_argument('--show-vid', action='store_true', default=True, help='display tracking video results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    # parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    # parser.add_argument('--save-trajectories', default=True, action='store_true', help='save trajectories for each track')
    # parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    # parser.add_argument('--save-vid', default=True, action='store_true', help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--exist-ok', default=True, action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    # parser.add_argument('--draw-track-lines-bottom', default=True, action='store_true', help='display object trajectory lines')
    parser.add_argument('--track-lines-len', default=50, action='store_true', help='len of drawing trajectory lines')
    # parser.add_argument('--save-track-lines-bottom', default=True, action='store_true', help='len of drawing trajectory lines')
    # parser.add_argument('--source-homorgafhy', type=str, default='', help='show byrd eye and homography file')
    parser.add_argument('--source-homorgafhy', type=str, help='show byrd eye and homography file')
    # parser.add_argument('--source-homorgafhy', type=str, default='./HallWayTracking/HallWayTracking/homography/001.txt', help='show byrd eye and homography file')
    parser.add_argument('--track-source', type=str, help='show byrd eye and homography file')
    # parser.add_argument('--track-source', type=str, default='./runs/track/exp/tracks/001.txt', help='show byrd eye and homography file')
    parser.add_argument('--homography-output-file', type=str, help='bird eye output fie')
    # parser.add_argument('--homography-output-file', type=str, default='./runs/track/exp/homography/001.avi',help='bird eye output fie')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


