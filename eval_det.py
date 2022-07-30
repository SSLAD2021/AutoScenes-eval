from collections import defaultdict
import numpy as np
from pyquaternion import Quaternion


class_names = ['car', 'truck', 'bus', 'bicycle', 'pedestrian']
dist_ths_list = [0.5, 1.0, 2.0, 4.0]
class_range_dict = {
    'car': 50,
    'truck': 50,
    'bus': 50,
    'trailer': 50,
    'construction_vehicle': 50,
    'pedestrian': 40,
    'motorcycle': 40,
    'bicycle': 40,
    'traffic_cone': 30,
    'barrier': 30
}


def center_distance(pred_box, gt_box):
    return np.linalg.norm(np.array(pred_box['translation'][:2]) - np.array(gt_box['translation'][:2]))


def scale_iou(pred_box, gt_box):
    sa_size = np.array(gt_box['size'])
    sr_size = np.array(pred_box['size'])
    assert all(sa_size > 0), "Error: sample_annotation sizes must be > 0."
    assert all(sr_size > 0), "Error: sample_result sizes must be > 0"

    min_wlh = np.minimum(sa_size, sr_size)
    volume_annotation = np.prod(sa_size)
    volume_result = np.prod(sr_size)
    intersection = np.prod(min_wlh)
    union = volume_annotation + volume_result - intersection
    iou = intersection / union
    return iou


def angle_diff(x, y, period):
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)
    return diff


def yaw_diff(pred_box, gt_box, period):
    yaw_gt = quaternion_yaw(Quaternion(gt_box['rotation']))
    yaw_est = quaternion_yaw(Quaternion(pred_box['rotation']))
    return abs(angle_diff(yaw_gt, yaw_est, period))


def quaternion_yaw(q):
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
    yaw = np.arctan2(v[1], v[0])
    return yaw


def cummean(x):
    if sum(np.isnan(x) == len(x)):
        return np.ones(len(x))
    else:
        sum_vals = np.nancumsum(x.astype(float))
        count_vals = np.cumsum(~np.isnan(x))
        return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)


def evaluate_det(gt_raw_data, pred_raw_data, dist_th_list):
    pred_data = defaultdict(list)
    gt_data = defaultdict(list)
    pred_cnt = 0
    gt_cnt = 0
    for sample_token, box_list in pred_raw_data['results'].items():
        assert len(box_list) <= 500, "Error: Only <= 500 boxes per sample allowed!"

        for item in box_list:
            pred_data[item['detection_name']].append(item)
            pred_cnt += 1

    for sample_token, box_list in gt_raw_data.items():
        for item in box_list:
            max_distance_limit = class_range_dict[item['detect_name']]
            if np.linalg.norm(
                np.array(item['translation'][:2]) - np.array(item['ego_pose_calibration']['translation'][:2])
            ) > max_distance_limit:
                continue
            if item['num_lidar_pts'] == 0:
                continue
            gt_data[item['detect_name']].append(item)
            gt_cnt += 1

    metric_data = dict()
    for class_name in class_names:
        metric_data[class_name] = dict()
        for dist_th in dist_th_list:
            metric_data[class_name][dist_th] = calc_class_dist(gt_data, pred_data, class_name, dist_th)

    metric = dict()
    for class_name in class_names:
        metric[class_name] = dict()
        for dist_th in dist_th_list:
            ap = calc_ap(metric_data[class_name][dist_th], 0.1, 0.1)
            metric[class_name][dist_th] = ap

        for metric_name in ['trans_err', 'scale_err', 'orient_err']:
            if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                tp = np.nan
            elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                tp = np.nan
            else:
                tp = calc_tp(metric_data[class_name][2], 0.1, metric_name)
            metric[class_name][metric_name] = tp

    metric_out = dict()
    ap_list = []
    for class_name in class_names:
        for dist_th in dist_th_list:
            ap_list.append(metric[class_name][dist_th])
    metric_out['mAP'] = sum(ap_list) / len(ap_list)

    tp_scores = dict()
    for metric_name in ['trans_err', 'scale_err', 'orient_err']:
        errors = []
        for class_name in class_names:
            errors.append(metric[class_name][metric_name])
        metric_out[metric_name] = float(np.nanmean(errors))
        score = 1.0 - metric_out[metric_name]
        score = max(0.0, score)
        tp_scores[metric_name] = score

    NDS = (metric_out['mAP'] * 3.0 + np.sum(list(tp_scores.values()))) / 6.0
    metric_out['NDS'] = NDS
    return metric_out


def calc_class_dist(gt_data, pred_data, class_name, dist_th):
    tp = []
    fp = []
    conf = []
    match_data = {'trans_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'conf': []}
    result_dict = dict()
    pred_class_boxes = pred_data[class_name]
    gt_class_boxes_all = gt_data[class_name]
    npos = len(gt_class_boxes_all)

    if npos == 0:
        return dict(recall=np.linspace(0, 1, 101),
                    precision=np.zeros(101),
                    confidence=np.zeros(101),
                    trans_err=np.ones(101),
                    vel_err=np.ones(101),
                    scale_err=np.ones(101),
                    orient_err=np.ones(101),
                    attr_err=np.ones(101))
    taken = set()
    pred_class_boxes.sort(key=lambda x: x['detection_score'], reverse=True)
    for pred_box in pred_class_boxes:
        sample_token = pred_box['sample_token']
        gt_class_boxes = list([item for item in gt_class_boxes_all if item['sample_token'] == sample_token])

        min_dist = np.inf
        match_gt_idx = None
        for gt_idx, gt_box in enumerate(gt_class_boxes):
            if gt_box['token'] in taken:
                continue
            this_distance = center_distance(pred_box, gt_box)
            if this_distance < min_dist:
                min_dist = this_distance
                match_gt_idx = gt_idx

        if min_dist < dist_th:
            taken.add(gt_class_boxes[match_gt_idx]['token'])
            tp.append(1)
            fp.append(0)
            conf.append(pred_box['detection_score'])
            gt_match_box = gt_class_boxes[match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_match_box, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_match_box, pred_box))
            match_data['orient_err'].append(yaw_diff(gt_match_box, pred_box, period=2 * np.pi))
            match_data['conf'].append(pred_box['detection_score'])
        else:
            tp.append(0)
            fp.append(1)
            conf.append(pred_box['detection_score'])

    if len(match_data['trans_err']) == 0:
        return dict(recall=np.linspace(0, 1, 101),
                    precision=np.zeros(101),
                    confidence=np.zeros(101),
                    trans_err=np.ones(101),
                    vel_err=np.ones(101),
                    scale_err=np.ones(101),
                    orient_err=np.ones(101),
                    attr_err=np.ones(101))

    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, 101)
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    for key in match_data.keys():
        if key == 'conf':
            continue
        else:
            tmp = cummean(np.array(match_data[key]))
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    result_dict['recall'] = rec
    result_dict['precision'] = prec
    result_dict['confidence'] = conf
    result_dict['trans_err'] = match_data['trans_err']
    result_dict['scale_err'] = match_data['scale_err']
    result_dict['orient_err'] = match_data['orient_err']
    return result_dict


def calc_ap(metric_data, min_recall, min_precision):
    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(metric_data['precision'])
    prec = prec[round(100 * min_recall) + 1:]
    prec -= min_precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp(metric_data, min_recall, metric_name):
    first_ind = round(100 * min_recall) + 1
    non_zero = np.nonzero(metric_data['confidence'])[0]
    if len(non_zero) == 0:
        last_ind = 0
    else:
        last_ind = non_zero[-1]
    if last_ind < first_ind:
        return 1.0
    else:
        return float(np.mean(metric_data[metric_name][first_ind: last_ind + 1]))
