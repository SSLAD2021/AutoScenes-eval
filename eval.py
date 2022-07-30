"""
Evaluation Server
"""
import os
os.system('pip install pyquaternion')
import sys
import pickle
from eval_bev import evaluate_map
from eval_det import evaluate_det


def eval(gt_data, pred_data):
    dist_ths_list = [0.5, 1.0, 2.0, 4.0]
    bev_metric = evaluate_map(gt_data['gt_bev'], pred_data['bev'])
    det_metric = evaluate_det(gt_data['gt_det'], pred_data['det'], dist_ths_list)
    return bev_metric, det_metric


def main():
    input_dir, output_dir = sys.argv[1], sys.argv[2]
    gt_data = pickle.load(open(os.path.join(input_dir, 'ref', 'ads_gt_infos.pkl'), 'rb'))
    pred_data = pickle.load(open(os.path.join(input_dir, 'test_res', 'result.pkl'), 'rb'))
    bev_metric, det_metric = eval(gt_data, pred_data)
    score = (bev_metric['map/mean/iou@max'] + det_metric['NDS']) / 2.0

    fwriter = open(os.path.join(output_dir, 'scores.txt'), 'w')
    for k, v in det_metric.items():
        fwriter.write('{}: {:.4f}\n'.format(k, v))
    fwriter.write('mIOU: {:.4f}\n'.format(bev_metric['map/mean/iou@max']))
    fwriter.write('Score: {:.4f}\n'.format(score))
    fwriter.close()


if __name__ == '__main__':
    main()
