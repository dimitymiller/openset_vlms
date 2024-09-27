import json
import sklearn.metrics as sm
import numpy as np
import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os, sys

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--file", type = str, help = "path to predictions")
parser.add_argument("--plots", action = 'store_true', help = "Plot the PR curve and uncertainty histograms")
parser.add_argument("--auroc", action = 'store_true', help = "Report ROC Curve metrics")
parser.add_argument("--save", action = 'store_true', help = "Save results")
parser.add_argument("--save_dir", type = str, help = "path to save results", default = 'my_results.json')
args = parser.parse_args()


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


##################################################################################################################################
#######################################      DETAILS ABOUT DETECTOR     ##########################################################
##################################################################################################################################

file_info = args.file.split('/')

detector = file_info[1]
test_type = file_info[2].replace(detector, '').replace('coco.json', '').replace('_', '')

if 'rand' in test_type:
    rand_type = test_type.replace('rand', '').split('s')
    emb_types = [rand_type[0]+'s']
    emb_counts = [int(rand_type[1])]
    
elif test_type == 'standard': #equivalent to having 0 words or 0 embeddings
    emb_types = ['words', 'embs']
    emb_counts = [0, 0]
    
else:
    if args.save:
        print(f'Havent implemented saving functionality for {test_type}')
        exit()
    

##################################################################################################################################
#######################################      LOAD GT ANNOS              ##########################################################
##################################################################################################################################
ann_file = 'data/coco/annotations/instances_val2017.json'
with open(ann_file, 'r') as f:
    ann_data = json.load(f)

#associate image id to image name
id_name_map = {}
name_id_map = {}
for im_dict in ann_data['images']:
    id_name_map[im_dict['id']] = im_dict['file_name']
    name_id_map[im_dict['file_name']] = im_dict['id']

#associate category id to category idx
cat_label_map = {}
class_names_all = []
name_to_idx = {}
for idx, item in enumerate(ann_data['categories']):
    cat_label_map[item['id']] = idx
    class_names_all.append(item['name'])
    name_to_idx[item['name']] = idx

label_cat_map = {v: k for k, v in cat_label_map.items()}
    
gt_annos = {}
for anno_dict in ann_data['annotations']:
    im_name = id_name_map[anno_dict['image_id']]
    
    if im_name not in gt_annos.keys():
        gt_annos[im_name] = {'boxes': [], 'labels': [], 'ignore': []}
        
    bbox = anno_dict['bbox']
    bbox_xyxy = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] #format as [xmin, ymin, xmax, ymax]
    
    lbl = anno_dict['category_id']
    lbl_idx = cat_label_map[lbl]

    gt_annos[im_name]['boxes'] += [bbox_xyxy]
    gt_annos[im_name]['labels'] += [lbl_idx]
    gt_annos[im_name]['ignore'] += ['iscrowd' in anno_dict.keys() and anno_dict['iscrowd']]

print(f"Finished loading GT annotations. There are {len(gt_annos.keys())}.")   
num_gt_classes = len(class_names_all)


##################################################################################################################################
#######################################      LOAD PREDICTIONS           ##########################################################
##################################################################################################################################
file = args.file

with open(file, 'r') as f:
    predictions = json.load(f)

print("Finished loading predictions.")

unc_types = predictions['unc-types']

# are there the correct number of entries in predictions?
if len(predictions['detections'].keys()) != len(gt_annos.keys()):
    print(f"There are {len(gt_annos.keys())} GT annotations and {len(predictions['detections'].keys())} predictions. Mismatch.")

all_results = {'tp' : {}, 'ose': {}}
for unc in unc_types:
    for k in all_results.keys():
        all_results[k][unc[0]] = []
        

##################################################################################################################################
#######################################     EVALUATE    FUNCTIONS       ##########################################################
##################################################################################################################################
#finds TP's given an image's detections and GT's
def evaluateImg(detections, gt, num_classes = num_gt_classes, iou_thresh = 0.5):
    tps = np.zeros(len(detections['labels']))
    det_idxs = np.arange(len(detections['labels']))
    
    for cat_id in range(num_classes):
        #filter for detections from this category, gt from this category
        mask_det = detections['labels'] == cat_id
        mask_gt = np.array(gt['labels']) == cat_id
        
        if np.sum(mask_gt) == 0 or np.sum(mask_det) == 0:
            continue
                
        d_s = detections['scores'][mask_det]
        d_b = detections['boxes'][mask_det]

        gt_b = np.array(gt['boxes'])[mask_gt]
        gt_ig = np.array(gt['ignore'])[mask_gt]

        # sort dt highest score first, sort gt ignore last
        dtind = np.argsort(d_s, kind='mergesort')[::-1]
        gtind = np.argsort(gt_ig, kind = 'mergesort')
        
        #compute ious
        ious = iouCalc(d_b, gt_b)
        gt_m  = np.zeros((len(gtind)))

        iou = iou_thresh
        for dind in dtind:
            # information about best match so far (m=-1 -> unmatched)
            m   = -1
            for gind in gtind:
                # if this gt already matched, and not a crowd, continue
                if gt_m[gind]>0 and not gt_ig[gind]:
                    continue
                         
                # if dt matched to reg gt, and on ignore gt, stop
                if m>-1 and gt_ig[m]==0 and gt_ig[gind]==1:
                    break
                    
                # continue to next gt unless better match made
                if ious[dind,gind] < iou:
                    continue
                    
                # if match successful and best so far, store appropriately
                iou=ious[dind,gind]
                m=gind
            
            if m != -1:
                gt_m[m] = 1
                base_idx = det_idxs[mask_det][dind]
                tps[base_idx] = 1
                
    return np.array(tps)
        
    
#function used to calculate IoU between boxes, taken from: https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
def iouCalc(boxes1, boxes2):
    def run(bboxes1, bboxes2):
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
        return iou
    return run(boxes1, boxes2)


map_results = []
for im_name in predictions['detections'].keys():
    if im_name not in gt_annos.keys():
        print(f"Prediction for {im_name} but {im_name} not in GT annotations file.")
        break

    gt_data = gt_annos[im_name]

    #Find TP's from testing with all labels
    all_data = predictions['detections'][im_name]['all']
    all_data['labels'] = [name_to_idx[l] for l in all_data['labels']]  #convert from text label to class idx
    for k in all_data.keys():
        all_data[k] = np.array(all_data[k]) #convert to np array for later masking operations
    
    tps = evaluateImg(all_data, gt_data)

    if np.sum(tps) != 0:
        for unc in all_results['tp'].keys():
            tp_data = all_data[unc][tps == 1]
            all_results['tp'][unc] += tp_data.tolist()

    #Find OSE when testing with open-set labels
    os_data = predictions['detections'][im_name]['open-set']
    os_data['labels'] = [name_to_idx[l] for l in os_data['labels']]  #convert from text label to class idx
    for unc in all_results['ose'].keys():
        all_results['ose'][unc] += os_data[unc]

    #save predictions in pycocotools format for mAP calculation
    im_id = name_id_map[im_name]
    for d_idx in range(len(all_data['labels'])):
        b = all_data['boxes'][d_idx].tolist().copy()
        b[2] = b[2]-b[0] #convert back to xywh
        b[3] = b[3]-b[1] #convert back to xywh
        l = label_cat_map[all_data['labels'][d_idx]] #convert back to category id
        map_results.extend(
            [
                {
                    "image_id": im_id,
                    "category_id": l,
                    "bbox": b,
                    "score": all_data['scores'][d_idx],
                }
            ]
        )

print("Evaluation complete, computing results...")

##################################################################################################################################
#######################################     CALCULATE RESULTS           ##########################################################
##################################################################################################################################
print("Calculating mAP")
with HiddenPrints():
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(map_results)
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


map50 = coco_eval.stats[1]

if args.plots:
    fig, ax = plt.subplots(1, figsize = (5, 4))
    

modelNames = {'ovrcnn': 'OVR-CNN', 'ovdetr': 'OV-DETR', 'vild': 'ViLD', 'regionclip': 'RegionCLIP', 'detic': 'Detic', 'vldet': 'VLDet', 'cora':'CORA'}

print(f'Testing {detector} in {test_type} configuration')   
for name, dir in unc_types:
    true = np.concatenate((np.ones(len(all_results['tp'][name])), np.zeros(len(all_results['ose'][name]))))       
    scores = np.concatenate((all_results['tp'][name], all_results['ose'][name]))
        
    if dir == 'high':
        scores = -scores
        
    auroc = sm.roc_auc_score(true, scores, average = 'macro')
    fpr, tpr, thresholds = sm.roc_curve(true, scores, pos_label=1)

    tpr95_idx = np.argmin(abs(tpr-0.95))
    fpr_95 = fpr[tpr95_idx]

    precision, recall, thresholds = sm.precision_recall_curve(true, scores)
    aupr = sm.auc(recall, precision)

    if args.plots:
        fig1, ax1 = plt.subplots(1, figsize = (4, 3))
        n = name.title()
        if len(unc_types) == 3 or name == 'scores': #if there is only scores and cosine, they are equivalent (sigmoid)
            ax.plot(100.*recall, 100.*precision, label = n)
        
        try:
            weights1 = np.ones_like(all_results['tp'][name])/float(len(all_results['tp'][name]))
            weights2 = np.ones_like(all_results['ose'][name])/float(len(all_results['ose'][name]))
            ax1.hist([all_results['tp'][name], all_results['ose'][name]], bins = 15, color = ['#32A7BF', '#F59D3B'], weights = [weights1, weights2], label = ['Correct closed-set', 'Open-set error'])
        except: #something weird with regionclip cosine saved results
            tp = np.array(all_results['tp'][name]).reshape(-1)
            ose = np.array(all_results['ose'][name]).reshape(-1)
            weights1 = np.ones_like(tp)/float(len(tp))
            weights2 = np.ones_like(ose)/float(len(ose))
            ax1.hist([tp, ose], bins = 15, color = ['#32A7BF', '#F59D3B'], weights = [weights1, weights2], label = ['Correct closed-set', 'Open-set error'])
            
        ax1.set_ylabel('Normalised Count', fontsize = 12)
        if name == 'scores':
            kN = 'Softmax score'
            ax1.set_xlim((0, 1))
        else:
            kN = name.title()
        ax1.set_xlabel(kN, fontsize = 12)
        ax1.legend(fontsize = 11)
        fig1.suptitle(modelNames[detector], fontsize = 13)
        fig1.savefig(f'figures/UncertaintyHistogram_{modelNames[detector]}_{name}.pdf', bbox_inches = 'tight')
    
    print(f'Results for uncertainty type: {name}')
    print(f'     AUPR: {aupr*100 :.1f}')

    if args.auroc:
        print(f'     AUROC: {auroc*100 :.1f}')

    for recall_rate in [0.95]:
        rec_idx = np.argmin(abs(recall-recall_rate))
        prec = precision[rec_idx]
        
        if abs(recall[rec_idx]-recall_rate) < 0.01:
            print(f'     Precision@{100.* recall_rate :.0f}Recall: {prec*100 :.1f}')
        else:
            print(f'Could not find a Recall within 1% of {100.*recall_rate}%')
            
    for precision_rate in [0.95]:
        prec_idx = np.argmin(abs(precision-precision_rate))
        rec = recall[prec_idx]
        
        if abs(precision[prec_idx]-precision_rate) < 0.01:
            print(f'     Recall@{100.* precision_rate :.0f}Precision: {rec*100 :.1f}')
        else:
            print(f'Could not find a Precision within 1% of {100.*precision_rate}%')

    if args.save and name == 'scores': #only setup to save for score uncertainty
        try:
            with open(args.save_dir, 'r') as f:
                save_results = json.load(f)
        except:
            print(f'No prior save results file exists at {args.save_dir}. Creating new.')
            save_results = {}
    
        if detector not in save_results.keys():
            save_results[detector] = {}
    
        for idx, emb_type in enumerate(emb_types):
            neg_count = emb_counts[idx]
    
            if emb_type not in save_results[detector].keys():
                save_results[detector][emb_type] = {'map': {}, 'aupr': {}, 'tp': {}, 'ose': {}, 'num_neg': {}, '95rec': {}, '95prec': {}}
                
            save_results[detector][emb_type]['map'][neg_count] = map50
            save_results[detector][emb_type]['aupr'][neg_count] = aupr
            save_results[detector][emb_type]['tp'][neg_count] = len(all_results['tp'][name])
            save_results[detector][emb_type]['ose'][neg_count] = len(all_results['ose'][name])
            save_results[detector][emb_type]['num_neg'][neg_count] = neg_count
            save_results[detector][emb_type]['95rec'][neg_count] = prec
            save_results[detector][emb_type]['95prec'][neg_count] = rec

        with open(args.save_dir, 'w') as f:
            json.dump(save_results, f)

print(f'Total TP: {len(all_results['tp']['scores'])}')
print(f'Total OSE: {len(all_results['ose']['scores'])}')

print(f'mAP at IoU 0.5: {100.*map50 :.1f}')

if args.plots:
    ax.legend(loc = 'lower left', fontsize = 12)
    ax.set_xlabel('Recall (%)', fontsize = 12)
    ax.set_ylabel('Precision (%)', fontsize = 12)
    fig.suptitle(f'{modelNames[detector]} with uncertainty baselines', fontsize = 14) 
    fig.savefig(f'figures/PRCurve_{modelNames[detector]}.pdf', bbox_inches = 'tight')
