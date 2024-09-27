import matplotlib.pyplot as plt

import numpy as np
import scipy
import scipy.stats as st
import argparse

import sklearn.metrics as sm

parser = argparse.ArgumentParser()
parser.add_argument("--file", type = str, help = "path to numpy file containing cosine predictions")
parser.add_argument("--plots", action = 'store_true', help = "Plot the PR curve")
parser.add_argument("--auroc", action = 'store_true', help = "Report ROC Curve metrics")
args = parser.parse_args()

data_sizes = {'imagenet': 1000, 'food101': 101, 'places365': 365, 'gtsrb': 43}

##################################################################################################################################
#######################################      DETAILS ABOUT TEST         ##########################################################
##################################################################################################################################

file_info = args.file.split('/')

classifier = file_info[1]
dataset = file_info[2]

test_info = file_info[3].split('_')
if test_info[0] == 'standard':
    test_type = 'standard'
else:
    neg_count = test_info[0]
    test_type = test_info[1]

gt_labels = np.load(args.file.replace('cosine', 'gt'))
all_cosine = np.load(args.file)

raw_results = {'softmax': [[], []], 'cosine': [[], []], 'entropy': [[], []]}
num_classes = data_sizes[dataset]
for idx in range(len(all_cosine)):
    target = gt_labels[idx]    
    cosine = all_cosine[idx]

    softmax = scipy.special.softmax(cosine)
    pred_idx = np.argmax(softmax)
   
    pred_score = softmax[pred_idx]
    pred_cos = cosine[pred_idx]
    entropy = -st.entropy(softmax) #negative because we want high numbers to equal low uncertainty
    
    correct = pred_idx == target

    if correct:
        raw_results['softmax'][0] += [pred_score]
        raw_results['cosine'][0] += [pred_cos] 
        raw_results['entropy'][0] += [entropy]

    ###################################################################################
    known_classes = np.concatenate((np.arange(0, target), np.arange(target+1, len(cosine))))

    cosine_subset = cosine[known_classes]

    softmax = scipy.special.softmax(cosine_subset)
    pred_idx = np.argmax(softmax)

    if pred_idx >= (num_classes-1): #-1 because we removed the target class
        continue
    
    pred_score = softmax[pred_idx]
    pred_cos = cosine_subset[pred_idx]
    entropy = -st.entropy(softmax) #negative because we want high numbers to equal low uncertainty

    raw_results['softmax'][1] += [pred_score]
    raw_results['cosine'][1] += [pred_cos] 
    raw_results['entropy'][1] += [entropy] 

if test_type == 'standard':
    print(f'Testing {classifier} on {dataset} in standard configuration')   
else:
    print(f'Testing {classifier} on {dataset} in {neg_count} random {test_type} configuration')   

if args.plots:
    fig, ax = plt.subplots(1, figsize = (5, 4))
    

modelNames = {'clip': 'CLIP', 'align': 'ALIGN', 'imagebind': 'ImageBind'}
for k in raw_results.keys():
    if args.plots:
        fig1, ax1 = plt.subplots(1, figsize = (4, 3))
        
    known_c = raw_results[k][0]
    unknown = raw_results[k][1]
    
    true = np.concatenate((np.ones(len(known_c)), np.zeros(len(unknown))))
    scores = np.concatenate((known_c, unknown))
    
    auroc = sm.roc_auc_score(true, scores, average = 'macro')
    fpr, tpr, thresholds = sm.roc_curve(true, scores, pos_label=1)
    
    tpr95_idx = np.argmin(abs(tpr-0.95))
    fpr_95 = fpr[tpr95_idx]
    
    acc = len(known_c)/len(gt_labels)
    
    precision, recall, thresholds = sm.precision_recall_curve(true, scores)
    aupr = sm.auc(recall, precision)

    if args.plots:
        kN = k.title()
        ax.plot(100.*recall, 100.*precision, label = kN)

        ax1.hist([known_c, unknown], bins = 30, color = ['#32A7BF', '#F59D3B'], label = ['Correct closed-set', 'Open-set error'])
        ax1.set_ylabel('Count', fontsize = 12)
        if k == 'scores':
            kN = 'Softmax score'
            ax1.set_xlim((0, 1))
        else:
            kN = k.title()
        ax1.set_xlabel(kN, fontsize = 12)
        ax1.legend(fontsize = 11)
        fig1.suptitle(modelNames[classifier], fontsize = 13)
        fig1.savefig(f'figures/UncertaintyHistogram_{modelNames[classifier]}_{k}.pdf', bbox_inches = 'tight')
        
    
    print(f'Results for uncertainty: {k}')
    print(f'     AUPR: {aupr*100 :.1f}')

    if args.auroc:
        print(f'     AuROC: {auroc*100 :.1f}')
        
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
            
   
print(f'Total TP: {len(known_c)}')
print(f'Total OSE: {len(unknown)}')

print(f'Accuracy: {100.*acc :.1f}')


if args.plots:
    ax.legend(loc = 'lower left')
    ax.set_xlabel('Recall (%)', fontsize = 12)
    ax.set_ylabel('Precision (%)', fontsize = 12)
    fig.suptitle(f'{modelNames[classifier]} with uncertainty baselines', fontsize = 14) 
    fig.savefig(f'figures/PRCurve_{modelNames[classifier]}.pdf', bbox_inches = 'tight')
    

    