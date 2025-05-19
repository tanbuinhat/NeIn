import json
import glob
import copy
from argparse import ArgumentParser
import numpy as np 

def parse_score(removal_results):
    scores = []
    for img in removal_results:
        if img["removal_evaluation_ovd"] == 1:
            scores.append(0)
        else:
            scores.append(img["highest_confidence_score"])
    return scores

def calc_auc(scores, gt=0, num_classes=2):
    y_true = np.array([0] * len(scores))
    scores_sorted = np.sort(scores)
    auc = 0.0
    for i in range(len(scores_sorted)):
        if i+1 == len(scores_sorted):
            auc += 1.0 - scores_sorted[i]
        else:
            auc += (i+1)*1.0 / len(scores_sorted) * (scores_sorted[i+1]-scores_sorted[i])

    return auc * 100

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--removal-json", type=str, help='json file of removal evaluation from specific method')
    args = parser.parse_args()
    
    with open(args.removal_json, 'r') as file:
        removal_results = json.load(file)
    
    scores = parse_score(removal_results)
    auc_roc = calc_auc(scores)

    print("AUC_removal", auc_roc)
