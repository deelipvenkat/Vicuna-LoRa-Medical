__author__ = 'Qiao Jin'

import json
from sklearn.metrics import accuracy_score, f1_score
import sys

def pub_scores(pred_path ):


    ground_truth = json.load(open('/home/medical-llama/Pubmedqa/pubmedqa/data/test_ground_truth.json')) 
    predictions = json.load(open(pred_path))

    assert set(list(ground_truth)) == set(list(predictions)), 'Please predict all and only the instances in the test set.'

    pmids = list(ground_truth)
    truth = [ground_truth[pmid] for pmid in pmids]
    preds = [predictions[pmid] for pmid in pmids]

    acc = accuracy_score(truth, preds) * 100
    maf = f1_score(truth, preds, average='macro')*100

    return acc , maf
