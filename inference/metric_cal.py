from glob import glob
import json
import sys
import re
import os

def check(gt,ans):
    if gt == ans:
        return True
    options = re.findall(r'([A-Z])\.', ans)
    if len(options) == 0:
        options = re.findall(r'([A-Z])', ans)
        if len(options) == 0:
            return None
        label = options[0]
    else:
        label = options[0]
    if label == gt:
        return True
    else:
        return False
    
def get_accuracy(data_dir):
    metrics = []
    dataset = []
    json_files = glob(os.path.join(data_dir,"*.json"))
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            dataset.append(data)
    for data in dataset:
        metrics.append(check(data['ground_truth'], data['history answer'][-1])) 



    metrics = [x for x in metrics if x is not None]
    print(f'Accuracy: {sum(metrics)/len(metrics)}')
    print(len(metrics))

if __name__=='__main__':
    data_dir = "./results/"
    get_accuracy(data_dir)


