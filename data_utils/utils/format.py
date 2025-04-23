import json
import json
import re
from tqdm import tqdm

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def dpo_format():
    file_path = './overall_shuffle.jsonl'
    data = read_jsonl_file(file_path)


    dpo_data_ls = []
    for dic in data[:]:
        question = dic['question']
        image = dic['image']
        answer = dic['answer_with_hallucinations']
        
        # 拼接三个model的批评
        model_detection_dic = {}
        for i in range(1, 4):    
            model_detections = []    
            num_detections = int(dic[f"model{i}_num_detections"])
            # 可能批评的数量与实际数量不一致
            try:
                for j in range(1, num_detections+1):
                    model_detections.append(dic[f"model{i}_detection_{j}"])
                        
                model_detections_str = ' '.join([f"{i+1}. {item}" for i, item in enumerate(model_detections)])

                model_detection_dic[f'model{i}'] = {
                        'detection':model_detections_str,
                        'score': dic[f'model{i}_total']
                    }
            except:
                pass
            # print(model_detections_str)

        print(json.dumps(model_detection_dic, indent=4, ensure_ascii=False))
        
        # 在score满足条件的前提下，两两组合
        valid_pairs = []
        for k in model_detection_dic:
            for j in model_detection_dic:
                if k != j and model_detection_dic[k]['detection'] and model_detection_dic[j]['detection']:
                    if model_detection_dic[k]['score'] > model_detection_dic[j]['score']:
                        valid_pairs.append((k, j))
                        print(k, j)
        
        for chosen, rejected in valid_pairs:
            dpo_data_ls.append({
                "conversations": 
                [
                    {
                        "from": "human",
                        "value": f'<image>\nQuestion: {question}, Answer: {answer}',
                    }
                ],
                "chosen": 
                {
                    "from": "gpt",
                    "value": model_detection_dic[chosen]['detection']
                },
                "rejected": 
                {
                    "from": "gpt",
                    "value": model_detection_dic[rejected]['detection']
                },
                "images": 
                [
                    image
                ]
            })
            

    with open('dpo_data_overall_new.json', 'wt', encoding='utf-8') as file:
        json.dump(dpo_data_ls, file, ensure_ascii=False, indent=4)


def clean_input(string):
    ret = string.replace('Question: ','\n#### Question\n')
    ret = ret.replace(', Answer: ','\n#### Answer\n')
    ret = ret + f'\n#### Task\nPlease provide a critique of the answer above. What are the weaknesses of the answer?'
    return ret

def clean_output(string):
    num_dot = re.compile(r'\d+\. ')
    ret = num_dot.sub('\n',string)
    ret = f'#### Critique\n{ret}'
    return ret

def clean_json():
    o = open('dpo_data_jacard_cleaned.json','w')
    outputs = []
    with open('./dpo_data_jacard.json','r') as f:
        data = json.load(f)
        for row in tqdm(data):
            
            for i in range(len(row['conversations'])):
                row['conversations'][i]['value'] = clean_input(row['conversations'][i]['value'])
            
            row['chosen']['value'] = clean_output(row['chosen']['value'])
            row['rejected']['value'] = clean_output(row['rejected']['value'])
            outputs.append(row)

    json.dump(outputs,o,indent=4)

clean_json()