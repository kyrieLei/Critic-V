from openai import OpenAI
import json
import logging
import re
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.get_reponse import get_gpt_response

def get_gpt_score(correct_answer, model_response,api_key):
    scoring_prompt = f"""
    Please evaluate the model's detection of hallucinations by answering the following questions:

    Given:
    Ground Truth: {correct_answer}
    Model's response: {model_response}

    Please analyze based on these specific criteria:

    1. Coverage Analysis:
    - Did the model identify all the hallucinations mentioned in the correct answer?
    - Are there any significant hallucinations that were missed?

    2. Accuracy Assessment:
    - Are the detected items genuine hallucinations (true positives)?
    - Are there any false detections (false positives)?

    3. Precision of Description:
    - How precise and clear are the model's descriptions of the detected hallucinations?
    - Is the explanation specific enough to understand what exactly is wrong?

    4. Overall Effectiveness:
    - How effective is this detection compared to an ideal detection?
    - Does it provide practical value for hallucination identification?

    Based on your analysis, please provide:
    1. A brief explanation of your evaluation
    2. A final score from 0 to 10, where:

    Score Range Criteria:
    0-2 (Severe Failure):
    - 0: No hallucinations detected at all
    - 1: Detected less than 20% of hallucinations, or majority of detections are false
    - 2: Detected only 20-30% of hallucinations, with multiple false positives

    3-4 (Poor Performance):
    - 3: Detected 30-40% of hallucinations, but with significant false positives
    - 4: Detected 40-50% of hallucinations, with some false positives or unclear descriptions

    5-6 (Moderate Performance):
    - 5: Detected 50-60% of hallucinations, with acceptable accuracy
    - 6: Detected 60-70% of hallucinations, with minor false positives and mostly clear descriptions

    7-8 (Good Performance):
    - 7: Detected 70-80% of hallucinations, with very few false positives
    - 8: Detected 80-90% of hallucinations, with clear and precise descriptions

    9-10 (Excellent Performance):
    - 9: Detected >90% of hallucinations, with precise descriptions and no false positives
    - 10: Perfect detection - all hallucinations identified correctly, with clear and accurate descriptions
    (because ground truth is incorrect part found in an answer, if response mentioned something is not, you should consider it as an overlap to ground truth)
    (because ground truth is incorrect part found in an answer, if response mentioned something is not, you should consider it as an overlap to ground truth)
    (because ground truth is incorrect part found in an answer, if response mentioned something is not, you should consider it as an overlap to ground truth)
    Please return your response in this format:
    your analysis: [brief explanation]
    ### score: [numerical score between 0 and 10]
    """
    while True:
        try:
            result = get_gpt_response("gpt-4o-mini",prompt=scoring_prompt,image_url=None,api_key=api_key)
            match = re.search(r'### score:\s*([\d.]+)', result)

            if match:
                res = match.group(1)  # 获取匹配的字符串
                return int(res) / 10
        except:
            pass


def get_jaccard_score(model_name1, model_name2, model_name3, hallucinations, critic1,critic2,critic3, api_key):
    format = f"""{{
                {model_name1}_score:
                {model_name2}_score:
                {model_name3}_score:
                }}
            """
    prompt = f"""
                You are given a ground truth and three model response related to this ground truth.
                Ground Truth:{hallucinations}
                {model_name1} Response: {critic1}
                {model_name2} Response: {critic2}
                {model_name3} Response: {critic3}
                Your task is to calculate the overlap or similarity between each reponse and ground truth using the Jaccard index(if no response from model, just assign a 0 score). 
                The Jaccard index measures the similarity between two sets by dividing the size of their intersection by the size of their union.

                Instructions:
                Treat each element in the lists as an individual description.For each pair, check if they convey the same or similar meaning.
                Based on the overlap of matching or similar descriptions between the two lists, compute the Jaccard index according to the following formula. 
                    JaccardIndex= |A \cap B|/|A \cup B|
                    (because ground truth is incorrect part found in an answer, if response mentioned something is not, you should consider it as an overlap to ground truth)
                    (because ground truth is incorrect part found in an answer, if response mentioned something is not, you should consider it as an overlap to ground truth)
                    (because ground truth is incorrect part found in an answer, if response mentioned something is not, you should consider it as an overlap to ground truth)
                Return the final Jaccard index score, rounded to two decimal places. Structure the output in the following Json format:
                {format}
            """

    result = get_gpt_response("gpt-4o-mini", prompt=prompt,image_url=None,api_key=api_key)



def gpt_compute(data):
    
    hallucinations = []
    model1_detections = []
    model2_detections = []
    model3_detections = []
    num_hallucinations = int(data["num_hallucinations"])
    for i in range(num_hallucinations):
        try:
            hallucinations.append(data[f"hallucination_{i+1}"])
        except:
            pass
    hallucinations = ' '.join([f"{i+1}. {item}" for i, item in enumerate(hallucinations)])

    model1_num_detections = int(data["model1_num_detections"])
    for i in range(model1_num_detections):
        try:
            model1_detections.append(data[f"model1_detection_{i+1}"])
        except:
            pass
    model1_detections = ' '.join([f"{i+1}. {item}" for i, item in enumerate(model1_detections)])

    model2_num_detections = int(data["model2_num_detections"])
    for i in range(model2_num_detections):
        try:
            model2_detections.append(data[f"model2_detection_{i+1}"])
        except:
            pass
    model2_detections = ' '.join([f"{i+1}. {item}" for i, item in enumerate(model2_detections)])
    
    model3_num_detections = int(data["model3_num_detections"])
    for i in range(model3_num_detections):
        try:
            model3_detections.append(data[f"model3_detection_{i+1}"])
        except:
            pass
    model3_detections = ' '.join([f"{i+1}. {item}" for i, item in enumerate(model3_detections)])
    
    
    model1_correction = get_gpt_score(hallucinations, model1_detections)

    model2_correction = get_gpt_score(hallucinations,model2_detections)

    model3_correction = get_gpt_score(hallucinations,model3_detections)
    

    data['model1_correction'] = model1_correction
    data['model2_correction'] = model2_correction
    data['model3_correction'] = model3_correction
    


    output_filename = f"./gpt_score.jsonl"
    with open(output_filename, 'w',encoding='utf-8') as file:
        file.write(json.dumps(data,ensure_ascii=False)+'\n')
    


def jaccard_compute(input_file,output_file,model_name1,model_name2,model_name3):
   
    with open(input_file, 'r') as in_f, open(output_file, 'a') as o_f:
        for line in in_f.readlines():
            data = json.loads(line)
        
            hallucinations = []
            model1_detections = []
            model2_detections = []
            model3_detections = []

            num_hallucinations = int(data["num_hallucinations"])
            for i in range(num_hallucinations):
                try:
                    hallucinations.append(data[f"hallucination_{i+1}"])
                except:
                    pass
            hallucinations = ' '.join([f"{i+1}. {item}" for i, item in enumerate(hallucinations)])

            model1_num_detections = int(data["{model_name_1}_num_detections"])
            for i in range(model1_num_detections):
                try:
                    model1_detections.append(data[f"model1_detection_{i+1}"])
                except:
                    pass
            model1_detections = ' '.join([f"{i+1}. {item}" for i, item in enumerate(model1_detections)])

            model2_num_detections = int(data["{model_name_2}_num_detections"])
            for i in range(model2_num_detections):
                try:
                    model2_detections.append(data[f"model2_detection_{i+1}"])
                except:
                    pass
            model2_detections = ' '.join([f"{i+1}. {item}" for i, item in enumerate(model2_detections)])
            
            model3_num_detections = int(data["{model_name_3}_num_detections"])
            for i in range(model3_num_detections):
                try:
                    model3_detections.append(data[f"model3_detection_{i+1}"])
                except:
                    pass
            model3_detections = ' '.join([f"{i+1}. {item}" for i, item in enumerate(model3_detections)])
            
            while True:
                try:
                    response=None
                    
                    while not response:
                        response = get_jaccard_score(model_name1, model_name2, model_name3, hallucinations, model1_detections, model2_detections, model3_detections)
                        pattern = r'```json\n({.*?})'  
                        response = re.search(pattern, response, re.DOTALL)
                
                    response = response.group(1)
                    
                    break
                except:
                    pass

            res = json.loads(response)
    
            o_f.write(json.dumps(data|res,ensure_ascii=False) + '\n')





def score_add(jaccard_path,gpt_path):

    with open(jaccard_path, 'r',encoding='utf-8') as in_f, open(gpt_path, 'w',encoding='utf-8') as out:
        for line in in_f:
            data = json.loads(line)  # Load each JSON object from the file

            # Calculate the sum of model_score and model_correction for each model
            data["model1_total"] = round(data.get('model1_score', 0) + data.get('model1_correction', 0),2)
            data["model2_total"] = round(data.get('model2_score', 0) + data.get('model2_correction', 0),2)
            data["model3_total"] = round(data.get('model3_score', 0) + data.get('model3_correction', 0),2)
            out.write(json.dumps(data,ensure_ascii=False)+'\n')


