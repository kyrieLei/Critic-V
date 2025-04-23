import os
import torch
import argparse
import sys
import json
import pandas as pd
from PIL import Image
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import re
import logging
from utils.load_model import load_model_and_tokenizer
from utils.get_reponse import get_gpt_response, get_internVL_response, get_deepseek_response, get_GLM_response, get_llava_response, get_miniCPM_response, get_Qwen2VL_response 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

inference_map = {
    'qwen2-vl':get_Qwen2VL_response,
    "chatglm":get_GLM_response,
    'intern-vl':get_internVL_response,
    'minicpm-v':get_miniCPM_response,
    'llava':get_llava_response,
    'deepseek-vl':get_deepseek_response
}
def get_open_source_response(func_name, *args, **kwargs):
    if func_name in inference_map:
        # 调用对应的函数
        return inference_map[func_name](*args, **kwargs)
    else:
        raise ValueError(f"Function '{func_name}' not found in the map.")
def error_detect(input_file, output_file,model_name, model_path, device_name='cuda'):
    device = torch.device(device_name)
    with open(input_file,'r') as in_f, open(output_file,'w') as o_f:
        for line in in_f.readlines():
            data = json.loads(line)
            question = data["question"]
            answer = data["answer_with_hallucinations"]
            image_url = data["image"]

            format = """{
                                    "num_detections": "Total number of hallucinations found",
                                    "detection_1": "first hallucination",
                                    "detection_2": "second hallucination",
                                    ...
                                }"""

            prompt = f"""
                            You will be provided with an image, a question related to the image, and an answer that may contain hallucinations (false information not corresponding to the actual content of the image).
                            
                            question: {question}
                            answer: {answer}

                            Your task is to:

                            Carefully examine the image , the related question, and the provided answer.
                            Identify any hallucinations (false or fabricated information) in the answer that do not match the actual content of the image or the question.
                            Return your analysis in the following Json format(if there doesn't exist any hallucinations, just need to return the number of hallucinations using following format):
                            {format}
                    """
            while True:
                try:
                    if model_path is None:
                        res = get_gpt_response()
                    else:
                        model, processor = load_model_and_tokenizer(model_name, model_path, device)
                        res = get_open_source_response(model_name,model,processor,prompt,image_url)
                    res = re.search(r'{.*}', res, re.DOTALL)

                    try:
                        res = res.group()
                    except:
                        break
                    
                    logging.info(f'{res}')
                    res = json.loads(res)
                    update_res = {f'{model_name}_{key}':value for key,value in res.items()}
                    output_file.write(json.dumps(data|update_res)+'\n')
                    break

                except:
                    pass
