import base64
import json
import logging
import re
from utils.get_reponse import get_gpt_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def error_inject(input_file, output_file, api_key, model_name='gpt-4o-mini'):
    format = """
            {
                    "answer_with_hallucinations": "The original question, modified to include hallucinations",
                    "num_hallucinations": "Total number of hallucinations",
                    "hallucination_1": "Details of the first hallucination",
                    "hallucination_2": "Details of the second hallucination",
                    ...
            }
            """
    
    with open(input_file,'r') as in_f, open(output_file,'w') as o_f:
        for line in in_f.readlines():
            data = json.loads(line)
            image_url = data['image']
            question = data['question']
            answer = data['answer']

            prompt = f"""
                        You are a knowledgeable assistant capable of analyzing images and introduce hallucinations. For each image and corresponding question-answer pair provided, perform the following tasks:
                        Introduce some hallucinations(range from 1 to 5) to the answer. Include the number of hallucinations and provide details for each in answers tainted by illusions.

                        Question: {question}
                        Answer: {answer}

                        Structure the output in the following Json format:
                        {format}
                    """
            while True:
                try:
                    result = None
                    while not result:
                        result = get_gpt_response(model_name, prompt, image_url, api_key)
                        result = re.search(r'{.*?}', result, re.DOTALL)
                    result = result.group()
                    logging.info(result)
                    res = json.loads(result)
                    o_f.write(json.dumps(data|res)+'\n')
                    break
                except:
                        pass
