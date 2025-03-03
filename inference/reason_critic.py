import os
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
from load_model import load_model_and_tokenizer, gen_one_answer
import json
import argparse
import re


def extract_question_and_options(text):
    # Use regex to find the context (everything before the QUESTION)
    # context_match = re.match(r'(.*?)(?=QUESTION:)', text, re.DOTALL)
    # context = context_match.group(1).strip() if context_match else None

    # Use regex to find the question
    # question_match = re.search(r'QUESTION:(.*?)\n', text)
    # question = question_match.group(1).strip() if question_match else None
    if 'options' not in text.lower():
        end = text.find('A.')
    else:
        end = text.lower().find('options')

    if 'QUESTION:' in text:
        context = text[:text.find('QUESTION:')]
        question = text[text.find('QUESTION:') + len('QUESTION:'):end]
    else:
        context = ''
        question = text[:end]

    # Use regex to find the options
    if 'options' not in text.lower():
        options = re.findall(r'[A-Z]\.', text)
    else:
        options = re.findall(r'[A-Z]\.', text[text.lower().find('options'):])
    print(options)
    options = [option for option in options if option.endswith('.')]
    options_text = [text[text.find(option)+2:text.find(options[i+1])] for i,option in enumerate(options[:-1])]
    options_text.append(text[text.find(options[-1])+2:])
    # if '\n' in options_text[-1]:
    #     context += options_text[-1].split('\n')[-1]
    #     options_text[-1] = '\n'.join(options_text[-1].split('\n')[:-1])
    # Format the options as a dictionary
    options_dict = {options[i].replace('.',''): options_text[i].strip() for i in range(len(options))}
    
    return {
        "context": context.strip(),
        "question": question.strip(),
        "options": options_dict
    }


# Helper function to check if a string contains Chinese characters
def cn_string(s):
    if re.search('[\u4e00-\u9fff]', s):
        return True
    return False


def mcq_re_prompt(question_text,model_name):
    # Extract question and options
    parsed = extract_question_and_options(question_text)
    question = parsed['question']
    options = parsed['options']
    
    # Prepare the options part
    options_prompt = 'Options:\n'
    for key, item in options.items():
        options_prompt += f'{key}. {item}\n'
    
    # Check if there's a hint (assuming it's passed in question_text as context)
    hint = parsed['context'] if parsed['context'] else None
    
    # Create the prompt
    prompt = ''
    if hint is not None:
        prompt += f'Hint: {hint}\n'
    prompt += f'Question: {question}\n'
    
    # Add options and determine if it should be in Chinese or English
    if len(options):
        prompt += options_prompt
        # Check if the prompt contains Chinese characters
        if model_name == 'llava':
            prompt += f'\n请直接回答选项字母。' if cn_string(prompt) else f'\nAnswer with the option\'s letter from the given choices directly.'
        else:
            prompt += f'请直接回答选项字母。例如{", ".join(options.keys())}' if cn_string(prompt) else f'Please select the correct answer from the options above. Eg. {", ".join(options.keys())}'
    
    prompt = prompt.rstrip()
    
    return prompt


def vqa_re_prompt(question_text,model_name):
    if model_name == 'llava':
        if cn_string(question_text):
            return f'{question_text}\n请直接回答问题。'
        else:
            return f'{question_text}\nAnswer the question directly.'
    return f'{question_text}\nPlease try to answer the question with short words or phrases if possible.'


def meta_re_prompt(question_text, model_name):
    if 'A.' in question_text:
        return mcq_re_prompt(question_text, model_name)
    else:
        return vqa_re_prompt(question_text, model_name)

@torch.inference_mode()
def critic_infer(original_question, image_url, reason_model_name, reason_model, reason_processor, critic_model_name, critic_model, critic_processor, device, chat_round=2, extra_prompt=""):
    conversation = []
    history_ans = []
    history_critic = []
    

    for rd in range(chat_round):
        if rd == 0:
            question = original_question
            question = meta_re_prompt(question, reason_model_name)
            
            if question is None:
                raise 99
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image":image_url,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            )
            res = gen_one_answer(reason_model_name, reason_model, reason_processor, image_url, conversation,device)
            history_ans.append(res)
            conversation.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": res},
                ],
            })
            if chat_round==1:
                break
            critic_question = f"""#### Question\n{question}\n#### Answer\n{res}\n#### Task\nPlease provide a critique of the answer above. What are the weaknesses of the answer?"""
            critic_conversation = [
                {
                    "role":"user",
                    "content":[
                        {
                            "type":"image",
                            "image":image_url,
                        },
                        {"type": "text", "text": critic_question},
                    ]
                }
            ]
            critic_res = gen_one_answer(critic_model_name, critic_model, critic_processor, image_url, critic_conversation, device)
            if 'correct' in critic_res.lower() and not 'incorrect' in critic_res.lower():
                break
            history_critic.append(critic_res)
            print(f'Question:\n{question}, \nRound {rd} done, \nCritic: {critic_res}, \nAnswer: {res}\n----------')


        elif rd>0 and rd < chat_round - 1:
            question = f"reflection on former answer:{history_critic[-1]}\n{original_question}" #+ "Let's think step by step."
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                ],
            })
            res = gen_one_answer(reason_model_name, reason_model,reason_processor, image_url,conversation,device)
            history_ans.append(res)
            conversation.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": res},
                ],
            })
            
            critic_question = f"""#### Question\n{original_question}\n#### Answer\n{res}\n#### Task\nPlease provide a critique of the answer above. What are the weaknesses of the answer?"""
            critic_conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": critic_question},
                ],
            }]
            critic_res = gen_one_answer(critic_model_name, critic_model, critic_processor, image_url, critic_conversation, device)
            history_critic.append(critic_res)
            print(f'Question:\n{question}\n, Round\n {rd} done, \nCritic:\n {critic_res}, \nAnswer:\n {res}\n-----------')
        elif rd > 0 and rd == chat_round - 1:
            question = f"reflection on former answer:{history_critic[-1]}\n{original_question}" #+ "Let's think step by step."
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                ],
            })
            res = gen_one_answer(reason_model_name, reason_model, reason_processor, image_url,conversation,device)
            history_ans.append(res)
            print(f'\nQuestion:\n{question},\n Round\n {rd} done, \nAnswer:\n {res}')



    return history_ans, history_critic


def setup_distributed(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    dist.init_process_group(
        backend = 'nccl', rank=rank,world_size=world_size
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())


def cleanup_distributed():
    dist.destroy_process_group()


def distributed_inference(rank, dataset, reason_model_name, reason_model_path, critic_model_name, critic_model_path, dataset_name, world_size, master_addr, master_port, chat_round, chunk_size, length_data):
    setup_distributed(rank, world_size, master_addr, master_port)
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    start_idx = rank*chunk_size
    end_idx = start_idx + chunk_size if rank != world_size - 1 else length_data
    batch = dataset[start_idx:end_idx]

    reason_model, reason_processor = load_model_and_tokenizer(reason_model_name, reason_model_path, device)
    critic_model, critic_processor = load_model_and_tokenizer(critic_model_name, critic_model_path, device)


    results = []
    for data in batch:
        print("data")
        image_url = data["images"][0]
        question = data['conversations'][0]['value']
        history_ans, history_critic = critic_infer(question, image_url, reason_model_name, reason_model, reason_processor, critic_model_name, critic_model, critic_processor, device, chat_round=chat_round)
        results.append({
            "id":data['id'],
            "images":image_url,
            "question":question,
            "history answer":history_ans,
            "history critic":history_critic,
            "ground_truth":data['conversations'][1]['value'],
        })

    os.makedirs(f'./results/{dataset_name}', exist_ok=True)
    output_file = f'./results/{dataset_name}/{reason_model_name}_{rank}.json'
    print(output_file)
    with open(output_file,'wt',encoding='utf-8') as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    cleanup_distributed()



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Run critic model on dataset with specific parameters.")
    parser.add_argument('--reason_model_name', type=str, required=True, help='Model name to use for reason')
    parser.add_argument('--reason_model_path', type=str, required=True, help='Model path to use for reason')
    parser.add_argument('--critic_model_name', type=str, required=True, help='The base model used for critic')
    parser.add_argument('--critic_model_path', type=str, required=True, help='Model path to use for critic')

    parser.add_argument('--dataset_name', type=str, required=True, help='The name of dataset')
    parser.add_argument('--dataset_path', type=str, required=True, help='The path of dataset')
    parser.add_argument('--world_size', type=int, required=True, help='The number of GPUs')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1', help='Communication address')
    parser.add_argument('--master_port', type=str, default='34567', help='Communication port')
    parser.add_argument('--chat_round', type=int, default=2, help='The Critique round')


    args = parser.parse_args()
    dataset = []

    with open(args.dataset_path,'r') as in_f:
        for line in in_f:
            data = json.loads(line)
            dataset.append(data)

    length_data = len(dataset)
    chunk_size = length_data // args.world_size

    spawn(
        distributed_inference,
        args=(dataset, args.reason_model_name, args.reason_model_path, args.critic_model_name, args.critic_model_path, args.dataset_name, args.world_size, args.master_addr, args.master_port, args.chat_round, chunk_size, length_data), 
        nprocs=args.world_size
        )


    