from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel, AutoModelForCausalLM,LlavaForConditionalGeneration
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from get_multi_turn_reponse import(
    get_Qwen2VL_response,
    get_GLM_response,
    get_internVL_response,
    get_miniCPM_response,
    get_llava_response,
    get_deepseek_response
)

def load_model_and_tokenizer(model_name, model_path, device):

    if model_name == 'qwen2-vl':
        model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,torch_dtype = "auto"
    ).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_path)
        return model,processor
    
    elif model_name == 'chatglm':
        model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True, torch_dtype='auto').to(device).eval()
        tokenzier = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

        return model,tokenzier

    
    elif model_name == 'intern-vl':
        model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        return model,tokenizer

    elif model_name == 'minicpm-v':
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
    attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model,tokenzier

    elif model_name == 'llava':
        model = LlavaForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device).eval()
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        return model,processor


    elif model_name == 'deepseek-vl':
        processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
        return model,processor

    else:
        print("This Model is still not implemented🥹")
        return None


@torch.inference_mode()
def gen_one_answer(model_name , model, processor, image_url, conversation, device):
    if model_name == 'qwen2-vl':
        ans = get_Qwen2VL_response(model, processor, image_url, conversation, device)
        return ans
    
    elif model_name == 'chatglm':
        ans = get_GLM_response(model, processor, image_url, conversation, device)
        return ans

    
    elif model_name == 'intern-vl':
        ans = get_internVL_response(model,processor, image_url, conversation, device)
        return ans

    elif model_name == 'minicpm-v':
        ans = get_miniCPM_response(model, processor, image_url, conversation, device)
        return ans

    elif model_name == 'llava':
        ans = get_llava_response(model, processor, image_url, conversation, device)
        return ans

    elif model_name == 'deepseek-vl':
        
        ans = get_deepseek_response(model, processor, image_url, conversation, device)
        return ans

    else:
        print("This Model is still not implemented🥹")
        return None