from openai import OpenAI
from qwen_vl_utils import process_vision_info
import base64
import torch
from PIL import Image
from deepseek_vl.utils.io import load_pil_images
from internvl_util import load_image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel, AutoModelForCausalLM,LlavaForConditionalGeneration
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM


generate_kwargs = dict(
    max_new_tokens=1024,
    top_p=0.001,
    top_k=1,
    temperature=0.01,
    repetition_penalty=1.0,
)

generate_kwargs_llava = dict(
        do_sample=False,
        temperature=0,
        max_new_tokens=512,
        top_p=None,
        num_beams=1,
        use_cache=True,
    )

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
def get_gpt_response(model_name, prompt, image_url, api_key):

    client=OpenAI(
        api_key=api_key,
        base_url="",
    )

    image = encode_image(image_url)
    if image_url is None:
       msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt,
            },
        ],
        }
    ],
    else:
       msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt,
            },
            {
            "type": "image_url",
            "image_url": {
                "url":  f"data:image/png;base64,{image}"
            },
            },
        ],
        }
    ],
       
    response = client.chat.completions.create(
    model=model_name,
    messages=msgs
    )

    result = response.choices[0].message.content
    return result
@torch.no_grad
def get_Qwen2VL_response(model, processor, prompt, image_url,device,max_new_tokens = 512):
    
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_url,
            },
            {"type": "text", "text": prompt},
        ],
    }
    ]
    text = processor.apply_chat_template(
       messages,tokenize=False, add_generation_prompt=True
    )


    image_inputs,video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


    input_ids = inputs.to(device)
    output_ids = model.generate(**input_ids, max_new_tokens = max_new_tokens)
    output = [
       out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids.input_ids,output_ids)
    ]

    result = processor.batch_decode(
       output, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return result
@torch.no_grad
def get_GLM_response(model, tokenizer, prompt, image_url, device):
   image = Image.open(image_url).convert('RGB')
   message = [
      {
         "role":"user",
         "image":image,
         "content":prompt
      }
   ]
   inputs = tokenizer.apply_chat_template(
      message,add_generation_prompt=True,tokenize=True,return_tensors='pt',return_dict=True
   )
   inputs = inputs.to(device)
   with torch.no_grad():
      outputs = model.generate(**inputs,**generate_kwargs)
      outputs = outputs[:,inputs['input_ids'].shape[1]:]
      result = tokenizer.decode(
         outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
      )
      return result
@torch.no_grad
def get_internVL_response(model, tokenizer, prompt, image_url,device):

    pixel_values = load_image(image_url).to(torch.bfloat16).to(device)

    generation_config = dict(max_new_tokens=1024, do_sample=True)
    prompt = f"<image>\n{prompt}"

    response, history = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)

    return response
@torch.no_grad
def get_miniCPM_response(model, tokenizer, prompt, image_url, device):
   
   image = Image.open(image_url).convert("RGB")
   msgs = [
      {
         "role":"user",
         'content':[image,prompt]
      }
   ]
   result = model.chat(
      image=None,
      msgs=msgs,
      tokenizer=tokenizer
   )

   return result
@torch.no_grad
def get_llava_response(model, processor, prompt, image_url, device):
   
   conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_url,
            },
            {"type": "text", "text": prompt},
        ],
    }
    ]
   image = Image.open(image_url).convert('RGB')
   
   text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
   
   inputs = processor(text=[text_prompt], images=image, padding=True, return_tensors="pt")
   inputs = inputs.to(device)
   output_ids = model.generate(**inputs, **generate_kwargs_llava)
   torch.cuda.empty_cache()
   generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
   res = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
   
   return res
@torch.no_grad
def get_deepseek_response(model, processor, prompt, image_url, device):
   
   tokenizer = processor.tokenizer
   conversation = [
    {
        "role": "User",
        "content": f"<image_placeholder>{prompt}",
        "images": [image_url]
    },
    {
        "role": "Assistant",
        "content": ""
    }
    ]
   images = load_pil_images(conversation)
   prepare_inputs = processor(
      conversations=conversation,
      images=images,
      force_batchify=True
   ).to(device)

   input_ids = model.prepare_inputs_embeds(**prepare_inputs)
   outputs = model.language_model.generate(
        inputs_embeds=input_ids,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=1024,
        do_sample=False,
        use_cache=True
   )
   answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

   return answer
   
   
   




@torch.inference_mode()
def gen_one_answer(model_name, model_path, conversation, device):
    if model_name == 'qwen2-vl':
        model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,torch_dtype = "auto", device_map='auto'
    ).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_path)
        ans = get_Qwen2VL_response(model,processor, conversation, device)
        return ans
    
    elif model_name == 'chatglm':
        model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True, torch_dtype='auto', device_map='auto').to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

        ans = get_GLM_response(model, tokenizer, conversation, device)

        return ans

    elif model_name == 'intern-vl':
        model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        ans = get_internVL_response(model, tokenizer, conversation, device)

        return ans

    elif model_name == 'minicpm-v':
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
    attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        ans = get_miniCPM_response(model, tokenizer, conversation, device)

        return ans

    elif model_name == 'llava':
        model = LlavaForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto', attn_implementation="flash_attention_2").to(device).eval()
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        ans = get_llava_response(model, processor, conversation, device)

        return ans


    elif model_name == 'deepseek-vl':
        processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()

        ans = get_deepseek_response(model, processor, conversation, device)
        return ans

    else:
        print("This Model is still not implemented🥹")
        return None