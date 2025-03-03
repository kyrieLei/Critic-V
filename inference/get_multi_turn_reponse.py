from openai import OpenAI
from qwen_vl_utils import process_vision_info
import base64
import torch
from PIL import Image
from deepseek_vl.utils.io import load_pil_images
from internvl_util import load_image

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
  
def gpt_conversation_reform(conversation, image_url):
    image = encode_image(image_url)

    ret = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in conversation:
        ans = {}
        ans['role'] = 'user'
        ans['content'] = [
          {"type": "text",
            "text": msg['content'][-1]['text']}
            ]
        if msg['content'][0]['type'] == 'image':
            ans['content'].append({
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/png;base64,{image}"
                },
            })
        ret.append(ans)
    return ret

   
def get_gpt_response(model_name, image_url, conversation, api_key):

    client=OpenAI(
        api_key=api_key,
        base_url="https://api.claudeshop.top/v1",
    )
    conversation = gpt_conversation_reform(conversation, image_url)


    response = client.chat.completions.create(
        model=model_name,
        messages=conversation
    )

    result = response.choices[0].message.content
    return result
@torch.no_grad
def get_Qwen2VL_response(model, processor, image_url, conversation,device,max_new_tokens = 512):
    
    messages = conversation
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
def GLM_format_reform(conversation, image_url):

    image = Image.open(image_url).convert('RGB')
    ret = []
    for msg in conversation:
        ans = {}
        ans['role'] = 'user'
        if msg['content'][0]['type'] == 'image':
            ans['image'] = image
        ans['content'] = msg['content'][-1]['text']

        ret.append(ans)
    
    return ret
      

@torch.no_grad
def get_GLM_response(model, tokenizer, image_url, conversation, device):
   image = Image.open(image_url).convert('RGB')
   
   message = GLM_format_reform(conversation, image_url)
   
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
   
def internVL_conversation_reform(conversation):
    prompt = conversation[-1]['content'][-1]['text']
    history = [(conversation[i-1]['content'][-1]['text'], conversation[i]['content'][-1]['text']) for i in range(1, len(conversation),2)]

    print(f"History: {history}")
    
    return prompt, history
   
   
@torch.no_grad
def get_internVL_response(model, tokenizer, image_url, conversation,device):
    

    pixel_values = load_image(image_url).to(torch.bfloat16).to(device)

    generation_config = dict(max_new_tokens=1024, do_sample=True)
    

    prompt, history = internVL_conversation_reform(conversation)

    response, history = model.chat(tokenizer, pixel_values, prompt, generation_config, history=history, return_history=True)

    return response

def miniCPM_conversation_reform(conversation, image_url):
    image = Image.open(image_url).convert("RGB")
    ret = []
    for msg in conversation:
       ans = {}
       ans['role']='user'
       ans['content']=[image,msg['content'][-1]['text']]
       ret.append(ans)

    return ret
      
@torch.no_grad
def get_miniCPM_response(model, tokenizer, image_url, conversation, device):
   msgs = miniCPM_conversation_reform(conversation, image_url)

   result = model.chat(
      image=None,
      msgs=msgs,
      tokenizer=tokenizer
   )

   return result

def llava_conversation_reform(conversation):
    ret = []
    for msg in conversation:
        for i,item in enumerate(msg['content']):
            if item['type'] == 'text':
                item['text'] = item['text'].replace('<image>', '').strip()
            msg['content'][i] = item
        ret.append(msg)
    return ret
@torch.no_grad
def get_llava_response(model, processor, image_url, conversation, device):
   
   conversation = llava_conversation_reform(conversation)
   
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


def  deepseek_format_reform(conversation, image_url=None):
   ret = []
   for msg in conversation:
    ans = {}
    ans['role'] = 'User' if msg['role'] == 'user' else 'Assistant'
    ans['content'] = msg['content'][-1]['text'].replace('<image>', '<image_placeholder>').strip()
    if ret == [] and image_url is not None:
        ans['images'] = [image_url]
        if not ans['content'].startswith('<image_placeholder>'):
            ans['content'] = '<image_placeholder>' + ans['content']
    ret.append(ans)

    if ret[-1]['role'] == 'User':
        ret.append({'role': 'Assistant', 'content': ''})

    return ret
   
@torch.no_grad
def get_deepseek_response(model, processor, image_url, conversation, device):
   conversation = deepseek_format_reform(conversation, image_url)
   
   tokenizer = processor.tokenizer
   
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
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
   )
   answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
   torch.cuda.empty_cache()

   return answer
   
   
   

