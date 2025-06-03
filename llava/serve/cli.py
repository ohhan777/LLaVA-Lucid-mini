import argparse
import requests
from PIL import Image
from io import BytesIO

import torch
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)

    # kwargs = {'device_map': 'auto', 'attn_implementation': 'sdpa', 'padding': True, 'truncation': True, 'max_length': 2048}   # ohhan

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name)


    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    elif "llama_3_1" in model_name.lower() or "llama-3.1" in model_name.lower() or "llama3.1" in model_name.lower():  # ohhan
        conv_mode = "llama_3_1"
    elif "qwen_2_5" in model_name.lower() or "qwen2.5" in model_name.lower():
        conv_mode = "qwen_2_5"
    elif "qwen_3" in model_name.lower() or "qwen3" in model_name.lower():
        conv_mode = "qwen_3"
    elif "phi_4" in model_name.lower() or "phi-4" in model_name.lower() or "phi4" in model_name.lower():
        conv_mode = "phi_4"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while --conv-mode is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    roles = conv.roles
  
    image = load_image(args.image_file)

    # check if vision tower is naflex
    naflex_model = False
    if "naflex" in args.model_path.lower():
        print("[INFO] Naflex model detected")
        naflex_model = True
        processed_tensor = image_processor.preprocess(image, return_tensors="pt")
        image_tensor = processed_tensor['pixel_values'].half().cuda()
        pixel_attention_mask = processed_tensor['pixel_attention_mask'].cuda()
        spatial_shapes = processed_tensor['spatial_shapes'].cuda()
    else:
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().cuda()


    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if conv_mode == "llama_3_1":
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            stop_str = "<|eot_id|>"
            tokenizer.pad_token_id = 128004
        elif conv_mode == "qwen_2_5": 
            prompt += "<|im_start|>assistant\n"
            stop_str = "<|im_end|>"
            tokenizer.pad_token_id = 151662
        elif conv_mode == "qwen_3":
            prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            stop_str = "<|im_end|>"
            tokenizer.pad_token_id = 151662
        elif conv_mode == "phi_4":
            prompt += "<|im_start|>assistant<|im_sep|>"
            stop_str = "<|im_end|>"
            tokenizer.pad_token_id = 100349
        else:
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            pass
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            if naflex_model:
                output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes, temperature=args.temperature, max_new_tokens=args.max_new_tokens, 
                                        streamer=streamer, use_cache=True, pad_token_id=tokenizer.pad_token_id, stopping_criteria=[stopping_criteria])
            else:
                output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=args.temperature, max_new_tokens=args.max_new_tokens, 
                                        streamer=streamer, use_cache=True, pad_token_id=tokenizer.pad_token_id, stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0]).strip()
        
                
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="./checkpoints/llava-phi-4-s2-finetune-kompsat")
    # parser.add_argument("--model-path", type=str, default="./checkpoints/llava-Qwen2.5-7B-Instruct-s2-finetune-kompsat")
    parser.add_argument("--model-path", type=str, default="./checkpoints/llava-Qwen3-14B-s2-finetune-kompsat")
    # parser.add_argument("--model-path", type=str, default="./checkpoints/llava-Llama-3.1-8B-Instruct-s2-finetune-kompsat")
    # parser.add_argument("--model-path", type=str, default="./checkpoints/llava-Llama-3.1-8B-Instruct-siglip2-finetune-kompsat")
    # parser.add_argument("--model-path", type=str, default="./checkpoints/llava-Llama-3.1-8B-Instruct-finetune-next-kompsat")
    # parser.add_argument("--model-path", type=str, default="./checkpoints/llava-Llama-3.1-8B-Instruct-siglip2-naflex-finetune-kompsat")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="./playground/data/kari_objects_33cls/test/images/png/OBJ00002_PS3_K3_AIDATA0001.png")  # required=True
    parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--conv-mode", type=str, default="qwen_2_5")
    parser.add_argument("--conv-mode", type=str, default="qwen_3")
    # parser.add_argument("--conv-mode", type=str, default="phi_4")
    #parser.add_argument("--conv-mode", type=str, default="llama_3_1")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)