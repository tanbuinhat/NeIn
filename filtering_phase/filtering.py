import torch
from PIL import Image
import transformers
from transformers import BlipProcessor, BlipForImageTextRetrieval
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from transformers import BitsAndBytesConfig
transformers.logging.set_verbosity_error()
import json
import glob
import re
from argparse import ArgumentParser
import os


def get_base_image_name(image_name):
    return ("_".join(image_name.split("_")[:-1])).split("/")[-1]


def filtering(output_json, generation_dict, generation_samples, blip_processor, blip_model, llava_processor, llava_model, device):
    json_results = []
    for image_name in sorted(os.listdir(generation_samples)):
        image_path = os.path.join(generation_samples, image_name)
        image_name = image_name.split('.')[0]
        raw_image = Image.open(image_path).convert('RGB')
    
        base_name = get_base_image_name(image_path).replace("NeIn", "COCO")
        # Extract the version number (1, 2, 3, 4, or 5)
        version_number = int(image_name.split("_")[-1].split(".")[0]) - 1 #since the order in generation starts at 1

        original_caption = generation_dict[base_name]['T_original'][version_number]
        generated_caption = generation_dict[base_name]['T_generated'][version_number]
        
        extracted_object = generated_caption.split("Add")[-1]

        if "." not in original_caption:
            positive_caption = original_caption + ". " + "This image has{object}".format(object=extracted_object)
        else:
            positive_caption = original_caption + " " + "This image has{object}".format(object=extracted_object)
        
        #BLIP
        inputs = blip_processor(raw_image, positive_caption, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        cosine_score = blip_model(**inputs, use_itm_head=False)[0][0]
        
        if cosine_score > 0.4: # threshold 0.4, note this sample pass the BLIP check (BLIP accept) and go to the LLaVA-NeXT check
            # LLaVA-NeXT
            prompt_caption = "[INST] <image>\nDoes the caption \"{caption}\" describe this image? Answer only yes or no [/INST]"
            prompt_object = "[INST] <image>\nDoes this image contain{object}? Answer only yes or no [/INST]"
    
            prompt_caption = prompt_caption.format(caption=original_caption)
            prompt_object = prompt_object.format(object=extracted_object.replace(".", ""))
            
            inputs_caption = llava_processor(images=raw_image, text=prompt_caption, return_tensors="pt").to(device)
            inputs_object = llava_processor(images=raw_image, text=prompt_object, return_tensors="pt").to(device)
            
            generate_ids_caption = llava_model.generate(**inputs_caption, max_new_tokens = 10)
            generate_ids_object = llava_model.generate(**inputs_object, max_new_tokens = 10)
            
            output_caption = llava_processor.batch_decode(generate_ids_caption, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            answer_caption = output_caption[(output_caption.find("[/INST]") + len("[/INST]")):].strip()
            
            output_object = llava_processor.batch_decode(generate_ids_object, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            answer_object = output_object[(output_object.find("[/INST]") + len("[/INST]")):].strip()
            # import ipdb; ipdb.set_trace()
            if answer_caption == "Yes" and answer_object == "Yes":
                #pass all conditions
                json_results.append({
                    "COCO": base_name,
                    "T_original": original_caption,
                    "T_negative": generation_dict[base_name]['T_negative'][version_number],
                    "T_generated": generated_caption, 
                    "NeIn": generation_dict[base_name]['NeIn'][version_number]
                })
                
        # accepted samples for final NeIn's samples       
        with open(output_json, 'w') as json_file:
            json.dump(json_results, json_file, indent=4)
                

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--output-json", type=str, help='json file to save output results after filtering')
    parser.add_argument("--generation-json", type=str, help='json of nein generation step')
    parser.add_argument("--generation-samples", type=str, help='samples from generation step')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #BLIP
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
    blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco")
    blip_model.eval()
    blip_model.to(device)
    
    #LLaVA-NeXT
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    llava_model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    llava_processor = AutoProcessor.from_pretrained(llava_model_id, device_map=device) 
    llava_model = LlavaNextForConditionalGeneration.from_pretrained(llava_model_id, quantization_config=quantization_config, device_map=device)
    
    with open(args.generation_json, 'r') as json_file:
        generation_data = json.load(json_file)
        
    generation_dict = {}
    for item in generation_data:
        base_name = item['COCO']
        generation_dict[base_name] = {
            'T_original': item['T_original'],
            'T_negative': item['T_negative'], 
            'T_generated': item['T_generated'], 
            'NeIn': item['NeIn']
        }
    
    filtering(args.output_json, generation_dict, args.generation_samples, blip_processor, blip_model, llava_processor, llava_model, device)
        
                
        

