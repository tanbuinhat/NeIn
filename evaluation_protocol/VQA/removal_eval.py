import requests
from PIL import Image
import torch
from argparse import ArgumentParser
import transformers
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from transformers import BitsAndBytesConfig
transformers.logging.set_verbosity_error()
import json
import glob
import os


def get_base_image_name(image_name):
    return ("_".join(image_name.split("_")[:-1])).split("/")[-1]

def removal_evaluation_vqa(output_json, captions_data, edited_folder, model_id, device):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained(model_id, device_map=device) 
    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map=device)
    json_dict = {entry['NeIn']: entry for entry in captions_data}

    with open(output_json, 'a') as json_file:
        json_file.write('[\n')  
        first_entry = True  # Flag to manage comma placement
        
        for image_name in sorted(os.listdir(edited_folder)):
            image_path = os.path.join(edited_folder, image_name)
            image_name = image_name.split('.')[0]

            if image_name in json_dict:
                element = json_dict[image_name]
                image = Image.open(image_path).convert('RGB') #image editing's output

                parts = element["T_generated"].split()
                object_remove = " ".join(parts[2:]).rstrip('.')

                prompt_object = "[INST] <image>\nDoes this image contain {object}? Answer only yes or no [/INST]"
                prompt_object = prompt_object.format(object=object_remove)

                inputs_object = processor(images=image, text=prompt_object, return_tensors="pt").to(device)
                generate_ids_object = model.generate(**inputs_object, max_new_tokens = 10)
                
                output_object = processor.batch_decode(generate_ids_object, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                answer_object = output_object[(output_object.find("[/INST]") + len("[/INST]")):].strip()
                
                if answer_object == "No": 
                    a_dict = {'removal_evaluation_vqa': 1}
                    element.update(a_dict)
                    
                else:
                    a_dict = {'removal_evaluation_vqa': 0}
                    element.update(a_dict)
                
                # Write to JSON file
                if not first_entry:
                    json_file.write(',\n')  # Add a comma before the new entry
                else:
                    first_entry = False  # After the first entry, set this to false
                
                json.dump(element, json_file, indent=4)
        
        json_file.write('\n]')
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--output-json", type=str, help='json file to save output results')
    parser.add_argument("--nein-json", type=str, help='json of nein validation')
    parser.add_argument("--edited-folder", type=str, help='folder of edited images')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    with open(args.nein_json, 'r') as json_file:
        captions_data = json.load(json_file)
        
    removal_evaluation_vqa(args.output_json, captions_data, args.edited_folder, model_id, device)
    
    with open(args.output_json, 'r') as file:
        results = json.load(file)
    
    correct = 0

    for sublist in results:
        if sublist['removal_evaluation_vqa'] == 1:
            correct += 1
        else:
            correct += 0
            
    print("Number of correct:", correct)
    print("Accuracy for removal evaluation by VQA:", correct/24182) #24182 is the number of validation set