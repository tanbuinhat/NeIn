import requests
from PIL import Image
import torch
import transformers
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from transformers import BitsAndBytesConfig
transformers.logging.set_verbosity_error()
import json
import glob
from argparse import ArgumentParser
import os
import spacy



def check_obj_appear_in_nein_vqa(output_json, captions_data, nein_folder, nlp, model_id, device):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(model_id, device_map=device) 
    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map=device)

    json_dict = {entry['NeIn']: entry for entry in captions_data}

    with open(output_json, 'a') as json_file:
        json_file.write('[\n')  # Start the JSON array
        first_entry = True  # Flag to manage comma placement
        
        for image_name in sorted(os.listdir(nein_folder)):
            image_path = os.path.join(nein_folder, image_name)
            image_name = image_name.split('.')[0]
            
            if image_name in json_dict:
                entry = json_dict[image_name]
                image = Image.open(image_path).convert('RGB')
                
                doc = nlp(entry["T_original"])
                try:
                    objects = [chunk.root.text for chunk in doc.noun_chunks]
                    original_objects = []
                    for obj in objects:
                        prompt_object = "[INST] <image>\nDoes this image contain {object}? Answer only yes or no [/INST]"
                        prompt_object = prompt_object.format(object=obj)
                        inputs_object = processor(images=image, text=prompt_object, return_tensors="pt").to(device)
                        generate_ids_object = model.generate(**inputs_object, max_new_tokens = 10)
                        output_object = processor.batch_decode(generate_ids_object, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        answer_object = output_object[(output_object.find("[/INST]") + len("[/INST]")):].strip()
                        
                        if answer_object.lower() == "yes":
                            original_objects.append(obj)
                except:
                    original_objects = []
                    
                ori_obj = {"obj_in_nein": original_objects}
                entry.update(ori_obj)
                
                if not first_entry:
                    json_file.write(',\n')  # Add a comma before the new entry
                else:
                    first_entry = False  # After the first entry, set this to false
                
                json.dump(entry, json_file, indent=4)
            
        json_file.write('\n]')
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--output-json", type=str, help='json file to save output results')
    parser.add_argument("--nein-json", type=str, help='json of nein validation')
    parser.add_argument("--nein-folder", type=str, help='folder of NeIn validation set') #be careful this is check objects in orginal images are still in NeIn's sample or not, so this is NeIn validation path
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nlp = spacy.load("en_core_web_sm")
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    with open(args.nein_json, 'r') as json_file:
        captions_data = json.load(json_file)
        
    check_obj_appear_in_nein_vqa(args.output_json, captions_data, args.nein_folder, nlp, model_id, device)