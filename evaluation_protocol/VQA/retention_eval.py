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



def get_base_image_name(image_name):
    return ("_".join(image_name.split("_")[:-1])).split("/")[-1]

def retention_evaluation_vqa(output_json, retention_data, edited_folder, model_id, device):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained(model_id, device_map=device) 
    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map=device)
    json_dict = {entry['NeIn']: entry for entry in retention_data}

    with open(output_json, 'w') as json_file:
        json_file.write('[\n')  # Start the JSON array
        first_entry = True  # Flag to manage comma placement
        
        for image_name in sorted(os.listdir(edited_folder)):
            image_path = os.path.join(edited_folder, image_name)
            image_name = image_name.split('.')[0]
            
            if image_name in json_dict:
                image = Image.open(image_path).convert('RGB')
                element = json_dict[image_name]

                try:
                    objects_retains = element['obj_in_nein']
                    correct_objects = []
                    for obj in objects_retains:
                        prompt_object = "[INST] <image>\nDoes this image contain {object}? Answer only yes or no [/INST]"
                        prompt_object = prompt_object.format(object=obj)
                        inputs_object = processor(images=image, text=prompt_object, return_tensors="pt").to(device)
                        generate_ids_object = model.generate(**inputs_object, max_new_tokens = 10)
                        output_object = processor.batch_decode(generate_ids_object, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        answer_object = output_object[(output_object.find("[/INST]") + len("[/INST]")):].strip()
                        
                        if answer_object.lower() == "yes":
                            correct_objects.append(obj)
                except: 
                    correct_objects = []
                    
                ori_obj = {"correct_retention_obj": correct_objects}
                element.update(ori_obj)
            
                        
                if not first_entry:
                    json_file.write(',\n')  # Add a comma before the new entry
                else:
                    first_entry = False  # After the first entry, set this to false
                    
                json.dump(element, json_file, indent=4)
                    
        json_file.write('\n]')
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--output-json", type=str, help='json file to save output results')
    parser.add_argument("--retention-json", type=str, help='output json of checking objects in NeIn samples of VQA')
    parser.add_argument("--edited-folder", type=str, help='folder of edited images')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    with open(args.retention_json, 'r') as json_file:
        retention_data = json.load(json_file)
    
    retention_evaluation_vqa(args.output_json, retention_data, args.edited_folder, model_id, device)
    
    scores = []
    for sublist in results:
        try:
            score_sublist = sublist['correct_retention_obj'] / len(sublist['obj_in_nein'])
            scores.append(score_sublist)
        except:
            continue
        
    acc_score = sum(scores) / 24120 # 24021 = 24182 - 62, 62 is the number of instances where no objects were detected with NeIn (empty in the retention json).

    print("Accuracy for retention evaluation by OVD:", acc_score)

    
  