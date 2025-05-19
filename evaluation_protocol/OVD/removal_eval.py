import requests
from argparse import ArgumentParser
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection, pipeline
import json
import glob
import os


def get_base_image_name(image_name):
    return ("_".join(image_name.split("_")[:-1])).split("/")[-1]

def removal_evaluation_ovd(output_json, captions_data, edited_folder, checkpoint, device):
    detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device=device)
    json_dict = {entry['NeIn']: entry for entry in captions_data}

    with open(output_json, 'a') as json_file:
        json_file.write('[\n')  
        first_entry = True  # Flag to manage comma placement

        for image_name in sorted(os.listdir(edited_folder)):
            image_path = os.path.join(edited_folder, image_name)
            image_name = image_name.split('.')[0]
            
            if image_name in json_dict:
                image = Image.open(image_path).convert('RGB')
                element = json_dict[image_name]
                parts = element["T_generated"].split()
                object_remove = " ".join(parts[2:]).rstrip('.')
                
                predictions = detector(
                    image,
                    candidate_labels=[object_remove],
                )
                
                if len(predictions) == 0: 
                    #removed
                    a_dict = {'removal_evaluation_ovd': 1}
                    element.update(a_dict)
                else:
                    #retained
                    a_dict = {'removal_evaluation_ovd': 0}
                    element.update(a_dict)
                    
                    highest_score = max(item['score'] for item in predictions)
                    score_dict = {'highest_confidence_score': highest_score}
                    element.update(score_dict)
                
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
    checkpoint = "google/owlv2-large-patch14-ensemble"

    with open(args.nein_json, 'r') as json_file:
        captions_data = json.load(json_file)
    
    removal_evaluation_ovd(args.output_json, captions_data, args.edited_folder, checkpoint, device)
    
    with open(args.output_json, 'r') as file:
        results = json.load(file)
        
    correct = 0

    for sublist in results:
        if sublist['removal_evaluation_ovd'] == 1:
            correct += 1
        else:
            correct += 0
            
    print("Number of correct:", correct)
    print("Accuracy for removal evaluation by OVD:", correct/24182) #24182 is the number of validation set
    
    