import requests
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection, pipeline
import json
import glob
from argparse import ArgumentParser
import os


def get_base_image_name(image_name):
    return ("_".join(image_name.split("_")[:-1])).split("/")[-1]


def retention_evaluation_ovd(output_json, retention_data, edited_folder, checkpoint, device):
    detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device=device)
    json_dict = {entry['NeIn']: entry for entry in retention_data}

    with open(output_json, 'w') as json_file:
        json_file.write('[\n')  # Start the JSON array
        first_entry = True  # Flag to manage comma placement
        
        for image_name in sorted(os.listdir(edited_folder)):
            image_path = os.path.join(edited_folder, image_name)
            image_name = image_name.split('.')[0]
            
            if image_name in json_dict:
                # import ipdb; ipdb.set_trace()
                image = Image.open(image_path).convert('RGB')
                element = json_dict[image_name]
                retention_obj = element['obj_in_nein']
                try:
                    predictions = detector(
                        image,
                        candidate_labels=retention_obj,
                    )
                    
                    #must sum the scores and divide by the total number of objects in the image to get the score
                    highest_scores = {label: 0 for label in retention_obj}
                    for predict in predictions:
                        label = predict['label']
                        score = predict['score']
                        if label not in highest_scores or score > highest_scores[label]:
                            highest_scores[label] = score
                    
                    objects = list(highest_scores.keys())
                    scores = list(highest_scores.values())
                    #calculate total_score
                    num_labels_with_scores = sum(1 for score in scores if score > 0)
                except: 
                    objects = []
                    scores = []
                    num_labels_with_scores = None
                    
                # score_obj = {"second_score_objects": scores}
                correct_retention = {"correct_retention_obj": num_labels_with_scores}
                
                if image_name in json_dict:
                    edit = json_dict[image_name]
                    # edit.update(score_obj)
                    edit.update(correct_retention)
                        
                    if not first_entry:
                        json_file.write(',\n')  # Add a comma before the new entry
                    else:
                        first_entry = False  # After the first entry, set this to false
                        
                    json.dump(edit, json_file, indent=4)
                    
        json_file.write('\n]')
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--output-json", type=str, help='json file to save output results')
    parser.add_argument("--retention-json", type=str, help='output json of checking objects in NeIn samples of OVD')
    parser.add_argument("--edited-folder", type=str, help='folder of edited images')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "google/owlv2-large-patch14-ensemble"
    
    with open(args.retention_json, 'r') as json_file:
        retention_data = json.load(json_file)
        
    retention_evaluation_ovd(args.output_json, retention_data, args.edited_folder, checkpoint, device)
    
    with open(args.output_json, 'r') as file:
        results = json.load(file)
    
    scores = []
    for sublist in results:
        try:
            score_sublist = sublist['correct_retention_obj'] / len(sublist['obj_in_nein'])
            scores.append(score_sublist)
        except:
            continue
        
    acc_score = sum(scores) / 24120 # 24021 = 24182 - 62, 62 is the number of instances where no objects were detected with NeIn (empty in the retention json).

    print("Accuracy for retention evaluation by OVD:", acc_score)

