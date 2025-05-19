import requests
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection, pipeline
import json
import glob
from argparse import ArgumentParser
import os
import spacy


def check_obj_appear_in_nein_ovd(output_json, captions_data, nein_folder, nlp, checkpoint, device):
    detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device=device)
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
                    objects = [chunk.root.text for chunk in doc.noun_chunks] #detect objects in orginal captions by Spacy
                    
                    predictions = detector(
                        image,
                        candidate_labels=objects,
                    )
                    highest_scores = {}
                    for predict in predictions:
                        label = predict['label']
                        score = predict['score']
                        if label not in highest_scores or score > highest_scores[label]:
                            highest_scores[label] = score
                    
                    objects = list(highest_scores.keys())
                    # scores = list(highest_scores.values()) #not need scores from model for now
                except:
                    objects = []
                    # scores = []
                    
                ori_obj = {"obj_in_nein": objects}
                # score_dict = {"highest_score": scores}

                entry.update(ori_obj)
                # entry.update(score_dict)
                
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
    checkpoint = "google/owlv2-large-patch14-ensemble"
    nlp = spacy.load("en_core_web_sm")
    
    with open(args.nein_json, 'r') as json_file:
        captions_data = json.load(json_file)
    
    check_obj_appear_in_nein_ovd(args.output_json, captions_data, args.nein_folder, nlp, checkpoint, device)