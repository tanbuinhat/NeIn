import os
import json
import shutil
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filtering-json", default="jsons/filtering.json", type=str, help='json file stored the names of filtered samples')
    parser.add_argument("--samples-path", default="outputs/nein_samples", type=str, help='path to generated samples')
    parser.add_argument("--filtered-samples-path", default="outputs/filtered_samples", type=str, help='path to folder saved filtered samples, this samples will be removed')
    args = parser.parse_args()
    
    if not os.path.exists(args.filtered_samples_path):
        os.makedirs(args.filtered_samples_path)
        
    # Read the JSON file
    with open(args.filtering_json, 'r') as file:
        data = json.load(file)
        images_to_delete = [item["image_name"] for item in data]
    
    for image_name in sorted(os.listdir(args_samples_path)):
        if image_name in images_to_delete:
            # Construct full file paths
            src = os.path.join(args.samples_path, image_name)
            dst = os.path.join(args.filtered_samples_path, image_name)
            
            # Move the image from folder B to folder A
            shutil.move(src, dst)
