import json 
import glob
import os
from argparse import ArgumentParser

def get_base_image_name(image_name):
    return ("_".join(image_name.split("_")[:-1])).split("/")[-1]


def format(input_data, images, output_json):
    captions_dict = {}
    for item in input_data:
        base_name = item['image_name'].rsplit('.', 1)[0]
        captions_dict[base_name] = {
            'original_caption': item['original_caption'],
            'generated_caption': item['generated_caption'],
            'negative_caption': item['negative_caption']
        }
    
    with open(output_json, 'a') as json_file:
        json_file.write('[\n')  # Start the JSON array
        first_entry = True  # Flag to manage comma placement

        for image_path in images:
            # import ipdb; ipdb.set_trace()
            image_name = image_path.split("/")[-1]
            base_name = get_base_image_name(image_path)

            if base_name in captions_dict:
                version_number = int(image_name.split("_")[-1].split(".")[0]) - 1

                original_caption = captions_dict[base_name]['original_caption'][version_number]
                if "." not in original_caption:
                    original_caption += '.'
                    
                generated_caption = captions_dict[base_name]['generated_caption'][version_number]
                negative_caption = captions_dict[base_name]['negative_caption'][version_number]

            json_results = {
                    "input": base_name,
                    "T_original": original_caption,
                    "T_generated": generated_caption,
                    "T_negative": negative_caption.split(original_caption)[1].strip(),
                    "output": image_name.split(".")[0]
                }
            
            # Write to JSON file
            if not first_entry:
                json_file.write(',\n')  # Add a comma before the new entry
            else:
                first_entry = False  # After the first entry, set this to false
            
            json.dump(json_results, json_file, indent=4)
        
        json_file.write('\n]')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-json", default="/home/ntnbinh/instruct-save/json/neginstruct_val_generate.json", type=str)
    parser.add_argument("--samples-path", default="/home/ntnbinh/instruct-save/neginstruct_val", type=str, help='path to nein samples')
    # parser.add_argument("--input-json", default="jsons/val_generation.json", type=str)
    # parser.add_argument("--samples-path", default="outputs/val_nein_samples", type=str, help='path to nein samples')
    parser.add_argument("--output-json", default="jsons/nein_val.json", type=str, help='final json file')
    args = parser.parse_args()

    output_json = args.output_json

    with open(args.input_json, 'r') as json_file:
        input_data = json.load(json_file)

    # image_files = sorted(glob.glob(args.samples_path))
    image_files = sorted(os.listdir(args.samples_path))
    
    format(input_data, image_files, output_json)


    