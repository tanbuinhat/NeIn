import torch
from PIL import Image
import transformers
from transformers import BlipProcessor, BlipForImageTextRetrieval
transformers.logging.set_verbosity_error()
import json
import glob
from argparse import ArgumentParser

def get_base_image_name(image_name):
    return ("_".join(image_name.split("_")[:-1])).split("/")[-1]


def filtering(model, clause_file, generated_samples, filtering_json):
    captions_dict = {}
    for item in clause_file:
        base_name = item['image_name'].rsplit('.', 1)[0]
        captions_dict[base_name] = {
            'original_caption': item['original_caption'],
            'generated_caption': item['generated_caption']
        }
        
    json_results = []
        
    for image_path in generated_samples:
        image_name = image_path.split("/")[-1]
        raw_image = Image.open(image_path).convert('RGB')
        
        base_name = get_base_image_name(image_path)
        if base_name in captions_dict:
            # Extract the version number (1, 2, 3, 4, or 5)
            version_number = int(image_name.split("_")[-1].split(".")[0]) - 1

            original_caption = captions_dict[base_name]['original_caption'][version_number]
            generated_caption = captions_dict[base_name]['generated_caption'][version_number]
            
        extracted_object = generated_caption.split("Add")[-1]

        if "." not in original_caption:
            positive_caption = original_caption + ". " + "This image has{object}".format(object=extracted_object)
        else:
            positive_caption = original_caption + " " + "This image has{object}".format(object=extracted_object)
        
        
        inputs = processor(raw_image, positive_caption, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        cosine_score = model(**inputs, use_itm_head=False)[0][0]

        
        if cosine_score < 0.4: # threshold 0.4
            json_results.append({
                    "image_name": image_name,
                    "original_caption": original_caption,
                    "object": extracted_object,
                    "generated_caption": generated_caption,
                    "cosine": cosine_score.cpu().detach().numpy().item()
                })
            # Save results to a JSON file
            with open(filtering_json, 'w') as json_file:
                json.dump(json_results, json_file, indent=4)

        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-json", default="jsons/val_generation.json", type=str, help='path to the clause file from clause_generation')
    parser.add_argument("--samples-path", default="outputs/val_nein_samples/*.jpg", type=str, help='path to generated samples')
    parser.add_argument("--output-json", default="jsons/val_filtering.json", type=str, help='json file to store the names of filtered images')
    args = parser.parse_args()

    with open(args.input_json, 'r') as json_file:
        clause_file = json.load(json_file)

    generated_samples = sorted(glob.glob(args.sample_path))
    filtering_json = args.output_json
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
    model.eval()
    model.to(device)

    filtering(model, clause_file, generated_samples, filtering_json)