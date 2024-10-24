import torch
from PIL import Image
import transformers
from transformers import BlipProcessor, BlipForImageTextRetrieval
transformers.logging.set_verbosity_error()
import random
import json
import glob
from collections import defaultdict
from pycocotools.coco import COCO
from argparse import ArgumentParser
import os


'''read coco annotation file for the captions'''
def read_captions_from_coco(file_path):
    captions_dict = defaultdict(list)
    coco = COCO(file_path)
    img_ids = coco.getImgIds()
    for img_id in sorted(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_name = img_info['file_name']
        caption_ids = coco.getAnnIds(imgIds=img_id)
        captions = coco.loadAnns(caption_ids)
        for caption in captions:
            captions_dict[img_name.strip()].append(caption['caption'].replace("\n", ""))
    return captions_dict


def get_determiner(word):
    vowels = 'aeiou'
    return 'an' if word[0].lower() in vowels else 'a'


def generation(model, objects, captions_dict, sentences, coco_images, device, output_path):
    json_results = []

    for image_path in sorted(os.listdir(coco_images)):
        raw_image = Image.open(image_path).convert('RGB')
        image_name = image_path.split("/")[-1]
        selected_objects = random.sample(objects, 15) #15 random object categories
        
        results = []
        for object in selected_objects:
            inputs = processor(raw_image, object, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            cosine_score = model(**inputs, use_itm_head=False)[0]
            results.append((object, cosine_score.item()))

        filtered_results = [result for result in results if result[1] < 0.4] #threshold 0.4
        top_5_objects = sorted(filtered_results, key=lambda x: x[1])[:5] #get top 5 categories with lowest scores
        top_5_objects = [result[0] for result in top_5_objects]
        
        # Create T_n (T_negative) and T_g (T_generated)
        T_original = captions_dict.get(image_name, []) #T_o
        T_negative = []
        T_generated = []
        for i, original_caption in enumerate(T_original[:5]):
            if "." not in original_caption: #ensure the original caption ends with a dot
                original_caption += '.'
            selected_object = top_5_objects[i]
            sentence_template = random.choice(sentences)
            negative_caption = sentence_template.format(object=selected_object)

            determiner = get_determiner(selected_object)
            generated_caption = "Add {article} {object}.".format(article=determiner, object=selected_object)
            
            if negative_caption.startswith(selected_object): #capitalize if the object is starting the caption
                negative_caption = negative_caption.replace(selected_object, selected_object.capitalize(), 1)

            T_negative.append(negative_caption)
            T_generated.append(generated_caption)
        
        json_results.append({
                    "image_name": image_name,
                    "original_caption": T_original,
                    "negative_caption": T_negative,
                    "generated_caption": T_generated
                })
        
        with open(output_path, 'w') as json_file:
            json.dump(json_results, json_file, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--object-file", default="label/coco_objects.txt", type=str, help='path to list coco objects')
    parser.add_argument("--coco-path", default="dataset/coco", type=str, help='path to coco image')
    parser.add_argument("--coco-annotation", default="dataset/annotations/captions.json", type=str, help='annotations of COCO')
    parser.add_argument("--output-json", default="jsons/generation.json", type=str, help='json file to store the image name and all clauses')
    args = parser.parse_args()

    coco_objects = args.object_file
    coco_annotation_file = args.coco_annotation
    coco_images = args.coco_path
    
    output_path = args.output_json
    if not os.path.exists(output_path.split("/")[0]):
        os.makedirs(output_path.split("/")[0])
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentences = [
        "The image doesn't have any {object}.",
        "The image does not have any {object}.",
        "The image cannot have any {object}.",
        "No {object} present in the image.",
        "No {object} in the image.",
        "The image is without {object}.",
        "Not a single {object} in sight.",
        "{object} is not part of the scene.",
        "The image lacks {object}.",
        "{object} is nowhere to be seen in the image.",
        "{object} is missing from the image.",
        "A scene without {object}.",
        "The image lacks the presence of {object}."
    ]
    
    '''use BLIP to check whether the objects in the image'''
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco")
    model.eval()
    model.to(device)
    
    '''read text file for the objects'''
    with open(coco_objects, 'r') as file:
        objects = file.read().splitlines()

    captions_dict = read_captions_from_coco(coco_annotation_file)
    
    generation(model, objects, captions_dict, sentences, coco_images, device, output_path)
    
    

