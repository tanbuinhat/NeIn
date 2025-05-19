"""
from v2 -> v3: add CLIP Score and CLIP I.
"""

'''https://github.com/OSU-NLP-Group/MagicBrush/blob/main/evaluation/image_eval.py'''

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import time
import torch
import clip
import json
import os
import io

from tqdm import tqdm
from PIL import Image
from scipy import spatial
from torch import nn
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF

def imread(img_source, rgb=False, target_size=None):
    """Read image"""
    if isinstance(img_source, bytes):
        img = Image.open(io.BytesIO(img_source))
    elif isinstance(img_source, str):
        assert os.path.isfile(img_source), f'{img_source} is not a valid image file.'
        img = Image.open(img_source)
    elif isinstance(img_source, Image.Image):
        img = img_source
    else:
        raise Exception("Unsupported source type")
    if rgb:
        img = img.convert('RGB')
    if target_size is not None:
        img = img.resize(target_size, Image.BICUBIC)
    return img

def eval_distance(image_pairs, metric='l1'):
    """Evaluate l1 or l2 distance"""
    if metric == 'l1':
        criterion = nn.L1Loss()
    elif metric == 'l2':
        criterion = nn.MSELoss()
    eval_score = 0
    for img_pair in tqdm(image_pairs):
        gen_img = Image.open(img_pair[0]).convert('RGB')
        gt_img = Image.open(img_pair[1]).convert('RGB')
        # resize to gt size
        gen_img = gen_img.resize(gt_img.size)
        # convert to tensor
        gen_img = transforms.ToTensor()(gen_img)
        gt_img = transforms.ToTensor()(gt_img)
        # calculate distance
        per_score = criterion(gen_img, gt_img).detach().cpu().numpy().item()
        eval_score += per_score
    return eval_score / len(image_pairs)

def eval_clip_i(args, image_pairs, model, transform, metric='clip_i'):
    """Calculate CLIP-I score"""
    def encode(image, model, transform):
        image_input = transform(image).unsqueeze(0).to(args.device)
        with torch.no_grad():
            if metric == 'clip_i':
                image_features = model.encode_image(image_input).detach().cpu().float()
            elif metric == 'dino':
                image_features = model(image_input).detach().cpu().float()
        return image_features
    eval_score = 0
    for img_pair in tqdm(image_pairs):
        generated_features = encode(Image.open(img_pair[0]).convert('RGB'), model, transform)
        gt_features = encode(Image.open(img_pair[1]).convert('RGB'), model, transform)
        similarity = 1 - spatial.distance.cosine(generated_features.view(generated_features.shape[1]),
                                                 gt_features.view(gt_features.shape[1]))
        if similarity > 1 or similarity < -1:
            raise ValueError("Strange similarity value")
        eval_score = eval_score + similarity
    return eval_score / len(image_pairs)

def eval_clip_score(args, image_pairs, clip_metric, caption_dict):
    """Calculate CLIP score"""
    trans = transforms.Compose([
        transforms.Resize(256),  # scale to 256x256
        transforms.CenterCrop(224),  # crop to 224x224
        transforms.ToTensor(),  # convert to pytorch tensor
    ])

    def clip_score(image_path, caption):
        image = Image.open(image_path).convert('RGB')
        image_tensor = trans(image).to(args.device)
        return clip_metric(image_tensor, caption).detach().cpu().float()
    
    gen_clip_score = 0
    gt_clip_score = 0
    
    for img_pair in tqdm(image_pairs):
        gen_img_path = img_pair[0]
        gt_img_path = img_pair[1]
        gt_img_name = gt_img_path.split('/')[-1]
        gt_caption = caption_dict.get(gt_img_name, "")
        gen_clip_score += clip_score(gen_img_path, gt_caption)
        gt_clip_score += clip_score(gt_img_path, gt_caption)
    
    return gen_clip_score / len(image_pairs), gt_clip_score / len(image_pairs)

def eval_clip_t(args, image_pairs, model, transform, caption_dict):
    """Calculate CLIP-T score"""
    def encode(image, model, transform):
        image_input = transform(image).unsqueeze(0).to(args.device)
        with torch.no_grad():
            image_features = model.encode_image(image_input).detach().cpu().float()
        return image_features
    gen_clip_t = 0
    gt_clip_t = 0
    
    for img_pair in tqdm(image_pairs):
        gen_img_path = img_pair[0]
        gt_img_path = img_pair[1]
        gt_img_name = gt_img_path.split('/')[-1]
        gt_caption = caption_dict.get(gt_img_name, "")
        generated_features = encode(Image.open(gen_img_path).convert('RGB'), model, transform)
        gt_features = encode(Image.open(gt_img_path).convert('RGB'), model, transform)
        text_features = clip.tokenize(gt_caption).to(args.device)
        with torch.no_grad():
            text_features = model.encode_text(text_features).detach().cpu().float()

        gen_clip_t += 1 - spatial.distance.cosine(generated_features.view(generated_features.shape[1]),
                                                    text_features.view(text_features.shape[1]))
        gt_clip_t += 1 - spatial.distance.cosine(gt_features.view(gt_features.shape[1]),
                                                    text_features.view(text_features.shape[1]))
        
    return gen_clip_t / len(image_pairs), gt_clip_t / len(image_pairs)

def load_data_from_json(json_path, generated_image_dir, gt_image_dir):
    """Load data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    final_turn_pairs = []
    caption_dict = []

    for entry in data:
        gen_img_path = os.path.join(generated_image_dir, entry['NeIn']) + ".jpg"
        gt_img_path = os.path.join(gt_image_dir, entry['COCO']) + ".jpg"
        final_turn_pairs.append((gen_img_path, gt_img_path))
        caption_dict.append(entry['T_negative'])

    return final_turn_pairs, caption_dict

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_workers',
                        type=int,
                        help='Number of processes to use for data loading.')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='Device to use. Like cuda or cpu')
    parser.add_argument('--generated_path',
                        help='Paths of generated images (folders) of considered methods')
    parser.add_argument('--gt_path',
                        default="/home/hdhieu/coco_for_val",
                        help='Paths to the gt images (folders)')
    parser.add_argument('--metric',
                        type=str,
                        default='l1,l2,clip-i,dino',
                        help='The metric to calculate (l1, l2, clip-i, dino, clip-t)')
    parser.add_argument('--save_path',
                        type=str,
                        default='results',
                        help='Path to save the results')
    parser.add_argument('--json_path',
                        default="/home/hdhieu/data/nein_val.json",
                        help='Path to the JSON file of NeIn validation')

    args = parser.parse_args()
    args.metric = args.metric.split(',')

    for arg in vars(args):
        print(arg, getattr(args, arg))

    if args.device is None:
        args.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        args.device = torch.device(args.device)

    # Load data from JSON
    final_turn_pairs, caption_dict = load_data_from_json(args.json_path, args.generated_path, args.gt_path)

    print(f"Number of pairs: {len(final_turn_pairs)}")

    evaluated_metrics_dict = {}

    if 'l1' in args.metric:
        final_turn_eval_score = eval_distance(final_turn_pairs, 'l1')
        print(f"L1 distance: {final_turn_eval_score}")
        evaluated_metrics_dict['l1'] = final_turn_eval_score

    if 'l2' in args.metric:
        final_turn_eval_score = eval_distance(final_turn_pairs, 'l2')
        print(f"L2 distance: {final_turn_eval_score}")
        evaluated_metrics_dict['l2'] = final_turn_eval_score

    if 'clip-i' in args.metric:
        model, transform = clip.load('ViT-B/32', device=args.device)
        final_turn_clip_i = eval_clip_i(args, final_turn_pairs, model, transform, 'clip_i')
        print(f"CLIP-I score: {final_turn_clip_i}")
        evaluated_metrics_dict['clip_i'] = final_turn_clip_i

    if 'dino' in args.metric:
        dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        dino_model.eval()
        dino_model.to(args.device)
        dino_transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        final_turn_dino = eval_clip_i(args, final_turn_pairs, dino_model, dino_transform, 'dino')
        print(f"DINO score: {final_turn_dino}")
        evaluated_metrics_dict['dino'] = final_turn_dino

    if 'clip-t' in args.metric:
        model, transform = clip.load('ViT-B/32', device=args.device)
        final_turn_clip_t, final_turn_clip_t_gt = eval_clip_t(args, final_turn_pairs, model, transform, caption_dict)
        print(f"CLIP-T score: {final_turn_clip_t}")
        print(f"CLIP-T GT score: {final_turn_clip_t_gt}")
        evaluated_metrics_dict['clip_t'] = final_turn_clip_t

    os.makedirs(args.save_path, exist_ok=True)
    evaluated_metrics_path = os.path.join(args.save_path, 'evaluated_metrics.json')
    with open(evaluated_metrics_path, 'w') as f:
        json.dump(evaluated_metrics_dict, f, indent=4)

    print(f"Evaluated metrics saved to {evaluated_metrics_path}")
