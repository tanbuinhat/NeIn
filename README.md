# NeIn: Telling What You Don’t Want 🌱🌻🌺

### [Project Page](https://tanbuinhat.github.io/NeIn/) | [Paper](https://arxiv.org/abs/2409.06481) | [Data (Drive)](https://drive.google.com/drive/folders/1pxiV6G__cWZ0qMOh4nDTTSiCQfkgyxPD?usp=drive_link) | [Data (Hugging Face)](https://huggingface.co/datasets/nhatttanbui/NeIn)

Negation is a fundamental linguistic concept used by humans to convey information that they do not desire. Despite this, minimal research has focused on negation within text-guided image editing. This lack of research means that vision-language models (VLMs) for image editing may struggle to understand negation, implying that they struggle to provide accurate results. One barrier to achieving human-level intelligence is the lack of a standard collection by which research into negation can be evaluated.
This paper presents the first large-scale dataset, **Ne**gative <b>In</b>struction (<b>NeIn</b>), for studying negation within instruction-based image editing. Our dataset comprises <b>366,957 quintuplets</b>, i.e., source image, original caption, selected object, negative sentence, and target image in total, including <b>342,775 queries for training</b> and <b>24,182 queries for benchmarking</b> image editing methods.
Specifically, we automatically generate NeIn based on a large, existing vision-language dataset, MS-COCO, via two steps: generation and filtering. 
During the generation phase, we leverage two VLMs, BLIP and InstructPix2Pix (fine-tuned on MagicBrush dataset), to generate NeIn's samples and the negative clauses that expresses the content of the source image. In the subsequent filtering phase, we apply BLIP and LLaVA-NeXT to remove erroneous samples.
Additionally, we introduce an evaluation protocol to assess the negation understanding for image editing models.
Extensive experiments using our dataset across multiple VLMs for text-guided image editing demonstrate that even recent state-of-the-art VLMs struggle to understand negative queries.


by [Nhat-Tan Bui](https://tanbuinhat.github.io/), [Dinh-Hieu Hoang](https://scholar.google.com/citations?user=713F7a8AAAAJ), [Quoc-Huy Trinh](https://huyquoctrinh.onrender.com/), [Minh-Triet Tran](https://en.hcmus.edu.vn/profile/tran-minh-triet/), [Truong Nguyen](https://jacobsschool.ucsd.edu/people/profile/truong-q-nguyen), [Susan Gauch](https://engineering.uark.edu/electrical-engineering-computer-science/computer-science-faculty/uid/sgauch/name/Susan+E.+Gauch/)


<image src="static/images/process.png">

<b>Note: Sorry if my code isn't very clean!!!</b>

### Generation
Note that please read the meaning of each parameter in the code file.
<ol>
<li> Download the MS-COCO dataset 2014 (images and annotations) <a href="https://cocodataset.org/#download">here.</a> Note that our dataset doesn't follow the train-val split of COCO.</li>
   
<li> Generate all clauses (T_original, T_generative, and T_generated)</li>
   
```
python generation_phase/clause_generation.py --object-file annotations/coco_objects.txt --coco-path coco --coco-annotation captions.json --output-json nein_generation.json
```

<li> Generate NeIn's samples by image editing method. In this work, we leverage InstructPix2Pix finetuned on MagicBrush. So you need to download <a href="https://osu-nlp-group.github.io/MagicBrush/">MagicBrush</a> checkpoint and set up environment for <a href="https://github.com/timothybrooks/instruct-pix2pix">InstructPix2Pix</a>. After that, you can run the generation</li>
   
```
python generation_phase/edit_cli_generation.py --input-json nein_generation.json --input-folder coco --output nein_samples
```
</ol>

### Filtering

```
python filtering_phase/filtering.py --output-json nein_selected.json --generation-json nein_generation.json --generation-samples nein_samples
```
The above code will create a JSON file containing the names of the samples after filtering; you need to delete or substitute the filtered-out samples in the generated samples from the generation phase.

### Evaluation Protocol
The evaluation step uses COCO images as ground truth. Since our dataset doesn't follow the standard COCO train-val split, you can download the pre-split data for evaluation <a href="https://drive.google.com/file/d/17vXF9ujsFyParFOYXoghwRdsd9SvhrXc/view?usp=drive_link">here</a>. This folder contains the COCO images corresponding to our validation set.

All the inference results of image editing and finetuned versions can be found <a href="https://drive.google.com/drive/folders/1CtlG1vkUsdBC9VvULuLYfcgaPTZTVHAL?usp=drive_link">here</a>.
<ol>
<li> Image quality is the same as MagicBrush. 

```
python evaluation_protocol/image_quality.py --generated_path considered_results --gt_path coco_for_val --save_path quality_metrics --json_path val.json
```
<li>Removal for OVD</li>

```
python evaluation_protocol/OVD/removal_eval.py --output-json removal_ovd_considered.json --nein-json val.json --edited-folder considered_results
python evaluation_protocol/OVD/removal_auc.py --removal-json removal_ovd_considered.json
```

<li>Retention for OVD. The first step of the retention eval is to check whether the original objects are still present in the NeIn samples. This step is run once using <em>check_obj.py</em>. You can find the pre-computed file <a href="https://drive.google.com/file/d/1EtRnMHaeGjWu5TWrPZdQSCakUW9H_W5O/view?usp=drive_link">here</a>.</li>

```
python evaluation_protocol/OVD/check_obj.py --output-json retention_obj_ovd.json --nein-json val.json --nein-folder nein_val
python evaluation_protocol/OVD/retention_eval.py --output-json retention_ovd_considered.json --retention-json retention_obj_ovd.json --edited-folder considered_results
```

<li>Removal for VQA</li>

```
python evaluation_protocol/VQA/removal_eval.py --output-json removal_vqa_considered.json --nein-json val.json --edited-folder considered_results
```

<li>Retention for VQA. The first step is similar to OVD. You can find the pre-computed file <a href="https://drive.google.com/file/d/1eKd0jy7m_iO0PEKRhVI5ioFT-_XhAE1k/view?usp=drive_link">here</a>.</li>

```
python evaluation_protocol/VQA/check_obj.py --output-json retention_obj_vqa.json --nein-json val.json --nein-folder nein_val
python evaluation_protocol/OVD/retention_eval.py --output-json retention_vqa_considered.json --retention-json retention_obj_vqa.json --edited-folder considered_results
```
</ol>



### Acknowledgement
NeIn is created based on these wonderful works: [InstructPix2Pix](https://www.timothybrooks.com/instruct-pix2pix), [MagicBrush](https://osu-nlp-group.github.io/MagicBrush/), and [BLIP](https://github.com/salesforce/BLIP). Thank you 🪐💗🌕.

### Citation

```
@article{bui2024nein,
      author={Bui, Nhat-Tan and Hoang, Dinh-Hieu and Trinh, Quoc-Huy and Tran, Minh-Triet and Nguyen, Truong and Gauch, Susan},
      title={NeIn: Telling What You Don't Want},
      journal={arXiv preprint arXiv:2409.06481},
      year={2024}
}
```