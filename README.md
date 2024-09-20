# NeIn: Telling What You Don’t Want 🌱🌻🌺

### [Project Page](https://tanbuinhat.github.io/NeIn/) | [Paper](https://arxiv.org/abs/2409.06481) | [Data]()

Negation is a fundamental linguistic concept used by humans to convey information that they do not desire. Despite this, there has been minimal research specifically focused on negation within vision-language tasks. 
This lack of research means that vision-language models (VLMs) may struggle to understand negation, implying that they struggle to provide accurate results. One barrier to achieving human-level intelligence is the lack of a standard collection by which research into negation can be evaluated. 
This paper presents the first large-scale dataset, **Ne**gative **In**struction (NeIn), for studying negation within the vision-language domain. Our dataset comprises **530,694 quadruples**, i.e., source image, original caption, negative sentence, and target image in total, including **495,694 queries for training** and **35,000 queries for benchmarking** across multiple vision-language tasks. 
Specifically, we automatically generate NeIn based on a large, existing vision-language dataset, MS-COCO, via two steps: generation and filtering. During the generation phase, we leverage two VLMs, BLIP and MagicBrush, to generate the target image and a negative clause that expresses the content of the source image. In the subsequent filtering phase, we apply BLIP to remove erroneous samples. 
Additionally, we introduce an evaluation protocol for negation understanding of image editing models. 
Extensive experiments using our dataset across multiple VLMs for instruction-based image editing tasks demonstrate that even recent state-of-the-art VLMs struggle to understand negative queries.


by [Nhat-Tan Bui](https://tanbuinhat.github.io/), [Dinh-Hieu Hoang](https://scholar.google.com/citations?user=713F7a8AAAAJ), [Quoc-Huy Trinh](https://huyquoctrinh.onrender.com/), [Minh-Triet Tran](https://en.hcmus.edu.vn/profile/tran-minh-triet/), [Truong Nguyen](https://jacobsschool.ucsd.edu/people/profile/truong-q-nguyen), [Susan Gauch](https://csce.uark.edu/~sgauch/)

### 📝 TODO

- [ ] Pipeline for generating NeIn
- [ ] Dataset

### Citation

```
@article{bui2024nein,
      author={Bui, Nhat-Tan and Hoang, Dinh-Hieu and Trinh, Quoc-Huy and Tran, Minh-Triet and Nguyen, Truong and Gauch, Susan},
      title={NeIn: Telling What You Don't Want},
      journal={arXiv preprint arXiv:2409.06481},
      year={2024}
}
```
