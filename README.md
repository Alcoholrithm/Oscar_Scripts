# Oscar
Scripts for to inference using Oscar in Image Captioning and VQA tasks

## Requirements
```
For inference One Image
Minimum VRAM = 6GB, You must run 'torch.cuda.empty_cache()' to flush gpu cache at every inference
Recommemd VRAM = 7GB or more
```

## Demo
```
To Know How to use Oscar models see *.ipynb
```

## Results

### Image Captioning
<p align=center>
<img src="figures/IC.png">
</p>

### GQA
<p align="center">
<img src="figures/GQA.png">
</p>

## Notice
Since scene_graph_benchmark repo, the vinvl encoder only support default cuda. <br>
So if you want to use other cuda device. <br>
You must change default cuda. <br>
Insert the code below on your code.<nr>

```Python
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
```

## MODEL ZOO

Task | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr |
-----|--------|--------|--------|--------|-------|
[Ours(XE)](https://drive.google.com/file/d/110N20FiHgyPVuwVnBTKgBHFCkn5Uf0Iz/view?usp=sharing) |  72.7  |  54.6  |  36.9  |  23.0  |  118.0  |
[Ours(CIDEr)](https://drive.google.com/file/d/1ORxgRWcM_mTKkr6jToRn4rR7dFD1qdYS/view?usp=sharing) |  72.9  |  54.92  |  37.4  |  23.7  |  118.0  | 
Oscar+ |  -   |    -   |    -   |  41.0  |  140.9  |
