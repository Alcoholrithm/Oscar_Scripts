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
## Notice
Since scene_graph_benchmark repo, the vinvl encoder only support default cuda <br>
So if you want to use other cuda device <br>
You must change default cuda <br>
Insert the code below <nr>

```Python
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
```

## Performance
Task | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr |
-----|--------|--------|--------|--------|-------|
Ours(XE) |  72.7  |  54.6  |  36.9  |  23.0  |  118.0  |
Ours(CIDEr) |  72.9  |  54.92  |  37.4  |  23.7  |  118.0  | 
Oscar+ |  -   |    -   |    -   |  41.0  |  140.9  |
