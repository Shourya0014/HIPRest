# HIRest
[HIREST]Real-Time Image Sharpening via Distilled HINet from Restormer


## 🔗 Download Full Project

Due to GitHub size limitations, the full project (including large model files and logs) is hosted externally.

👉 [Click here to download the full project from Google Drive](https://drive.google.com/drive/folders/1nl_JCtxhQ8Rdi8N_oydG-p1cvAoGgSH2?usp=sharing)

---
#### Aman Kanodia,Nomesh Shourya Thakur,Suman Sourav
#### Paper: Coming Soon..
>•Goal: Enhance image sharpness in video streams (e.g. video conferencing) by removing blur and restoring details lost due to compression or low resolution. 
•	Target Performance: Achieve image quality with SSIM > 90% (Structural Similarity) compared to original high-quality frames. Maintain visual fidelity for text, faces, nature scenes, etc. 

### Network Architecture
<img src="demo/20250712_2328_Image Sharpening Pipeline_simple_compose_01jzzwhepxfx4bzmszd4188bvd.png" alt="arch" style="zoom:100%;" />


### Installation

This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks. 

```python
python 3.6.9
pytorch 1.5.1
cuda 10.1
```



```
git clone ...
cd HINet
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### Quick Start (Single Image Inference)
---

* ```python basicsr/demo.py -opt options/demo/demo.yml```
  * modified your [input and output path](https://github.com/megvii-model/HINet/blob/main/options/demo/demo.yml#L16-L17)
  * [define network](https://github.com/megvii-model/HINet/blob/main/options/demo/demo.yml#L20-L24)
  * [pretrained model](https://github.com/megvii-model/HINet/blob/main/options/demo/demo.yml#L28), it should match the define network.
     * for pretrained model, see [here](https://drive.google.com/drive/folders/1-XhXFHy0G7-ja45hT1r7EEpEqbCJpxBs)

### Image Restoration Tasks
---

Image deblur

<details>
  <summary>Image Deblur - GoPro dataset (Click to expand) </summary>

* prepare data

  * ```mkdir ./datasets/GoPro ```
  
  * download the [train](https://drive.google.com/drive/folders/1AsgIP9_X0bg0olu2-1N6karm2x15cJWE) set in ./datasets/GoPro/train and [test](https://drive.google.com/drive/folders/1a2qKfXWpNuTGOm2-Jex8kfNSzYJLbqkf) set in ./datasets/GoPro/test 
  * it should be like:
  
    ```bash
    ./datasets/
    ./datasets/GoPro/
    ./datasets/GoPro/train/
    ./datasets/GoPro/train/input/
    ./datasets/GoPro/train/target/
    ./datasets/GoPro/test/
    ./datasets/GoPro/test/input/
    ./datasets/GoPro/test/target/
    ```
  
  * ```python scripts/data_preparation/gopro.py```
  
    * crop the train image pairs to 512x512 patches.


* eval
  * download [pretrained model](https://drive.google.com/drive/folders/1-XhXFHy0G7-ja45hT1r7EEpEqbCJpxBs) to ./experiments/pretrained_models/HINet-GoPro.pth
  * ```python basicsr/test.py -opt options/test/GoPro/HINet-GoPro.yml  ```
  
* train

  * ```python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/GoPro/HINet.yml --launcher pytorch```

</details>




### Results

---
Some of the following results are higher than the original paper as we optimized some hyper-parameters.


<div align="center">
<img src="demo/GOPR0384_11_00-000001.png" height="400px" alt="blurred image"><img src="demo/demo1.png" height="400px" alt="SIDD Result">
</div>




### License

 It is based on [BasicSR](https://github.com/xinntao/BasicSR) which is under the Apache 2.0 license.


### Citations

If HIPRest helps your research or work, please consider citing HIPRest.
```
{
    author    = {Aman Kumar ,  Nomesh Shourya Thakur and Suman Sourav},
    title     = {HIREST: Real-Time Image Sharpening via Distilled HINet from Restormer},
    month     = {July},
    year      = {2025},
    
}
```

### Contact
If you have any questions, please contact nomeshshourya146@gmail.com or amankanodia77@gmail.com -->
