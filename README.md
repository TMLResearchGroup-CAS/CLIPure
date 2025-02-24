# CLIPure
<p align="center">

Official PyTorch implementation of the ICLR 2025 paper:<br>

[CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification](https://arxiv.org/abs/2410.23091)

<br>

MingKun Zhang, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng<br>

https://github.com/ZhangMingKun1/CLIPure

<br>  

<img width="300" height="230" src=".zeroshot_classification.pdf">

</p>


Abstract: *In this paper, we aim to build an adversarially robust zero-shot image classifier. We ground our work on CLIP, a vision-language pre-trained encoder model that can perform zero-shot classification by matching an image with text prompts ''a photo of $<$class-name$>$''. Purification is the path we choose since it does not require adversarial training on specific attack types and thus can cope with any foreseen attacks. We then formulate purification risk as the KL divergence between the joint distributions of the purification process of denoising the adversarial samples and the attack process of adding perturbations to benign samples, through bidirectional Stochastic Differential Equations (SDEs). The final derived results inspire us to explore purification in the multi-modal latent space of CLIP. We propose two variants for our CLIPure approach: \textit{CLIPure-Diff} which models the likelihood of images' latent vectors with the DiffusionPrior module in DaLLE-2 (modeling the generation process of CLIP's latent vectors), and \textit{CLIPure-Cos} which models the likelihood with the cosine similarity between the embeddings of an image and ''a photo of a.''. As far as we know, CLIPure is the first purification method in multi-modal latent space and CLIPure-Cos is the first purification method that is not based on generative models, which substantially improves defense efficiency. We conducted extensive experiments on CIFAR-10, ImageNet, and 13 datasets that previous CLIP-based defense methods used for evaluating zero-shot classification robustness. Results show that CLIPure boosts the SOTA robustness by a large margin, e.g., from 71.7\% to \textbf{91.1\%} on CIFAR10, from 59.6\% to \textbf{72.6\%} on ImageNet, and \textbf{108\%} relative improvements of average robustness on the 13 datasets over previous SOTA.*

## Requirements

The code is achieved with Python 3.7.13. To install the required packages, run:

  ```bash

  pip install -r requirements.txt

  ```


## Run experiments on ImageNet



### CLIPure-Cos
- To get results of CLIPure-Cos defending against AutoAttack Linf on ImageNet (the CLIP model e.g., ViT-L-14 would be downloaded automatically):

```bash

CUDA_VISIBLE_DEVICES=0 nohup python -u -m CLIP_eval.CLIPure_Cos --clip_model_name ViT-L-14 --pretrained openai --dataset imagenet --imagenet_root /data/resources/datasets/ImageNet --wandb False --norm linf --eps 4 > CLIPure_Cos_imagenet_L_14_eps4.log 2>&1 &


```

### CLIPure-Diff
- To get results of CLIPure-Cos defending against AutoAttack Linf on ImageNet (Prepare DaLLE2 following [dalle2_pytorch](https://github.com/lucidrains/DALLE2-pytorch/tree/680dfc4d93b70f9ab23c814a22ca18017a738ef6) and set the model path in script_state.json):

```bash

CUDA_VISIBLE_DEVICES=0 nohup python -u -m CLIP_eval.CLIPure_Diff --clip_model_name ViT-L-14 --pretrained openai --dataset imagenet --imagenet_root /data/resources/datasets/ImageNet --wandb False --norm linf --eps 4 > CLIPure_Diff_imagenet_L_14_eps4.log 2>&1 &


```



This work may be used non-commercially, meaning for research or evaluation

purposes only. For business inquiries, please contact [zhangmingkun20z@ict.ac.cn](zhangmingkun20z@ict.ac.cn).



## Citation



Please cite our paper, if you happen to use this codebase:

```
@article{zhang2024causaldiff,
  title={CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense},
  author={Zhang, Mingkun and Bi, Keping and Chen, Wei and Chen, Quanrun and Guo, Jiafeng and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2410.23091},
  year={2024}
}
```
