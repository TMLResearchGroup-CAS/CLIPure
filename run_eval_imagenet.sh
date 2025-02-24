# CUDA_VISIBLE_DEVICES=2 nohup python -u -m CLIP_eval.CLIPure_Cos --clip_model_name ViT-L-14 --pretrained openai --dataset imagenet --imagenet_root /data/resources/datasets/ImageNet --wandb False --norm linf --eps 4 > CLIPure_Cos_imagenet_L_14_eps4.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u -m CLIP_eval.CLIPure_Diff --clip_model_name ViT-L-14 --pretrained openai --dataset imagenet --imagenet_root /data/resources/datasets/ImageNet --wandb False --norm linf --eps 4 > CLIPure_Diff_imagenet_L_14_eps4.log 2>&1 &