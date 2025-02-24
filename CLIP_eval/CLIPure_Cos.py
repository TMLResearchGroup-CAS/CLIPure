import json
import os
import sys
import time

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize
from torchvision import transforms
from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL
import wandb
import argparse
from robustbench import benchmark
from robustbench.data import load_clean_dataset
from autoattack import AutoAttack
from robustbench.model_zoo.enums import BenchmarkDataset
from CLIP_eval.eval_utils import compute_accuracy_no_dataloader, load_clip_model
from train.utils import str2bool
import matplotlib
matplotlib.use('Agg')
import sys
import os
import json
from collections import namedtuple

import torch
import os
from resize_right import resize


from dalle2_pytorch.tokenizer import tokenizer
import numpy as np

from torch.nn.functional import cosine_similarity
torch.set_printoptions(precision=3)

parser = argparse.ArgumentParser(description="Script arguments")

parser.add_argument('--clip_model_name', type=str, default='none', help='ViT-L-14, ViT-B-32, don\'t use if wandb_id is set')
parser.add_argument('--pretrained', type=str, default='openai', help='Pretrained model ckpt path, don\'t use if wandb_id is set')
parser.add_argument('--wandb_id', type=str, default='none', help='Wandb id of training run, don\'t use if clip_model_name and pretrained are set')
parser.add_argument('--logit_scale', type=str2bool, default=True, help='Whether to scale logits')
parser.add_argument('--full_benchmark', type=str2bool, default=False, help='Whether to run full RB benchmark')
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument('--imagenet_root', type=str, default='/mnt/datasets/imagenet', help='Imagenet dataset root directory')
parser.add_argument('--cifar10_root', type=str, default='/mnt/datasets/CIFAR10', help='CIFAR10 dataset root directory')
parser.add_argument('--cifar100_root', type=str, default='/mnt/datasets/CIFAR100', help='CIFAR100 dataset root directory')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--n_samples_imagenet', type=int, default=1000, help='Number of samples from ImageNet for benchmark')
parser.add_argument('--n_samples_cifar', type=int, default=1000, help='Number of samples from CIFAR for benchmark')
parser.add_argument('--template', type=str, default='ensemble', help='Text template type; std, ensemble')
parser.add_argument('--norm', type=str, default='linf', help='Norm for attacks; linf, l2')
parser.add_argument('--eps', type=float, default=4., help='Epsilon for attack')
parser.add_argument('--beta', type=float, default=0., help='Model interpolation parameter')
parser.add_argument('--alpha', type=float, default=2., help='APGD alpha parameter')
parser.add_argument('--experiment_name', type=str, default='', help='Experiment name for logging')
parser.add_argument('--blackbox_only', type=str2bool, default=False, help='Run blackbox attacks only')
parser.add_argument('--save_images', type=str2bool, default=False, help='Save images during benchmarking')
parser.add_argument('--wandb', type=str2bool, default=True, help='Use Weights & Biases for logging')
parser.add_argument('--devices', type=str, default='', help='Device IDs for CUDA')


CIFAR10_LABELS = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
EmbeddedText = namedtuple('EmbedTextReturn', ['text_embed', 'text_encodings'])
EmbeddedImage = namedtuple('EmbedImageReturn', ['image_embed', 'image_encodings'])

import math
from collections import Counter

def l2norm(t):
	return F.normalize(t, dim = -1)

def resize_image_to(image, target_image_size):
	orig_image_size = image.shape[-1]

	if orig_image_size == target_image_size:
		return image

	scale_factors = target_image_size / orig_image_size
	return resize(image, scale_factors = scale_factors)

def model_dtype_half(model):
	"""
	将特定模块中的参数转换为 FP16。
	"""
	for name, param in model.named_parameters():
		# 检查模块名称是否包含指定关键字
		if any(module_name in name for module_name in ['ln_']):
			continue
		if any(module_name in name for module_name in ['mlp', 'attn', 'conv1', 'visual.proj', 'text_projection', 'q_proj', 'proj.bias', 'proj']):
		# if any(module_name in name for module_name in ['mlp', 'attn', 'conv1', 'visual.proj', 'text_projection']):
			if param.dtype != torch.float16:
				param.data = param.data.half()  # 转为 FP16
				# print(f"参数 {name} 转换为 FP16")
	return True

class ClassificationModel(torch.nn.Module):
	def __init__(self, clip_model_name, model, text_embedding, templates, args, input_normalize, resizer=None, logit_scale=True, tokenizer=None):
			super().__init__()
			self.clip_model_name = clip_model_name
			self.clip = model
			self.args = args
			self.input_normalize = input_normalize
			self.resizer = resizer if resizer is not None else lambda x: x
			self.text_embedding = text_embedding
			self.logit_scale = logit_scale
			self.tokenizer = tokenizer
			model_dtype_half(self.clip)
			self.device = device

			self.cleared = False

			null_templates = [template.format(c="") for template in templates]
			temp_emb_all = []
			for temp in null_templates:
				text_purify = self.tokenizer(temp).to(device)
				text_embed, _ = self.embed_text(text_purify)
				text_embed = text_embed / text_embed.norm()
				temp_emb_all.append(text_embed)

			self.temp_emb_all = torch.stack(temp_emb_all, dim=1).to(device)

			self.iter = 10
			self.step_size = 30.

	def find_layer(self,  layer):
		modules = dict([*self.clip.named_modules()])
		return modules.get(layer, None)

	def clear(self):
		if self.cleared:
			return

		self.handle()

	def _hook(self, _, inputs, outputs):
		self.text_encodings = outputs

	def uniform_noise(self, *args, begin: float = 0.0, end: float = 1.0, **kwargs):
		x = torch.rand(*args, **kwargs, device=self.device)
		x = x * (end - begin) + begin
		return x
	
	@torch.enable_grad()
	def embed_text(self, text):
		text = text[..., :256]
		text_mask = text != 0
		text_embed = self.clip.encode_text(text)
		return EmbeddedText(l2norm(text_embed.float()), l2norm(text_embed.float()))
	
	@torch.enable_grad()
	def embed_image(self, image):
		image = resize_image_to(image, 224)
		image = self.input_normalize(image)
		image_embed = self.clip.encode_image(image.half())
		return EmbeddedImage(l2norm(image_embed.float()), None)


	def purify_zi(self, img_emb, iter=10, step_size=10.):
			step_size_u = step_size
			batch, device = img_emb.shape[0], img_emb.device
			if not img_emb.requires_grad:
				img_emb.requires_grad = True  # 确保图像嵌入需要梯度

			text_embed = self.temp_emb_all.mean(dim=1)
			text_embed = text_embed.repeat(batch, 1).to(device)
			
			momentum = torch.zeros_like(img_emb)
			norm = "L2"
			gamma = 0.
			for i in range(iter):
				r = torch.norm(img_emb, dim=1, keepdim=True)
				u = img_emb / r

				logits_uncond = cosine_similarity(img_emb, text_embed, dim=1)
				loss = - logits_uncond
				grad = torch.autograd.grad(loss, img_emb, torch.ones_like(loss), retain_graph=True)[0]

				grad_u = r * grad

				if norm == "Linf":
					momentum = gamma * momentum - (1 - gamma) * grad_u / torch.norm(grad_u, p=1)
					u = u + step_size_u * momentum.sign()
				elif norm == "L2":
					momentum = gamma * momentum - (1 - gamma) * grad_u / torch.norm(grad_u, p=2)
					u = u + step_size_u * momentum
				
				u = u / torch.norm(u, dim=1, keepdim=True)
				img_emb = r * u

			return img_emb
	

	def forward(self, vision, output_normalize=True):
			assert output_normalize
			
			with torch.enable_grad():
				embedding_norm_, _ = self.embed_image(vision)
				embedding_norm_ = self.purify_zi(vision, embedding_norm_, iter=self.iter, step_size=self.step_size)
			logits = embedding_norm_ @ self.text_embedding.to(embedding_norm_.dtype)
			
			if self.logit_scale:
					logits *= self.clip.logit_scale.exp()
			return logits


if __name__ == '__main__':
	# set seeds
	torch.manual_seed(0)
	np.random.seed(0)

	# Parse command-line arguments
	args = parser.parse_args()
	# print args
	print(f"Arguments:\n{'-' * 20}", flush=True)
	for arg, value in vars(args).items():
			print(f"{arg}: {value}")
	print(f"{'-' * 20}")

	args.eps /= 255
	# make sure there is no string in args that should be a bool
	assert not any(
			[isinstance(x, str) and x in ['True', 'False'] for x in args.__dict__.values(
			)])

	if args.dataset == 'imagenet':
			num_classes = 1000
			data_dir = args.imagenet_root
			n_samples = args.n_samples_imagenet
			resizer = None
	elif args.dataset == 'cifar100':
			num_classes = 100
			data_dir = args.cifar100_root
			n_samples = args.n_samples_cifar
			resizer = Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=False)
	elif args.dataset == 'cifar10':
			num_classes = 10
			data_dir = args.cifar10_root
			n_samples = args.n_samples_cifar
			resizer = Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=False)
	eps = args.eps

	# init wandb
	os.environ['WANDB__SERVICE_WAIT'] = '300'
	wandb_user, wandb_project = None, None
	while True:
		try:
			run_eval = wandb.init(
				project=wandb_project,
				job_type='eval',
				name=f'{"rb" if args.full_benchmark else "aa"}-clip-{args.dataset}-{args.norm}-{eps:.2f}'
					f'-{args.wandb_id if args.wandb_id is not None else args.pretrained}-{args.blackbox_only}-{args.beta}',
				save_code=True,
				config=vars(args),
				mode='online' if args.wandb else 'disabled'
			)
			break
		except wandb.errors.CommError as e:
			print('wandb connection error', file=sys.stderr)
			print(f'error: {e}', file=sys.stderr)
			time.sleep(1)
			print('retrying..', file=sys.stderr)

	if args.devices != '':
		# set cuda visible devices
		os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
	main_device = 0
	device = torch.device(main_device)
	num_gpus = torch.cuda.device_count()
	if num_gpus > 1:
		print(f"Number of GPUs available: {num_gpus}")
	else:
		print("No multiple GPUs available.")

	if not args.blackbox_only:
		attacks_to_run = ['apgd-ce', 'apgd-t']
		# attacks_to_run = ['apgd-t']
		# attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
	else:
		attacks_to_run = ['square']
	print(f'[attacks_to_run] {attacks_to_run}')


	if args.wandb_id not in [None, 'none', 'None']:
		assert args.pretrained in [None, 'none', 'None']
		assert args.clip_model_name in [None, 'none', 'None']
		api = wandb.Api()
		run_train = api.run(f'{wandb_user}/{wandb_project}/{args.wandb_id}')
		clip_model_name = run_train.config['clip_model_name']
		print(f'clip_model_name: {clip_model_name}')
		pretrained = run_train.config["output_dir"]
		if pretrained.endswith('_temp'):
				pretrained = pretrained[:-5]
		pretrained += "/checkpoints/final.pt"
	else:
		clip_model_name = args.clip_model_name
		pretrained = args.pretrained
		run_train = None
	del args.clip_model_name, args.pretrained

	print(f'[loading pretrained clip] {clip_model_name} {pretrained}')

	model, preprocessor_without_normalize, normalize = load_clip_model(clip_model_name, pretrained, args.beta)
	
	# load FARE_eps4 / TeCoA_eps
	# print('Load FARE/TeCoA Eps 4 checkpoint')
	# checkpoint = torch.load('ckpts/fare_eps_4.pt', map_location=torch.device('cpu'))
	# checkpoint = torch.load('ckpts/tecoa_eps_4.pt', map_location=torch.device('cpu'))
	# model.visual.load_state_dict(checkpoint)


	if args.dataset != 'imagenet':
		# make sure we don't resize outside the model as this influences threat model
		preprocessor_without_normalize = transforms.ToTensor()
	print(f'[resizer] {resizer}')
	print(f'[preprocessor] {preprocessor_without_normalize}')

	# model.eval()
	model.float()
	model.to(main_device)


	tokenizer = open_clip.get_tokenizer(clip_model_name)
	with torch.no_grad():
		# Get text label embeddings of all ImageNet classes
		if not args.template == 'ensemble':
			if args.template == 'std':
				template = 'This is a photo of a {}'
			else:
				raise ValueError(f'Unknown template: {args.template}')
			print(f'template: {template}')
			if args.dataset == 'imagenet':
				texts = [template.format(c) for c in IMAGENET_1K_CLASS_ID_TO_LABEL.values()]
			elif args.dataset == 'cifar10':
				texts = [template.format(c) for c in CIFAR10_LABELS]
			text_tokens = open_clip.tokenize(texts)
			embedding_text_labels_norm = []
			text_batches = [text_tokens[:500], text_tokens[500:]] if args.dataset == 'imagenet' else [text_tokens]
			for el in text_batches:
				# we need to split the text tokens into two batches because otherwise we run out of memory
				# note that we are accessing the model directly here, not the CustomModel wrapper
				# thus its always normalizing the text embeddings
				embedding_text_labels_norm.append(
					model.encode_text(el.to(main_device), normalize=True).detach().cpu()
				)
			model.cpu()
			embedding_text_labels_norm = torch.cat(embedding_text_labels_norm).T.to(main_device)
		else:
			assert args.dataset == 'imagenet', 'ensemble only implemented for imagenet'
			with open('CLIP_eval/zeroshot-templates.json', 'r') as f:
					templates = json.load(f)
			templates = templates['imagenet1k']
			print(f'[templates] {templates}')
			embedding_text_labels_norm = []
			text_encoding_classes = []
			for c in IMAGENET_1K_CLASS_ID_TO_LABEL.values():
				texts = [template.format(c=c) for template in templates]
				text_tokens = tokenizer(texts).to(main_device)
				class_embeddings = model.encode_text(text_tokens)
				text = text_tokens
				text = text[..., :256]
				text_mask = text != 0
				text_embed = model.encode_text(text)
				text_embed, text_encodings = EmbeddedText(l2norm(text_embed.float()), l2norm(text_embed.float()))
				class_embedding = F.normalize(class_embeddings, dim=-1)#.mean(dim=0)
				class_embedding = class_embedding.mean(dim=0)
				class_embedding /= class_embedding.norm()
				embedding_text_labels_norm.append(class_embedding)
			embedding_text_labels_norm = torch.stack(embedding_text_labels_norm, dim=1).to(main_device)

	print('clip_model_name: {}'.format(clip_model_name))
	model = ClassificationModel(
			clip_model_name=clip_model_name,
			model=model,
			text_embedding=embedding_text_labels_norm,
			templates=templates,
			args=args,
			resizer=resizer,
			input_normalize=normalize,
			logit_scale=args.logit_scale,
			tokenizer=open_clip.get_tokenizer(clip_model_name)
	)

	if num_gpus > 1:
			model = torch.nn.DataParallel(model)
	model = model.cuda()
	model.eval()

	model_name = None
	torch.cuda.empty_cache()
	dataset_short = (
			'img' if args.dataset == 'imagenet' else
			'c10' if args.dataset == 'cifar10' else
			'c100' if args.dataset == 'cifar100' else
			'unknown'
	)

	start = time.time()
	if args.full_benchmark:
		clean_acc, robust_acc = benchmark(
			model, model_name=model_name, n_examples=n_samples,
			batch_size=args.batch_size,
			dataset=args.dataset, data_dir=data_dir,
			threat_model=args.norm.replace('l', 'L'), eps=eps,
			preprocessing=preprocessor_without_normalize,
			device=device, to_disk=False
		)
		clean_acc *= 100
		robust_acc *= 100
		duration = time.time() - start
		print(f"[Model] {pretrained}")
		print(
			f"[Clean Acc] {clean_acc:.2f}% [Robust Acc] {robust_acc:.2f}% [Duration] {duration / 60:.2f}m"
			)
		if run_train is not None:
			del api, run_train
			api = wandb.Api()
			run_train = api.run(f'{wandb_user}/{wandb_project}/{args.wandb_id}')
			eps_descr = str(int(eps * 255)) if args.norm == 'linf' else str(eps)
			run_train.summary.update({f'rb/acc-{dataset_short}': clean_acc})
			run_train.summary.update({f'rb/racc-{dataset_short}-{args.norm}-{eps_descr}': robust_acc})
			run_train.update()
	else:
		x_test, y_test = load_clean_dataset(
			BenchmarkDataset(args.dataset), n_examples=n_samples, data_dir=data_dir,
			prepr=preprocessor_without_normalize,)
			
		for iter in [10]:
			for lr in [30.]:
				print('iter = {}, lr = {}'.format(iter, lr))
				model.iter = iter
				model.step_size = lr
				adversary = AutoAttack(
					model, norm=args.norm.replace('l', 'L'), eps=eps, version='custom', attacks_to_run=attacks_to_run,
					alpha=args.alpha, verbose=True
				)

				x_adv, y_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size, return_labels=True)  # y_adv are preds on x_adv

	run_eval.finish()













