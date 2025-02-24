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
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import ipywidgets as widgets
import sys
import os
import json
import random
from collections import namedtuple

import torch
import os
from resize_right import resize


from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, OpenAIClipAdapter, train_configs
from dalle2_pytorch.tokenizer import tokenizer

import numpy as np

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

def l2norm(t):
	return F.normalize(t, dim = -1)

def resize_image_to(image, target_image_size):
	orig_image_size = image.shape[-1]

	if orig_image_size == target_image_size:
			return image

	scale_factors = target_image_size / orig_image_size
	return resize(image, scale_factors = scale_factors)

class ClassificationModel(torch.nn.Module):
	def __init__(self, model, dalle2_clip, diffusion_prior, decoder, text_embedding, text_embedding_temp, text_embed_classes, text_encoding_classes, templates, args, input_normalize, resizer=None, logit_scale=True, tokenizer=None):
			super().__init__()
			self.clip = model
			self.dalle2_clip = dalle2_clip
			self.args = args
			self.input_normalize = input_normalize
			self.resizer = resizer if resizer is not None else lambda x: x
			self.text_embedding = text_embedding
			self.text_embedding_temp = text_embedding_temp
			self.logit_scale = logit_scale
			self.tokenizer = tokenizer

			self.diffusion_prior = diffusion_prior
			self.diffusion_prior.cond_drop_prob = 1.
			# self.EDMLM = EDMLM
			self.decoder = decoder
			self.decoder.clip = dalle2_clip
			self.decoder.image_cond_drop_prob = 1.
			self.decoder.text_cond_drop_prob = 1.
			self.device = device
			text_attention_final = self.find_layer('ln_final')

			self.dim_latent_ = text_attention_final.weight.shape[0]
			self.handle = text_attention_final.register_forward_hook(self._hook)

			# self.clip_normalize = preprocess.transforms[-1]
			self.cleared = False
			self.text_embed_classes = text_embed_classes
			self.text_encoding_classes = text_encoding_classes

			null_templates = [template.format(c="") for template in templates]
			for index in range(len(null_templates)):
				text_temp = null_templates[index]
				text_embed, text_encoding = self.get_text_emb(text_temp)
				if index == 0:
					text_embed_all = text_embed
					text_encoding_all = text_encoding
				else:
					text_embed_all = text_embed_all + text_embed
					text_encoding_all = text_encoding_all + text_encoding
			text_embed_all = text_embed_all/len(null_templates)
			text_encoding_all = text_encoding_all/len(null_templates)

			self.text_embed_all = text_embed_all.to(device)
			self.text_encoding_all = text_encoding_all.to(device)

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
	
	def get_text_emb(self, text):
		text = self.tokenizer(text)
		text = text[..., :256]
		eos_id = 49407

		is_eos_id = (text == eos_id)
		text_mask_excluding_eos = is_eos_id.cumsum(dim = -1) == 0
		text_mask = F.pad(text_mask_excluding_eos, (1, -1), value = True)
		text_mask = (text_mask & (text != 0)).to(device)

		text_embed = self.clip.encode_text(text.to(device))
		text_encodings = text_embed.masked_fill(~text_mask[..., None], 0.)

		text_embed = F.normalize(text_embed, dim = -1)
		text_encodings = text_encodings.float()
		return text_embed, text_encodings

	def purify_zi(self, img_emb):
		batch, device = img_emb.shape[0], img_emb.device
		if not img_emb.requires_grad:
				img_emb.requires_grad = True

		text_embed = self.text_embed_all.repeat(batch, 1).to(device)
		text_encodings = self.text_encoding_all.repeat(batch, 1, 1).to(device)
		
		text_cond = {'text_embed': text_embed}
		
		if self.diffusion_prior.condition_on_text_encodings:
				text_cond['text_encodings'] = text_encodings

		iter = 10
		momentum = torch.zeros_like(img_emb)
		step_size = 30.
		norm = "L2"
		gamma = 0.

		for i in range(iter):
			with torch.enable_grad():
				if not img_emb.requires_grad:
					img_emb.requires_grad = True
				r = torch.norm(img_emb, dim=1, keepdim=True)
				u = img_emb / r


				scaled_image_embed = img_emb * self.diffusion_prior.image_embed_scale

				time = random.randint(900, 999)
				times = torch.full((batch,), time, device=device, dtype=torch.long)
				loss = self.diffusion_prior.p_losses(scaled_image_embed, times, text_cond=text_cond)

				grad = torch.autograd.grad(loss, scaled_image_embed, torch.ones_like(loss))[0]

				grad_u = r * grad

				if norm == "Linf":
					momentum = momentum - grad / torch.norm(grad, p=1)
					img_emb = img_emb + step_size * momentum.sign()
				elif norm == "L2":
					momentum = gamma * momentum - (1 - gamma) * grad_u / torch.norm(grad_u, p=2)
					u = u + step_size * momentum
				u = u / torch.norm(u, dim=1, keepdim=True)
				img_emb = r * u

		return img_emb


	def forward(self, vision, output_normalize=True):
			assert output_normalize

			with torch.enable_grad():
				embedding_norm_, _ = self.dalle2_clip.embed_image(vision)
				embedding_norm_ = self.purify_zi(embedding_norm_)

			logits = embedding_norm_ @ self.text_embedding.to(embedding_norm_.dtype)

			if self.logit_scale:
					logits *= self.clip.logit_scale.exp()
			return logits

decoder_versions = [{
	"name": "Original",
	"dalle2_install_path": "git+https://github.com/Veldrovive/DALLE2-pytorch@f4b687798d367fc434d8127ab31141f0fea0db26",
	"decoder_path": "https://huggingface.co/Veldrovive/DA-VINC-E/resolve/main/text_conditioned_epoch_34.pth",
	"config_path": "https://huggingface.co/Veldrovive/DA-VINC-E/raw/main/text_conditioned_config.json"
},{
	"name": "New 1B (Aesthetic)",
	"dalle2_install_path": "dalle2_pytorch==0.15.4",
	"decoder_path": "https://huggingface.co/laion/DALLE2-PyTorch/resolve/main/decoder/small_32gpus/latest.pth",
	"config_path": "https://huggingface.co/laion/DALLE2-PyTorch/raw/main/decoder/small_32gpus/decoder_config.json"
},{
	"name": "New 1.5B (Aesthetic)",
	"dalle2_install_path": "dalle2_pytorch==0.15.4",
	"decoder_path": "https://huggingface.co/laion/DALLE2-PyTorch/resolve/main/decoder/1.5B/latest.pth",
	"config_path": "https://huggingface.co/laion/DALLE2-PyTorch/raw/main/decoder/1.5B/decoder_config.json"
},{
	"name": "New 1.5B (Laion2B)",
	"dalle2_install_path": "dalle2_pytorch==0.15.4",
	"decoder_path": "https://huggingface.co/laion/DALLE2-PyTorch/resolve/main/decoder/1.5B_laion2B/latest.pth",
	"config_path": "https://huggingface.co/laion/DALLE2-PyTorch/raw/main/decoder/1.5B_laion2B/decoder_config.json"
},{
	"name": "Upsampler",
	"dalle2_install_path": "git+https://github.com/Veldrovive/DALLE2-pytorch@b2549a4d17244dab09e7a9496a9cb6330b7d3070",
	"decoder": [
			{
					"unets": [0],
					"model_path": "https://huggingface.co/laion/DALLE2-PyTorch/resolve/main/decoder/1.5B_laion2B/latest.pth",
					"config_path": "https://huggingface.co/laion/DALLE2-PyTorch/raw/main/decoder/1.5B_laion2B/decoder_config.json"
			},
			{
					"unets": [1],
					"model_path": "https://huggingface.co/Veldrovive/upsamplers/resolve/main/working/latest.pth",
					"config_path": "https://huggingface.co/Veldrovive/upsamplers/raw/main/working/decoder_config.json"
			}
	],
	"prior": {
			"model_path": "https://huggingface.co/zenglishuci/conditioned-prior/resolve/main/vit-l-14/prior_aes_finetune.pth",
			"config_path": ""
	}
}]


def load_state():
	state_path = "script_state.json"
	try:
			assert os.path.exists(state_path)
			with open(state_path, "r") as f:
					state = json.load(f)
			# Make sure the save config is up to date. You might think this is a stupid system but...
			decoder = state["decoder"]
			if decoder is not None:
					current_decoder_name = decoder["name"]
			try:
					current_decoder_index = decoder_options.index(current_decoder_name)
					state["decoder"] = decoder_versions[current_decoder_index]
			except ValueError:
					print("The decoder you were using no longer exists. Please pick a new option.")
					state["decoder"] = None
			
			# Check if models are where they say they are
			for filekey in ["decoder", "decoder_config", "prior", "prior_config"]:
					path = state["model_paths"][filekey]
					if path is not None and not os.path.exists(path):
							print(f"{filekey} not found in expected place. Removing decoder config.")
							state["decoder"] = None
							state["model_paths"] = {
								"decoder": None,
								"decoder_config": None,
								"prior": None,
								"prior_config": None
							}
							save_state()
	except Exception as e:
			state = {
					"text_input": '',
					"text_repeat": 3,
					"prior_conditioning": 1.0,
					"img_repeat": 1,
					"decoder_conditioning": 3.5,
					"include_prompt_checkbox": True,
					"upsample_checkbox": True,
					"decoder": None,
					"model_paths": {
							"decoder": None,
							"decoder_config": None,
							"prior": None,
							"prior_config": None
					}
			}
	return state


def save_state():
	global current_state
	state_path = "script_state.json"
	with open(state_path, "w") as f:
			json.dump(current_state, f)

def choice_equal(new_choice):
	global current_state
	if current_state["decoder"] is None:
			return False
	return current_state["decoder"]["decoder_path"] == new_choice["decoder_path"]

def dalle2_imported():
	return "dalle2_pytorch" in sys.modules


def setup(decoder_version_name):
	global current_state
	global chosen_decoder
	new_choice = decoder_versions[decoder_options.index(decoder_version_name)]
	current_choice = current_state["decoder"]
	
	new_choice_equal = choice_equal(new_choice)
	already_imported = dalle2_imported()
	
	requires_restart = not new_choice_equal and already_imported  # The wrong dalle2_pytorch version is already imported
	requires_download = not new_choice_equal  # The wrong decoder version is downloaded
	
	print(f"You are using the model {new_choice['name']} which will be downloaded from {new_choice['decoder_path']}\n")
	if requires_restart:
			print("You environment already has dalle2 imported and collab requires a restart for you to be able to import the new version.")
			print("Restart your runtime to proceed.")
	elif requires_download:
			print("The models are not downloaded. They will be when you proceed.")
	else:
			print("You are ready to run inference. If you suspect your models are out of date, force update them.")
	
	chosen_decoder = new_choice


def conditioned_on_text(config):
	try:
			return config.decoder.unets[0].cond_on_text_encodings
	except AttributeError:
			pass
	
	try:
			return config.decoder.condition_on_text_encodings
	except AttributeError:
			pass
	
	return False


def load_decoder(decoder_state_dict_path, config_file_path):
	config = train_configs.TrainDecoderConfig.from_json_path(config_file_path)
	global decoder_text_conditioned
	decoder_text_conditioned = conditioned_on_text(config)
	global clip_config
	clip_config = config.decoder.clip
	config.decoder.clip = None
	# print("Decoder conditioned on text", decoder_text_conditioned)
	decoder = config.decoder.create().to(device)
	decoder_state_dict = torch.load(decoder_state_dict_path, map_location='cpu')
	decoder.load_state_dict(decoder_state_dict, strict=False)
	del decoder_state_dict
	decoder.eval()
	return decoder

def load_prior(model_path):
	prior_network = DiffusionPriorNetwork(
			dim=768,
			depth=24,
			dim_head=64,
			heads=32,
			normformer=True,
			attn_dropout=5e-2,
			ff_dropout=5e-2,
			num_time_embeds=1,
			num_image_embeds=1,
			num_text_embeds=1,
			num_timesteps=1000,
			ff_mult=4
	)

	diffusion_prior = DiffusionPrior(
			net=prior_network,
			clip=OpenAIClipAdapter("ViT-L/14"),
			image_embed_dim=768,
			timesteps=1000,
			cond_drop_prob=0.1,
			loss_type="l2",
			condition_on_text_encodings=True,
	).to(device)

	state_dict = torch.load(model_path, map_location='cpu')
	if 'ema_model' in state_dict:
			print('Loading EMA Model')
			diffusion_prior.load_state_dict(state_dict['ema_model'], strict=True)
	else:
			print('Loading Standard Model')
			diffusion_prior.load_state_dict(state_dict['model'], strict=False)
	del state_dict
	return diffusion_prior

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

	if args.dataset != 'imagenet':
			# make sure we don't resize outside the model as this influences threat model
			preprocessor_without_normalize = transforms.ToTensor()
	print(f'[resizer] {resizer}')
	print(f'[preprocessor] {preprocessor_without_normalize}')

	model.eval()
	model.to(main_device)


	decoder_options = [version["name"] for version in decoder_versions]
	current_state = load_state()
	chosen_decoder = current_state["decoder"] if current_state["decoder"] is not None else decoder_versions[-1]

	decoder_version_dropdown = widgets.Dropdown(
			options=decoder_options,
			value=chosen_decoder["name"],
			description='Decoder:',
			disabled=False,
	)

	start_setup_button = widgets.Button(
			description="Setup"
	)

	redownload_button = widgets.Button (
			description="Force Update Models"
	)

	main_layout = widgets.VBox([decoder_version_dropdown, start_setup_button, redownload_button])

	decoder_text_conditioned = False
	clip_config = None

	decoder = load_decoder(current_state["model_paths"]["decoder"], current_state["model_paths"]["decoder_config"])
	diffusion_prior = load_prior(current_state["model_paths"]["prior"])

	#@title
	dalle2_clip = None
	if clip_config is not None:
			dalle2_clip = clip_config.create()
			print('dalle2_clip loaded!')

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
				embedding_text_temp = []
				assert args.dataset == 'imagenet', 'ensemble only implemented for imagenet'
				with open('CLIP_eval/zeroshot-templates.json', 'r') as f:
						templates = json.load(f)
				templates = templates['imagenet1k']
				print(f'[templates] {templates}')
				embedding_text_labels_norm = []
				text_embed_classes = []
				text_encoding_classes = []
				for c in IMAGENET_1K_CLASS_ID_TO_LABEL.values():
					texts = [template.format(c=c) for template in templates]
					text_tokens = open_clip.tokenize(texts).to(main_device)
					class_embeddings = model.encode_text(text_tokens)
					text_embed, text_encodings, _ = dalle2_clip.embed_text(text_tokens)
					text_embed_classes.append(text_embed.mean(dim=0))
					text_encoding_classes.append(text_encodings.mean(dim=0))
					class_embedding = F.normalize(class_embeddings, dim=-1)#.mean(dim=0)
					embedding_text_temp.append(class_embedding / class_embedding.norm())
					class_embedding = class_embedding.mean(dim=0)
					class_embedding /= class_embedding.norm()
					embedding_text_labels_norm.append(class_embedding)
				embedding_text_labels_norm = torch.stack(embedding_text_labels_norm, dim=1).to(main_device)
				embedding_text_temp = torch.stack(embedding_text_temp, dim=2).to(main_device)
				text_embed_classes = torch.stack(text_embed_classes, dim=1)
				text_encoding_classes = torch.stack(text_encoding_classes, dim=1)

			if clip_model_name == 'ViT-B-32':
					assert embedding_text_labels_norm.shape == (512, num_classes), embedding_text_labels_norm.shape
			elif clip_model_name == 'ViT-L-14':
					assert embedding_text_labels_norm.shape == (768, num_classes), embedding_text_labels_norm.shape
			else:
					raise ValueError(f'Unknown model: {clip_model_name}')


	model = ClassificationModel(
			model=model,
			dalle2_clip=dalle2_clip,
			diffusion_prior=diffusion_prior,
			decoder=decoder,
			text_embedding=embedding_text_labels_norm,
			text_embedding_temp=embedding_text_temp,
			text_embed_classes=text_embed_classes,
			text_encoding_classes=text_encoding_classes,
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
					# reload the run to make sure we have the latest summary
					del api, run_train
					api = wandb.Api()
					run_train = api.run(f'{wandb_user}/{wandb_project}/{args.wandb_id}')
					eps_descr = str(int(eps * 255)) if args.norm == 'linf' else str(eps)
					run_train.summary.update({f'rb/acc-{dataset_short}': clean_acc})
					run_train.summary.update({f'rb/racc-{dataset_short}-{args.norm}-{eps_descr}': robust_acc})
					run_train.update()
	else:
			adversary = AutoAttack(
					model, norm=args.norm.replace('l', 'L'), eps=eps, version='custom', attacks_to_run=attacks_to_run,
					alpha=args.alpha, verbose=True
			)

			x_test, y_test = load_clean_dataset(
					BenchmarkDataset(args.dataset), n_examples=n_samples, data_dir=data_dir,
					prepr=preprocessor_without_normalize,)
			x_adv, y_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size, return_labels=True)  # y_adv are preds on x_adv
			# results = {'x_val': x_test, 'x_adv': x_adv, 'y_val': y_test, 'y_adv': y_adv}
			
			if args.save_images:
					# save the adversarial images
					img_save_path = (f'/path/to/save/dir/'
												f'{args.dataset}/{args.wandb_id}-{args.pretrained}-{args.norm}-{eps:.3f}-'
												f'alph{args.alpha:.3f}-{n_samples}smpls-{time.strftime("%Y-%m-%d_%H-%M-%S")}')
					os.makedirs(img_save_path, exist_ok=True)

					print(f'[saving images to] {img_save_path}')
					x_adv = x_adv.detach().cpu()
					y_adv = y_adv.detach().cpu()
					x_clean = x_test.detach().cpu()
					y_clean = y_test.detach().cpu()
					torch.save(x_adv, f'{img_save_path}/x_adv.pt')
					torch.save(y_adv, f'{img_save_path}/y_adv.pt')
					torch.save(x_clean, f'{img_save_path}/x_clean.pt')
					torch.save(y_clean, f'{img_save_path}/y_clean.pt')
					with open(f'{img_save_path}/args.json', 'w') as f:
							json.dump(vars(args), f)
					with open(f'{img_save_path}/results.json', 'w') as f:
							f.write(f"acc:{acc:.2f}%")
							f.write(f"Racc:{racc:.2f}%")

			# write to wandb
			if run_train is not None:
					# reload the run to make sure we have the latest summary
					del api, run_train
					api = wandb.Api()
					run_train = api.run(f'{wandb_user}/{wandb_project}/{args.wandb_id}')
					if args.dataset == 'imagenet':
							assert args.norm == 'linf'
							eps_descr = str(int(eps * 255))
							if eps_descr == '4':
								descr = dataset_short
							else:
								descr = f'{dataset_short}-eps{eps_descr}'
							if n_samples != 5000:
								acc = f'{acc:.2f}*'
								racc = f'{racc:.2f}*'
					elif args.dataset == 'cifar10':
							if args.norm == 'linf':
								descr = dataset_short
							else:
								descr = f'{dataset_short}-{args.norm}'
							if n_samples != 10000:
								acc = f'{acc:.2f}*'
								racc = f'{racc:.2f}*'
					else:
							raise ValueError(f'Unknown dataset: {args.dataset}')
					run_train.summary.update({f'aa/acc-{dataset_short}': acc})
					run_train.summary.update({f'aa/racc-{descr}': racc})
					run_train.summary.update()
					run_train.update()
	run_eval.finish()













