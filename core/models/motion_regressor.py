import torch
import torch.nn.functional as F
import torch.nn as nn
from math import ceil

from x_transformers.x_transformers import AttentionLayers, Encoder, Decoder, exists, default, always,ScaledSinusoidalEmbedding,AbsolutePositionalEmbedding, l2norm
from einops import rearrange, reduce, pack, unpack
from tqdm import tqdm



def exists(val):
	return val is not None

# nucleus

def top_p(logits, thres = 0.9):
	sorted_logits, sorted_indices = torch.sort(logits, descending=True)
	cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

	sorted_indices_to_remove = cum_probs > (1 - thres)
	sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
	sorted_indices_to_remove[:, 0] = 0

	sorted_logits[sorted_indices_to_remove] = float('-inf')
	return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk

def top_k(logits, thres = 0.9):
	k = ceil((1 - thres) * logits.shape[-1])
	val, ind = torch.topk(logits, k)
	probs = torch.full_like(logits, float('-inf'))
	probs.scatter_(1, ind, val)
	return probs

# top_a

def top_a(logits, min_p_pow=2.0, min_p_ratio=0.02):
	probs = F.softmax(logits, dim=-1)
	limit = torch.pow(torch.max(probs), min_p_pow) * min_p_ratio
	logits[probs < limit] = float('-inf')
	logits[probs >= limit] = 1
	return logits


class TokenEmbedding(nn.Module):
	def __init__(self, dim, num_tokens, l2norm_embed = False , pad_idx = 0):
		super().__init__()
		self.l2norm_embed = l2norm_embed
		self.emb = nn.Embedding(num_tokens, dim , padding_idx = pad_idx)

	def forward(self, x):
		token_emb = self.emb(x)
		return l2norm(token_emb) if self.l2norm_embed else token_emb

class MotionTokenTransformer(nn.Module):

	def __init__(
		self,
		*,
		num_tokens,
		max_seq_len=None,
		attn_layers,
		emb_dim = None,
		cond_dim = None,
		emb_dropout = 0.,
		post_emb_norm = False,
		logits_dim = None,
		use_abs_pos_emb = True,
		scaled_sinu_pos_emb = False,
		l2norm_embed = False,
		pad_idx = 1,
		emb_frac_gradient = 1. # GLM-130B and Cogview successfully used this, set at 0.1
		):
		super().__init__()
		assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

		dim = attn_layers.dim
		emb_dim = default(emb_dim, dim)
		self.emb_dim = emb_dim
		self.cond_emb_dim = cond_dim

		self.max_seq_len = max_seq_len



		self.l2norm_embed = l2norm_embed
		
		self.token_emb = TokenEmbedding(emb_dim, num_tokens, l2norm_embed = l2norm_embed , pad_idx=pad_idx)

		if not (use_abs_pos_emb and not attn_layers.has_pos_emb):
			self.pos_emb = always(0)
		elif scaled_sinu_pos_emb:
			self.pos_emb = ScaledSinusoidalEmbedding(emb_dim)
		else:
			self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len, l2norm_embed = l2norm_embed)

		self.emb_frac_gradient = emb_frac_gradient # fraction of the gradient that should go to the embedding, https://arxiv.org/abs/2105.13290

		self.post_emb_norm = nn.LayerNorm(emb_dim) if post_emb_norm else nn.Identity()
		self.emb_dropout = nn.Dropout(emb_dropout)

		self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
		
		if exists(cond_dim):
			self.project_cond_emb = nn.Linear(self.cond_emb_dim, self.emb_dim) if self.cond_emb_dim != self.emb_dim else nn.Identity()
		
		self.attn_layers = attn_layers
		self.norm = nn.LayerNorm(self.emb_dim)

		self.init_()
		logits_dim = default(logits_dim, num_tokens)
		self.logits_dim = logits_dim

		if exists(logits_dim):
			self.to_logits = nn.Linear(self.emb_dim, self.logits_dim)


	def init_(self):
		if self.l2norm_embed:
			nn.init.normal_(self.token_emb.emb.weight, std = 1e-5)
			if not isinstance(self.pos_emb, always):
				nn.init.normal_(self.pos_emb.emb.weight, std = 1e-5)
			return

		nn.init.kaiming_normal_(self.token_emb.emb.weight)

	def forward(
		self,
		x,
		context = None,
		mask = None,
		context_mask = None,
		attn_mask = None,
		self_attn_context_mask = None,
	
		return_embeddings = False,
		return_logits_and_embeddings = False,
		return_intermediates = False,
		return_attn = False,
		pos = None,
		prepend_embeds = None,
		sum_embeds = None,
		**kwargs
	):
  
		b, n, device, emb_frac_gradient = *x.shape, x.device, self.emb_frac_gradient
		return_hiddens = return_attn

		# absolute positional embedding
		if not exists(self.logits_dim):
			return_embeddings = True

		external_pos_emb = exists(pos) and pos.dtype != torch.long
		pos_emb = self.pos_emb(x, pos = pos) if not external_pos_emb else pos
		x = self.token_emb(x) + pos_emb

		# for summing embeddings passed externally - needs this for self-conditioning in non-autoregressive training

		if exists(sum_embeds):
			x = x + sum_embeds

		# post embedding norm, purportedly leads to greater stabilization

		x = self.post_emb_norm(x)

		# whether to append embeds, as in PaLI, for image embeddings

		if exists(prepend_embeds):
			prepend_seq, prepend_dim = prepend_embeds.shape[1:]
			assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as model dimensions'

			x = torch.cat((prepend_embeds, x), dim = -2)

		# whether to reduce the gradient going to the embedding, from cogview paper, corroborated by GLM-130B model

		if emb_frac_gradient < 1:
			assert emb_frac_gradient > 0
			x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)

		# embedding dropout

		x = self.emb_dropout(x)

		x = self.project_emb(x)
		
		if exists(context):
			context = self.project_cond_emb(context)
					

		if return_hiddens:
			x, intermediates = self.attn_layers(x,context, mask,context_mask,attn_mask,self_attn_context_mask, return_hiddens = True, **kwargs)
		else:
			x = self.attn_layers(x,context, mask,context_mask,attn_mask,self_attn_context_mask, **kwargs)

		x = self.norm(x)

		if return_logits_and_embeddings:
			out = (self.to_logits(x), x)
		elif return_embeddings:
			out = x
		else:
			out = self.to_logits(x)

		if return_intermediates:
			return out, intermediates

		if return_attn:
			attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
			return out, attn_maps

		return out



	
class MotionRegressorModel(nn.Module):
	"""Audio Motion VQGAN model."""

	def __init__(self, args, mask_prob = 0. , ignore_index = -100, pad_value = 0, device = "cuda"):
		"""Initializer for MotionRegressorModel.

		Args:
		config: `MotionRegressorModel` instance.
		is_training: bool. true for training model, false for eval model. Controls
			whether dropout will be applied.
		"""
		super(MotionRegressorModel , self).__init__()

		self.device = device
		self.pad_index = pad_value
		self.args = args
		self.max_seq_len = None




		# self.motionDecoder = MotionTokenTransformer(
		# max_seq_len = args.max_seq_length,
		# num_tokens = args.num_tokens,
		# cond_dim  =args.music_dim,
		# scaled_sinu_pos_emb = True,
		# pad_idx=self.pad_index,
		# attn_layers = Decoder(
		# 	cross_attn_tokens_dropout = 0.3,
		# 	cross_attend = True,
		# 	dim = args.dec_dim,
		# 	depth = args.depth,
		# 	heads = args.heads,
		# )
		# )


		# self.decoder = AutoregressiveWrapper(self.decoder, ignore_index=ignore_index, pad_value=pad_value)

		self.motionDecoder = MotionTokenTransformer(
		max_seq_len = None,
		num_tokens = args.num_tokens,
		cond_dim  =args.music_dim,
		use_abs_pos_emb = False,
		attn_layers = Decoder(
			cross_attend = True,
			cross_attn_tokens_dropout = 0.2,
			alibi_pos_bias = True, # turns on ALiBi positional embedding
			alibi_num_heads = 4 ,
			dim = args.dec_dim,
			depth = args.depth,
			heads = args.heads,
		)
		)


		assert mask_prob < 1.
		self.mask_prob = mask_prob



		




	@torch.no_grad()
	def generate(
		self,
		start_tokens,
		seq_len,
		eos_token = None,
		temperature = 1.,
		filter_logits_fn = top_k,
		filter_thres = 0.9,
		min_p_pow = 2.0,
		min_p_ratio = 0.02,
		context = None,
		context_mask = None,

		# **kwargs
	):
		num_dims = len(start_tokens.shape)

		if num_dims == 1:
			start_tokens = start_tokens[None, :]

		b, t = start_tokens.shape

		was_training = self.training
		self.eval()

		out = start_tokens

		for sl in tqdm(range(seq_len)):

			x = out

			# logits = self.forward(motion = x, context = context[:,:sl] , context_mask = context_mask[:,:sl])[:, -1]
			logits = self.forward(motion = x, context = context[:,:(sl+1)] , context_mask = context_mask[:,:(sl+1)])[:, -1]

			if filter_logits_fn in {top_k, top_p}:
				filtered_logits = filter_logits_fn(logits, thres = filter_thres)
				probs = F.softmax(filtered_logits / temperature, dim=-1)

			elif filter_logits_fn is top_a:
				filtered_logits = filter_logits_fn(logits, min_p_pow = min_p_pow, min_p_ratio= min_p_ratio)
				probs = F.softmax(filtered_logits / temperature, dim=-1)

			sample = torch.multinomial(probs, 1)

			out = torch.cat((out, sample), dim=-1)

			if exists(eos_token):
				is_eos_tokens = (out == eos_token)

				if is_eos_tokens.any(dim = -1).all():
					# mask out everything after the eos tokens
					shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
					mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
					out = out.masked_fill(mask, self.pad_value)
					break

		# out = out[:, t:]

		if num_dims == 1:
			out = out.squeeze(0)

		self.train(was_training)
		return out

	def forward(self, motion , mask = None , context = None,context_mask = None):
		"""Predict sequences from inputs. 

		This is a single forward pass that been used during training. 

		Args:
			inputs: Input dict of tensors. The dict should contains 
			`motion_input` ([batch_size, motion_seq_length-1])
			`mask` ([batch_size, motion_seq_length-1])
			`context` ([batch_size, motion_seq_length, music_feature_dimension])
			`context_mask` ([batch_size, motion_seq_length])

		Returns:
			Final output after the transformer. A tensor with shape 
			[batch_size, motion_seq_length, num_tokens]
		"""

		b , t  = motion.shape

		# if self.training:
		# 	inp, targets = motion[:, :-1], motion[:, 1:]
		
		

		# if self.mask_prob > 0. and self.training:
		# 	rand = torch.randn(motion.shape)
		# 	rand[:, 0] = -torch.finfo(rand.dtype).max # first token should not be masked out
		# 	num_mask = min(int(t * self.mask_prob), t - 1)
		# 	indices = rand.topk(num_mask, dim = -1).indices
		# 	token_mask = ~torch.zeros_like(motion).scatter(1, indices, 1.).bool()
		# 	mask = mask*token_mask

		logits = self.motionDecoder(x = motion, mask = mask , context = context , context_mask = context_mask)

		# print(logits.shape)


		# loss = F.cross_entropy(logits.contiguous().view(-1, logits.shape[-1]), targets.contiguous().view(-1),
		#     ignore_index=self.pad_index, reduction='mean')
		
		
		return logits