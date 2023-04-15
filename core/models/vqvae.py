import torch
import torch.nn.functional as F
import torch.nn as nn
from x_transformers.x_transformers import AttentionLayers, Encoder, Decoder, exists, default, always,ScaledSinusoidalEmbedding,AbsolutePositionalEmbedding, l2norm
from vector_quantize_pytorch import ResidualVQ, VectorQuantize
from einops import rearrange, reduce, pack, unpack
from core.quantization.core_vq import VectorQuantization

class LinearEmbedding(nn.Module):
	def __init__(self, input_dim,dim, l2norm_embed = False):
		super().__init__()
		self.l2norm_embed = l2norm_embed
		self.emb = nn.Linear(input_dim, dim)

	def forward(self, x):
		linear_emb = self.emb(x)
		return l2norm(linear_emb) if self.l2norm_embed else linear_emb

class MotionTransformer(nn.Module):
	def __init__(
		self,
		*,
		max_seq_len,
		attn_layers,
		emb_dim = None,
		inp_dim = None,
		max_mem_len = 0.,
		shift_mem_down = 0,
		emb_dropout = 0.,
		post_emb_norm = False,
		num_memory_tokens = None,
		tie_embedding = False,
		logits_dim = None,
		use_abs_pos_emb = True,
		scaled_sinu_pos_emb = False,
		l2norm_embed = False,
		emb_frac_gradient = 1. # GLM-130B and Cogview successfully used this, set at 0.1
		):
		super().__init__()
		assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

		dim = attn_layers.dim
		emb_dim = default(emb_dim, dim)
		self.emb_dim = emb_dim

		self.max_seq_len = max_seq_len
		self.logits_dim = logits_dim
	  

		self.l2norm_embed = l2norm_embed
		
		self.token_emb = LinearEmbedding(inp_dim, emb_dim, l2norm_embed = l2norm_embed)

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
		self.attn_layers = attn_layers
		self.norm = nn.LayerNorm(dim)

		self.init_()
		
		if exists(logits_dim):
			self.to_logits = nn.Linear(dim, logits_dim)


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
		return_embeddings = False,
		return_logits_and_embeddings = False,
		return_intermediates = False,
		mask = None,
		return_attn = False,
		pos = None,
		prepend_embeds = None,
		sum_embeds = None,
		**kwargs
	):
		b, n,d, device, emb_frac_gradient = *x.shape, x.device, self.emb_frac_gradient
		return_hiddens = return_attn
		
		if not exists(self.logits_dim):
			return_embeddings = True

		# absolute positional embedding

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
			assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as text model dimensions'

			x = torch.cat((prepend_embeds, x), dim = -2)

		# whether to reduce the gradient going to the embedding, from cogview paper, corroborated by GLM-130B model

		if emb_frac_gradient < 1:
			assert emb_frac_gradient > 0
			x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)

		# embedding dropout

		x = self.emb_dropout(x)

		x = self.project_emb(x)

		if return_hiddens:
			x, intermediates = self.attn_layers(x, mask = mask, return_hiddens = True, **kwargs)
		else:
			x = self.attn_layers(x, mask = mask, **kwargs)

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
	

class MotionDecoder(nn.Module):


	def __init__(self, dim, logit_dim , attn_layers):
		super(MotionDecoder , self).__init__()
		
		self.to_logit = nn.Linear(dim,logit_dim)
		self.attn_layers = attn_layers
		

	def forward(self, x , mask = None):
		"""x: b n c"""
		
	   
		x = self.attn_layers(x , mask = mask)
		logits = self.to_logit(x)

		return logits
	




class VQMotionModel(nn.Module):
	"""Audio Motion VQGAN model."""

	def __init__(self, args, device = "cuda"):
		"""Initializer for VQGANModel.

		Args:
		config: `VQGANModel` instance.
		is_training: bool. true for training model, false for eval model. Controls
			whether dropout will be applied.
		"""
		super(VQMotionModel , self).__init__()

		self.device = device
		self.motionEncoder = MotionTransformer(
			inp_dim = args.motion_dim,
			max_seq_len = args.max_seq_length,
			scaled_sinu_pos_emb = True,
			attn_layers = Encoder(
				dim = args.enc_dec_dim,
				depth = args.depth,
				heads = args.heads,
				
			)
		)
				
		self.motionDecoder = MotionDecoder(
			dim = args.enc_dec_dim,
			logit_dim = args.motion_dim,
			attn_layers = Decoder(
					dim = args.enc_dec_dim,
					depth = args.depth,
					heads = args.heads,
				)
			)
		
		# self.rq = ResidualVQ(
		# 	dim = 128,
		# 	num_quantizers = 8,
		# 	codebook_size = 1024,
		# 	decay = 0.95,
		# 	commitment_weight = 1,
		# 	kmeans_init = True,
		# 	threshold_ema_dead_code = 2,
		# 	quantize_dropout = True,
		# 	quantize_dropout_cutoff_index = 1
		# )
		self.vq = VectorQuantization(
			dim = args.enc_dec_dim,
			codebook_dim = args.codebook_dim,
			codebook_size = args.codebook_size,
			decay = 0.95,
			commitment_weight = 1,
			kmeans_init = True,
			threshold_ema_dead_code = 2,
		)

	def forward(self, motion , mask = None):
		"""Predict sequences from inputs. 

		This is a single forward pass that been used during training. 

		Args:
			inputs: Input dict of tensors. The dict should contains 
			`motion_input` ([batch_size, motion_seq_length, motion_feature_dimension])

		Returns:
			Final output after the cross modal transformer. A tensor with shape 
			[batch_size, motion_seq_length, motion_feature_dimension]
		"""
		# Computes motion features.
		motion_input = motion #b n d
		
		
		embed_motion_features = self.motionEncoder(motion_input , mask = mask) #b n d

		##codebook
		quantized_enc_motion, indices, commit_loss = self.vq(embed_motion_features,mask = mask)
		# b n d , b n q , q
		
		## decoder
		decoded_motion_features = self.motionDecoder(quantized_enc_motion,mask = mask) # b n d
		

		return decoded_motion_features , indices, commit_loss.sum()


	def encode(self, motion_input , mask = None):

		with torch.no_grad():
			embed_motion_features = self.motionEncoder(motion_input,mask = mask)
			indices = self.vq.encode(embed_motion_features)
			return indices

	def decode(self, indices):

		with torch.no_grad():
			quantized = self.vq.decode(indices)
			out_motion = self.motionDecoder(quantized)
			return quantized, out_motion




	# def decode(self, indices):
	# 	all_codes =  self.rq.get_codes_from_indices(indices)
	# 	quantized_enc_motion = torch.sum(all_codes,0)
	# 	decoded_motion_features = self.motionDecoder(quantized_enc_motion)
	# 	return decoded_motion_features

	# def decode_from_codebook_indices(self, quantized_indices):
	# 		## quantized_indices batch,seq,quantize_dim
	# 		codes = self.rq.get_codes_from_indices(quantized_indices)
	# 		x = reduce(codes, 'q ... -> ...', 'sum')
	# 		# x = rearrange(x, 'b n c -> b c n')
	# 		return codes , self.motionDecoder(x)
	
	