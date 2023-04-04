from vector_quantize_pytorch import ResidualVQ, VectorQuantize

import math
import typing as tp

import torch
from torch import nn

class VectorQuantization(nn.Module):
	def __init__(
		self,
		dimension: int = 128,
		codebook_size: int = 1024,
		decay: float = 0.99,
		kmeans_init: bool = True,
		kmeans_iters: int = 50,
		threshold_ema_dead_code: int = 2,
		sync_codebook = False

	):
		super().__init__()
		self.dimension = dimension
		self.codebook_size= codebook_size
		self.decay = decay
		self.kmeans_init = kmeans_init
		self.kmeans_iters = kmeans_iters
		self.threshold_ema_dead_code = threshold_ema_dead_code
		self.sync_codebook = sync_codebook
		self.vq = VectorQuantize(
			dim=self.dimension,
			codebook_size=self.codebook_size,
			decay=self.decay,
			kmeans_init=self.kmeans_init,
			kmeans_iters=self.kmeans_iters,
			threshold_ema_dead_code=self.threshold_ema_dead_code,
			sync_codebook = self.sync_codebook
		)


	def forward(self , motion_input:torch.Tensor):
		quantize, embed_ind, loss = self.vq(motion_input)
		return quantize, embed_ind, loss
	
	def encode(self, x):
		x = self.vq.project_in(x)
		embed_in = self.vq._codebook.encode(x)
		return embed_in

	def decode(self, indices):

		quantized = self.vq.decode(indices)
		return quantized

