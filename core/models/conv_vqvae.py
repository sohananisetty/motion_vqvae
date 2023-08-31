import torch
import torch.nn.functional as F
import torch.nn as nn

# from x_transformers.x_transformers import AttentionLayers, Encoder, Decoder, exists, default, always,ScaledSinusoidalEmbedding,AbsolutePositionalEmbedding, l2norm
from core.quantization.core_vq import VectorQuantization
from core.models.attention import (
    AttentionLayers,
    Decoder,
    Encoder,
    ScaledSinusoidalEmbedding,
    AbsolutePositionalEmbedding,
    exists,
    default,
    always,
    l2norm,
)

from core.quantization.vector_quantize_pytorch import VectorQuantize
from core.models.encdec import Decoder, Encoder


class ConvVQMotionModel(nn.Module):
    """Audio Motion VQGAN model."""

    def __init__(self, args, device="cuda"):
        """Initializer for VQGANModel.

        Args:
        config: `VQGANModel` instance.
        is_training: bool. true for training model, false for eval model. Controls
                whether dropout will be applied.
        """
        super(ConvVQMotionModel, self).__init__()

        self.device = device
        self.dim = args.motion_dim

        self.motionEncoder = Encoder(
            input_emb_dim=args.motion_dim,
            down_t=3,
            stride_t=2,
            dim=args.enc_dec_dim,
            depth=3,
            dilation_growth_rate=3,
            activation="relu",
            norm=None,
        )

        self.motionDecoder = Decoder(
            input_emb_dim=args.motion_dim,
            down_t=3,
            stride_t=2,
            dim=args.enc_dec_dim,
            depth=3,
            dilation_growth_rate=3,
            activation="relu",
            norm=None,
        )

        # self.vq = VectorQuantization(
        # 	dim = args.enc_dec_dim,
        # 	codebook_dim = args.codebook_dim,
        # 	codebook_size = args.codebook_size,
        # 	decay = 0.95,
        # 	commitment_weight = 5,
        # 	kmeans_init = True,
        # 	threshold_ema_dead_code = 2,
        # )

        self.vq = VectorQuantize(
            dim=args.enc_dec_dim,
            codebook_size=args.codebook_size,  # codebook size
            decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=5.0,  # the weight on the commitment loss
            kmeans_init=True,  # set to True
            kmeans_iters=10,
            threshold_ema_dead_code=2,
            sync_codebook=True,
            sync_affine_param=True,
            sync_kmeans=True,
            learnable_codebook=True,
            in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.SGD(
                *args, **kwargs, lr=1.0, momentum=0.9
            ),
            affine_param=True,
            affine_param_batch_decay=0.99,
            affine_param_codebook_decay=0.9,
            ema_update=False,
        )

    def forward(self, motion, mask=None):
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
        motion_input = motion  # b n d

        embed_motion_features = self.motionEncoder(motion_input)  # b n d

        ##codebook
        quantized_enc_motion, indices, commit_loss = self.vq(embed_motion_features)
        # b n d , b n q , q

        ## decoder
        decoded_motion_features = self.motionDecoder(quantized_enc_motion)  # b n d

        return decoded_motion_features, indices, commit_loss.sum()

    def encode(self, motion_input):
        with torch.no_grad():
            embed_motion_features = self.motionEncoder(motion_input)
            quantized_enc_motion, indices, commit_loss = self.vq(embed_motion_features)
            return indices

    def decode(self, indices):
        with torch.no_grad():
            quantized = self.vq.get_codes_from_indices(indices).reshape(
                quantized.shape[0], -1, self.dim
            )
            out_motion = self.motionDecoder(quantized)
            return quantized, out_motion
