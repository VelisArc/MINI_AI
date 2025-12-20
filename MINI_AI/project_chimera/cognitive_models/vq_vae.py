# project_chimera/cognitive_models/vq_vae.py
from ..nn.module import Module
from ..nn.conv2d import Conv2d
from ..nn.conv_transpose2d import ConvTranspose2d
from ..nn.resnet_block import ResnetBlock
from ..nn.vector_quantizer import VectorQuantizer
from ..nn.activations import ReLU
from ..nn.containers import Sequential

class Encoder(Module):
    def __init__(self, in_channels, hidden_channels, num_resnet_blocks):
        super().__init__()
        self.model = Sequential(
            Conv2d(in_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            ReLU(),
            Conv2d(hidden_channels // 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            ReLU(),
            Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            *[ResnetBlock(hidden_channels, hidden_channels) for _ in range(num_resnet_blocks)]
        )
    def forward(self, x):
        return self.model(x)

class Decoder(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_resnet_blocks):
        super().__init__()
        self.model = Sequential(
            Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            *[ResnetBlock(hidden_channels, hidden_channels) for _ in range(num_resnet_blocks)],
            ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            ReLU(),
            ConvTranspose2d(hidden_channels // 2, out_channels, kernel_size=4, stride=2, padding=1)
        )
    def forward(self, x):
        return self.model(x)

class VQVAE(Module):
    def __init__(
        self, in_channels=3, hidden_channels=128, num_resnet_blocks=2,
        num_embeddings=512, embedding_dim=64, commitment_cost=0.25
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, num_resnet_blocks)
        self.pre_vq_conv = Conv2d(hidden_channels, embedding_dim, kernel_size=1, padding=0)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_channels, in_channels, num_resnet_blocks)

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        quantized, vq_loss, indices = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, indices
