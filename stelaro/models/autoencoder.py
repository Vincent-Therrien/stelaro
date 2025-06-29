"""
    Autoencoder neural network module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2025
    - License: MIT
"""

from torch.nn import Module, Conv1d, ReLU, Sequential, Flatten, Linear


class SequenceAutoencoderClassifier(nn.Module):
    def __init__(self, input_dim=4, latent_dim=128, num_classes=3):
        super().__init__()

        # Encoder (1D Conv -> Pool -> FC)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # [B, 32, L/2]
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),      # [B, 64, 1]
        )
        self.latent_layer = nn.Linear(64, latent_dim)

        # Decoder (for reconstruction)
        self.decoder_fc = nn.Linear(latent_dim, 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, input_dim, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Assuming input in [0,1] (e.g. one-hot or normalized)
        )

        # Classifier head
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        # x: [B, L, 4] → transpose to [B, 4, L] for Conv1d
        x = x.transpose(1, 2)
        z = self.encoder(x).squeeze(-1)      # [B, 64]
        latent = self.latent_layer(z)        # [B, latent_dim]

        # Classifier path
        logits = self.classifier(latent)     # [B, num_classes]

        # Decoder path
        decoded_input = self.decoder_fc(latent).unsqueeze(-1)  # [B, 64, 1]
        reconstructed = self.decoder(decoded_input)             # [B, 4, L]
        reconstructed = reconstructed.transpose(1, 2)           # [B, L, 4]

        return logits, reconstructed

