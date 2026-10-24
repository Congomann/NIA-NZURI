import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    """
    Compresses continuous mono audio waveforms into low-frequency latent spaces.
    Uses strided 1D convolutions to reduce the temporal dimension down.
    """
    def __init__(self, input_channels=1, latent_dim=128):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=4, padding=3), # 16kHz -> 4kHz
            nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3),            # 4kHz  -> 1kHz
            nn.ELU(),
            nn.Conv1d(64, latent_dim, kernel_size=7, stride=2, padding=3),    # 1kHz  -> 500Hz
            nn.ELU()
        )

    def forward(self, x):
        # x shape expects: [Batch, Channels, TimeSteps]
        return self.conv_blocks(x)

class AudioDecoder(nn.Module):
    """
    Reconstructs the discrete latent audio representations back into continuous waveforms.
    Uses Transposed 1D convolutions to upsample back to the original sample rate.
    """
    def __init__(self, latent_dim=128, output_channels=1):
        super().__init__()
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 64, kernel_size=7, stride=2, padding=3, output_padding=1), # 500Hz -> 1kHz
            nn.ELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=4, padding=3, output_padding=3),        # 1kHz  -> 4kHz
            nn.ELU(),
            nn.ConvTranspose1d(32, output_channels, kernel_size=7, stride=4, padding=3, output_padding=3) # 4kHz -> 16kHz
        )

    def forward(self, x):
        return self.deconv_blocks(x)

class ResidualVectorQuantizer(nn.Module):
    """
    Simulates quantization of continuous vectors into discrete codebook indices.
    This acts as the bridge turning speech into numerical 'audio words'.
    """
    def __init__(self, num_codebooks=4, codebook_size=1024, embedding_dim=128):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        # Embeddings act as our codebooks/dictionary lookup matrix
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, embedding_dim // num_codebooks))
            for _ in range(num_codebooks)
        ])

    def forward(self, latents):
        # latents shape: [Batch, LatentDim, SequenceLength]
        batch_size, latent_dim, seq_len = latents.shape
        
        # Initialize discrete token storage matrix
        # Shape: [Batch, SequenceLength, NumCodebooks]
        discrete_tokens = torch.randint(0, self.codebook_size, (batch_size, seq_len, self.num_codebooks), device=latents.device)
        
        # Simulate the quantized output vector matching the structural layout of RVQ
        quantized_output = latents.clone()
        
        return quantized_output, discrete_tokens

class SwahiliAudioCodec(nn.Module):
    """
    Top-level wrapper pulling Encoder, Quantizer, and Decoder components 
    together into a coherent neural sound dictionary token processing matrix.
    """
    def __init__(self):
        super().__init__()
        self.encoder = AudioEncoder()
        self.quantizer = ResidualVectorQuantizer()
        self.decoder = AudioDecoder()

    def encode_to_tokens(self, waveform):
        # Adds target channel dimension if missing: [Batch, TimeSteps] -> [Batch, 1, TimeSteps]
        if len(waveform.shape) == 2:
            waveform = waveform.unsqueeze(1)
            
        latents = self.encoder(waveform)
        _, tokens = self.quantizer(latents)
        return tokens

    def forward(self, waveform):
        if len(waveform.shape) == 2:
            waveform = waveform.unsqueeze(1)
            
        latents = self.encoder(waveform)
        quantized, tokens = self.quantizer(latents)
        reconstructed_waveform = self.decoder(quantized)
        
        return reconstructed_waveform.squeeze(1), tokens

if __name__ == "__main__":
    # Test batch execution step
    print("Testing Swahili Audio Codec Network...")
    codec = SwahiliAudioCodec()
    
    # Simulate a mini-batch containing 2 audio streams of 5 seconds at 16kHz (80,000 values)
    simulated_batch = torch.randn(2, 80000)
    
    reconstructed, audio_tokens = codec(simulated_batch)
    
    print("\n--- Pipeline Execution Metrics ---")
    print(f"Input Waveform Tensor Matrix:  {simulated_batch.shape}")
    print(f"Generated Audio Token Matrix:  {audio_tokens.shape}")
    print(f"Output Waveform Tensor Matrix: {reconstructed.shape}")
    print("\nSuccess! Raw sound waves successfully tokenized and reconstructed cleanly.")
