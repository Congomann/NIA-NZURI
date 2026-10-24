import os
import sys

# Ensure src directory is accessible on the Python module path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from data_pipeline.dataset import get_swahili_dataloader
from core_model.tokenizer import SwahiliAudioCodec
from core_model.transformer import SwahiliS2STransformer

def train_s2s_model(data_dir, checkpoint_dir="./checkpoints", epochs=10, batch_size=8, lr=1e-4):
    """
    Unified training loop running directly on RunPod Cloud GPU instances.
    """
    # 1. Device Allocation Layer (NVIDIA CUDA -> Apple Silicon Metal MPS -> CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Executing training matrix on device architecture: {device}")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 2. Instantiate Data Streams
    dataloader = get_swahili_dataloader(data_dir, batch_size=batch_size, shuffle=True)

    # 3. Instantiate Neural Network Architecture Nodes
    codec = SwahiliAudioCodec().to(device)
    codec.eval() # The Neural Audio Codec stays fixed during transformer training
    
    model = SwahiliS2STransformer(num_tokens=1024, num_codebooks=4).to(device)
    model.train()

    # 4. Objective Functions and Optimization Algorithms
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Auto-resume from latest checkpoint if available
    start_epoch = 0
    available_pts = sorted(
        [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    ) if os.path.exists(checkpoint_dir) else []
    if available_pts:
        latest_ckpt = available_pts[-1]
        try:
            ckpt_data = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(ckpt_data['model_state_dict'])
            optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
            start_epoch = ckpt_data.get('epoch', 0)
            print(f"Resuming training from existing checkpoint: {latest_ckpt} (Completed epoch: {start_epoch})")
        except Exception as e:
            print(f"Warning: Could not resume from {latest_ckpt}: {e}. Starting fresh.")

    print("\n=== Initializing Training Operations ===")
    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        
        for batch_idx, waveforms in enumerate(dataloader):
            # Move raw float waves to target compute memory [Batch, 80000]
            waveforms = waveforms.to(device)
            
            # Extract discrete training tokens via our neural sound dictionary
            with torch.no_grad():
                # Expected token output structure: [Batch, SeqLen, NumCodebooks]
                tokens = codec.encode_to_tokens(waveforms)
            
            # Slice tokens to build target causal inputs and labels
            # src: what the user said; tgt_input: auto-regressive teacher forcing block
            src_tokens = tokens[:, :60, :] 
            tgt_tokens = tokens[:, 60:, :]
            
            tgt_input = tgt_tokens[:, :-1, :]
            tgt_target = tgt_tokens[:, 1:, :]

            # Zero network gradients
            optimizer.zero_grad()

            # Execute forward mathematical mapping matrix pass
            # Output: list of 4 tensor matrices representing each codebook stream track
            logits_list = model(src_tokens, tgt_input)

            # Accumulate loss across all discrete codebook branches
            loss = 0.0
            for i in range(len(logits_list)):
                # Reshape matrix mappings to conform to CrossEntropy target structures
                loss += criterion(
                    logits_list[i].reshape(-1, 1024), 
                    tgt_target[:, :, i].reshape(-1)
                )

            # Backpropagation execution pass
            loss.backward()
            
            # Prevent exploding gradients over high-resource computing matrices
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Iterative Stream Loss: {loss.item():.4f}")

        epoch_loss = running_loss / max(1, len(dataloader))
        print(f"--- Epoch {epoch+1} Completed. Mean Metric Target Loss: {epoch_loss:.4f} ---")
        
        # Save model weight state mappings back to persistent disk layers
        checkpoint_path = os.path.join(checkpoint_dir, f"swahili_s2s_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved securely at: {checkpoint_path}")

if __name__ == "__main__":
    # Path mappings point to the outputs generated by your audio preprocessing pipeline script
    PROCESSED_DATA_DIR = "./data/processed_training_chunks"
    train_s2s_model(PROCESSED_DATA_DIR, epochs=5, batch_size=4)
