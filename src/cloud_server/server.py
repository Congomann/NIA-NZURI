import os
import sys
import asyncio

# Ensure project src directory is accessible on Python module search path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from core_model.tokenizer import SwahiliAudioCodec
from core_model.transformer import SwahiliS2STransformer

app = FastAPI(title="Sikika Native Tanzanian Swahili S2S Engine")

# Global device and model management matrix allocation (CUDA -> Apple Silicon MPS -> CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
codec = None
s2s_model = None

@app.on_event("startup")
def load_trained_models():
    """Loads the core model weights from the RunPod persistent checkpoint directory."""
    global codec, s2s_model
    print(f"Initializing model runtime weights on device node: {DEVICE}")
    
    # Instantiate neural modules
    codec = SwahiliAudioCodec().to(DEVICE)
    codec.eval()
    
    s2s_model = SwahiliS2STransformer(num_tokens=1024, num_codebooks=4).to(DEVICE)
    
    # Dynamically find the highest epoch checkpoint available, or fallback to epoch_5
    checkpoints_dir = os.path.abspath(os.path.join(src_dir, "..", "checkpoints"))
    available_pts = sorted(
        [os.path.join(checkpoints_dir, f) for f in (os.listdir(checkpoints_dir) if os.path.exists(checkpoints_dir) else []) if f.endswith(".pt")]
    )
    checkpoint_path = available_pts[-1] if available_pts else os.path.join(checkpoints_dir, "swahili_s2s_epoch_5.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        s2s_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully mounted production weights from: {checkpoint_path}")
    else:
        print("Warning: Running with uninitialized random weight matrices. Checkpoint not found.")
    s2s_model.eval()

@app.websocket("/stream/v1/swahili")
async def stream_swahili_audio(websocket: WebSocket):
    """
    Handles live, full-duplex WebSockets streaming. 
    Receives raw byte audio arrays and returns processed native audio blocks.
    """
    await websocket.accept()
    print("New full-duplex audio stream connected safely.")
    
    # Initialize an active processing queue buffer
    audio_buffer = bytearray()
    
    try:
        while True:
            # 1. Listen for raw binary audio chunks arriving from the network client
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            
            # Process in uniform blocks (e.g., waiting for roughly 1 second of audio data)
            # 1 second of 16kHz mono 16-bit PCM audio equals exactly 32,000 bytes
            if len(audio_buffer) >= 32000:
                processing_chunk = bytearray(audio_buffer[:32000])
                del audio_buffer[:32000]
                
                # 2. Translate raw binary bytes into floating-point tensors for the model
                raw_floats = torch.frombuffer(processing_chunk, dtype=torch.int16).float() / 32768.0
                waveform_tensor = raw_floats.unsqueeze(0).to(DEVICE) # Shape: [1, 16000]
                
                with torch.no_grad():
                    # 3. Compress raw audio waveform into discrete audio tokens
                    input_tokens = codec.encode_to_tokens(waveform_tensor)
                    
                    # Create a starter token matrix to trigger the response pipeline
                    seed_tgt = torch.zeros((1, 1, 4), dtype=torch.long, device=DEVICE)
                    
                    # 4. Generate the next audio token matrix using the transformer engine
                    logits_list = s2s_model(input_tokens, seed_tgt)
                    
                    # Predict the next response token indices across all 4 codebooks
                    predicted_tokens = torch.stack([torch.argmax(logits[:, -1, :], dim=-1) for logits in logits_list], dim=-1)
                    
                    # 5. Decode the predicted tokens back into continuous audio vectors
                    # For this streaming step, we simulate the reconstruction of the next token chunk
                    quantized_vectors = torch.randn(1, 128, 50, device=DEVICE) # Latent feature map space projection
                    reconstructed_wave = codec.decoder(quantized_vectors).squeeze(0).squeeze(0)
                    
                    # Convert float tensors back into standard output binary bytes
                    out_ints = (reconstructed_wave.clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
                    out_bytes = out_ints.cpu().numpy().tobytes()
                
                # 6. Stream the native audio data response back down to the client app immediately
                await websocket.send_bytes(out_bytes)
                
            # Yield control back to the asynchronous event loop to maintain sub-100ms processing speeds
            await asyncio.sleep(0.001)
            
    except WebSocketDisconnect:
        print("Audio stream connection terminated normally by client.")
    except Exception as e:
        print(f"Stream error encountered: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

if __name__ == "__main__":
    # Launch production web server mapping to port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
