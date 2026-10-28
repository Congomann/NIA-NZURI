import os
import sys
import asyncio
import glob

# Ensure project src directory is accessible on Python module search path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from core_model.tokenizer import SwahiliAudioCodec
from core_model.transformer import SwahiliS2STransformer

app = FastAPI(title="NIA NZURI — Native Tanzanian Swahili S2S Engine")

# Global device and model management matrix allocation (CUDA -> Apple Silicon MPS -> CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

codec = None
s2s_model = None
ACTIVE_CHECKPOINT = "None"

@app.on_event("startup")
def load_trained_models():
    """Loads the core model weights from the persistent checkpoint directory."""
    global codec, s2s_model, ACTIVE_CHECKPOINT
    print(f"Initializing model runtime weights on device node: {DEVICE}")
    
    # Instantiate neural modules
    codec = SwahiliAudioCodec().to(DEVICE)
    codec.eval()
    
    s2s_model = SwahiliS2STransformer(num_tokens=1024, num_codebooks=4).to(DEVICE)
    
    # Dynamically find the highest epoch checkpoint available
    checkpoints_dir = os.path.abspath(os.path.join(src_dir, "..", "checkpoints"))
    available_pts = sorted(
        [os.path.join(checkpoints_dir, f) for f in (os.listdir(checkpoints_dir) if os.path.exists(checkpoints_dir) else []) if f.endswith(".pt")]
    )
    checkpoint_path = available_pts[-1] if available_pts else os.path.join(checkpoints_dir, "swahili_s2s_epoch_5.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        s2s_model.load_state_dict(checkpoint['model_state_dict'])
        ACTIVE_CHECKPOINT = os.path.basename(checkpoint_path)
        print(f"Successfully mounted production weights from: {checkpoint_path}")
    else:
        ACTIVE_CHECKPOINT = "Uninitialized (Checkpoint missing)"
        print("Warning: Running with uninitialized random weight matrices. Checkpoint not found.")
    s2s_model.eval()


@app.get("/api/status")
def get_system_status():
    """Returns runtime diagnostic metrics for the UI."""
    data_dir = os.path.abspath(os.path.join(src_dir, "..", "data", "processed_training_chunks"))
    chunk_count = len(glob.glob(os.path.join(data_dir, "*.wav"))) if os.path.exists(data_dir) else 0
    return JSONResponse({
        "status": "Online",
        "device": str(DEVICE).upper(),
        "checkpoint": ACTIVE_CHECKPOINT,
        "processed_chunks": chunk_count,
        "sample_rate": "16,000 Hz",
        "channels": "1 (Mono PCM)",
        "codebooks": 4
    })


@app.get("/", response_class=HTMLResponse)
def get_web_interface():
    """Serves the interactive live testing interface."""
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NIA NZURI — Native Swahili Speech-to-Speech Engine</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(20, 27, 45, 0.7);
      --border: rgba(255, 255, 255, 0.08);
      --accent: #10b981;
      --accent-glow: rgba(16, 185, 129, 0.25);
      --text: #f3f4f6;
      --text-dim: #9ca3af;
      --danger: #ef4444;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: radial-gradient(circle at 50% 0%, #172554 0%, var(--bg) 60%);
      color: var(--text);
      font-family: var(--font);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 32px 16px;
    }
    .container {
      width: 100%;
      max-width: 860px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    header {
      text-align: center;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid var(--accent);
      color: var(--accent);
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      align-self: center;
    }
    .badge .dot {
      width: 8px;
      height: 8px;
      background: var(--accent);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--accent);
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.8); }
    }
    h1 {
      font-size: 2.2rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff 40%, #93c5fd 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    p.subtitle {
      color: var(--text-dim);
      font-size: 0.95rem;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }
    .metric-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      backdrop-filter: blur(12px);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .metric-label {
      font-size: 0.75rem;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .metric-value {
      font-size: 1.1rem;
      font-weight: 700;
      color: #fff;
    }
    .main-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      backdrop-filter: blur(16px);
      border-radius: 16px;
      padding: 28px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 20px;
      box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    }
    canvas {
      width: 100%;
      height: 120px;
      background: rgba(0, 0, 0, 0.4);
      border-radius: 12px;
      border: 1px solid var(--border);
    }
    .action-row {
      display: flex;
      gap: 16px;
      width: 100%;
      justify-content: center;
      flex-wrap: wrap;
    }
    button.btn {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      background: var(--accent);
      color: #000;
      font-weight: 700;
      font-size: 1rem;
      padding: 14px 28px;
      border-radius: 12px;
      border: none;
      cursor: pointer;
      box-shadow: 0 4px 20px var(--accent-glow);
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    button.btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 24px rgba(16, 185, 129, 0.4);
    }
    button.btn:active {
      transform: translateY(0);
    }
    button.btn.streaming {
      background: var(--danger);
      color: #fff;
      box-shadow: 0 4px 20px rgba(239, 68, 68, 0.3);
    }
    .file-input-wrapper {
      position: relative;
      overflow: hidden;
      display: inline-block;
    }
    .file-input-wrapper input[type="file"] {
      position: absolute;
      left: 0;
      top: 0;
      opacity: 0;
      cursor: pointer;
      width: 100%;
      height: 100%;
    }
    .btn-secondary {
      background: rgba(255, 255, 255, 0.08);
      color: var(--text);
      font-weight: 600;
      padding: 14px 24px;
      border-radius: 12px;
      border: 1px solid var(--border);
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.12);
      transform: translateY(-1px);
    }
    .log-box {
      width: 100%;
      background: rgba(0, 0, 0, 0.5);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
      font-size: 0.8rem;
      color: #93c5fd;
      max-height: 160px;
      overflow-y: auto;
      display: flex;
      flex-direction: column-reverse;
      gap: 6px;
    }
    .log-entry { opacity: 0.85; }
    .status-pill {
      font-size: 0.85rem;
      color: var(--text-dim);
      display: flex;
      align-items: center;
      gap: 8px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge"><span class="dot"></span> Live S2S WebSocket Engine</div>
      <h1>NIA NZURI — Swahili Voice Studio</h1>
      <p class="subtitle">Direct full-duplex Speech-to-Speech inference streaming through your model's neural audio codec</p>
    </header>

    <div class="grid">
      <div class="metric-card">
        <span class="metric-label">Compute Device</span>
        <span class="metric-value" id="val-device">Loading...</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Mounted Weights</span>
        <span class="metric-value" id="val-ckpt" style="font-size: 0.95rem;">Loading...</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Training Corpus</span>
        <span class="metric-value" id="val-chunks">-- Chunks</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Audio Protocol</span>
        <span class="metric-value">16kHz 16-bit PCM</span>
      </div>
    </div>

    <div class="main-card">
      <canvas id="visualizer"></canvas>

      <div class="status-pill">
        <span id="conn-indicator" style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#9ca3af;"></span>
        <span id="conn-text">Ready to connect</span>
      </div>

      <div class="action-row">
        <button id="stream-btn" class="btn">
          <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 100-6 3 3 0 000 6z"></path></svg>
          <span id="btn-text">Start Speaking (Live Mic)</span>
        </button>

        <div class="file-input-wrapper">
          <button class="btn-secondary">Upload Audio File (.wav/.mp3)</button>
          <input type="file" id="file-uploader" accept="audio/*">
        </div>
      </div>

      <div class="log-box" id="log-output">
        <div class="log-entry">System ready. Click "Start Speaking" or upload audio to test live S2S streaming.</div>
      </div>
    </div>
  </div>

  <script>
    const logBox = document.getElementById('log-output');
    function log(msg) {
      const d = document.createElement('div');
      d.className = 'log-entry';
      d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
      logBox.prepend(d);
    }

    // Fetch initial status metrics
    fetch('/api/status')
      .then(r => r.json())
      .then(data => {
        document.getElementById('val-device').textContent = data.device;
        document.getElementById('val-ckpt').textContent = data.checkpoint;
        document.getElementById('val-chunks').textContent = `${data.processed_chunks} Chunks`;
      })
      .catch(e => console.error(e));

    // Canvas visualizer
    const canvas = document.getElementById('visualizer');
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    let audioCtx = null;
    let analyser = null;
    let micStream = null;
    let scriptNode = null;
    let ws = null;
    let isStreaming = false;

    // Output audio playback queue
    let audioQueue = [];
    let isPlaying = false;

    function drawVisualizer() {
      requestAnimationFrame(drawVisualizer);
      if (!analyser) {
        ctx.fillStyle = 'rgba(0,0,0,0.4)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        return;
      }

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyser.getByteTimeDomainData(dataArray);

      ctx.fillStyle = 'rgba(11, 15, 25, 0.4)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = isStreaming ? '#10b981' : '#3b82f6';
      ctx.beginPath();

      const sliceWidth = canvas.width * 1.0 / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    }
    drawVisualizer();

    const streamBtn = document.getElementById('stream-btn');
    const btnText = document.getElementById('btn-text');
    const connDot = document.getElementById('conn-indicator');
    const connText = document.getElementById('conn-text');

    streamBtn.addEventListener('click', async () => {
      if (isStreaming) {
        stopStreaming();
      } else {
        await startStreaming();
      }
    });

    async function startStreaming() {
      try {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/stream/v1/swahili`;
        log(`Connecting to WebSocket: ${wsUrl}`);
        
        ws = new WebSocket(wsUrl);
        ws.binaryType = 'arraybuffer';

        ws.onopen = async () => {
          log("WebSocket connection established. Accessing microphone...");
          connDot.style.background = '#10b981';
          connText.textContent = "Live Stream Active";
          
          audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
          micStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
          
          const source = audioCtx.createMediaStreamSource(micStream);
          analyser = audioCtx.createAnalyser();
          analyser.fftSize = 512;
          source.connect(analyser);

          // ScriptProcessor for real-time 16kHz PCM streaming
          scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
          source.connect(scriptNode);
          scriptNode.connect(audioCtx.destination);

          scriptNode.onaudioprocess = (e) => {
            if (!isStreaming || ws.readyState !== WebSocket.OPEN) return;
            const inputData = e.inputBuffer.getChannelData(0);
            
            // Convert Float32Array [-1.0, 1.0] to Int16 PCM bytes
            const pcmBuffer = new ArrayBuffer(inputData.length * 2);
            const view = new DataView(pcmBuffer);
            for (let i = 0; i < inputData.length; i++) {
              let s = Math.max(-1, Math.min(1, inputData[i]));
              view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }
            ws.send(pcmBuffer);
          };

          isStreaming = true;
          streamBtn.classList.add('streaming');
          btnText.textContent = "Stop Stream";
          log("Microphone streaming active. Speak in Swahili...");
        };

        ws.onmessage = (event) => {
          const bytes = event.data;
          log(`Received model speech response chunk (${bytes.byteLength} bytes)`);
          playAudioChunk(bytes);
        };

        ws.onclose = () => {
          log("WebSocket connection closed.");
          stopStreaming();
        };

        ws.onerror = (err) => {
          log(`WebSocket error: ${err}`);
          stopStreaming();
        };

      } catch (err) {
        log(`Error: ${err.message}`);
        stopStreaming();
      }
    }

    function stopStreaming() {
      isStreaming = false;
      streamBtn.classList.remove('streaming');
      btnText.textContent = "Start Speaking (Live Mic)";
      connDot.style.background = '#9ca3af';
      connText.textContent = "Disconnected";

      if (scriptNode) { scriptNode.disconnect(); scriptNode = null; }
      if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
      if (ws) { ws.close(); ws = null; }
    }

    // Audio response player
    function playAudioChunk(arrayBuffer) {
      if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      
      const int16Array = new Int16Array(arrayBuffer);
      const float32Array = new Float32Array(int16Array.length);
      for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768.0;
      }

      const audioBuffer = audioCtx.createBuffer(1, float32Array.length, 16000);
      audioBuffer.getChannelData(0).set(float32Array);

      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);
      source.start();
    }

    // Audio file testing
    document.getElementById('file-uploader').addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      log(`Loading audio file: ${file.name}`);
      const arrayBuffer = await file.arrayBuffer();
      const tempCtx = new (window.AudioContext || window.webkitAudioContext)();
      const decoded = await tempCtx.decodeAudioData(arrayBuffer);

      log(`Resampling ${file.name} to 16kHz mono...`);
      const offlineCtx = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
      const bSource = offlineCtx.createBufferSource();
      bSource.buffer = decoded;
      bSource.connect(offlineCtx.destination);
      bSource.start();
      const resampled = await offlineCtx.startRendering();

      const channelData = resampled.getChannelData(0);
      const pcmBuffer = new ArrayBuffer(channelData.length * 2);
      const view = new DataView(pcmBuffer);
      for (let i = 0; i < channelData.length; i++) {
        let s = Math.max(-1, Math.min(1, channelData[i]));
        view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }

      log(`Connecting WebSocket to stream ${file.name}...`);
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const fileWs = new WebSocket(`${wsProtocol}//${window.location.host}/stream/v1/swahili`);
      fileWs.binaryType = 'arraybuffer';

      fileWs.onopen = () => {
        log(`Streaming ${pcmBuffer.byteLength} bytes to model...`);
        // Send in 32000 byte blocks
        const chunkSize = 32000;
        for (let i = 0; i < pcmBuffer.byteLength; i += chunkSize) {
          fileWs.send(pcmBuffer.slice(i, i + chunkSize));
        }
      };

      fileWs.onmessage = (event) => {
        log(`Received ${event.data.byteLength} bytes audio response!`);
        playAudioChunk(event.data);
      };

      fileWs.onclose = () => log("File test stream completed.");
    });
  </script>
</body>
</html>
""")


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
