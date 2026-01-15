import torch
import safetensors.torch
import numpy as np
import os
import logging
import soundfile as sf
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.default_parameters import DEFAULT_VARIANT

def generate_refs():
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    
    print("Loading model...")
    # Load model with default params but zero temperature for deterministic output
    model = TTSModel.load_model(DEFAULT_VARIANT, temp=0.0)
    model.eval()
    
    print("Mimi Encoder weight stats:")
    for name, param in model.mimi.encoder.named_parameters():
        if any(x in name for x in ["0.conv", "1.block.1.conv", "1.block.3.conv"]):
             print(f"{name}: shape={list(param.shape)}, mean={param.mean().item():.6f}, min={param.min().item():.6f}, max={param.max().item():.6f}, abs_max={param.abs().max().item():.6f}")
    ref_wav_path = "ref.wav"
    ref_24k_path = "ref_24k.wav"
    
    if not os.path.exists(ref_wav_path):
        # Create dummy ref.wav if not exists, for testing
        print("ref.wav not found, creating dummy sine wave")
        sr = 44100 # Let's simulate a mismatch
        t = torch.linspace(0, 1, sr)
        wav = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
        sf.write(ref_wav_path, wav.squeeze().numpy(), sr)
        
    print(f"Loading reference audio: {ref_wav_path}")
    wav, sr = audio_read(ref_wav_path)
    # Ensure correct sample rate/channels
    # audio_read returns [channels, samples]
    
    # Resample to 24k for strict parity
    wav_24k = convert_audio(wav, sr, model.config.mimi.sample_rate, 1).to(model.device)
    
    # Save ref_24k.wav for Rust to use
    print(f"Saving {ref_24k_path} (24kHz mono)")
    sf.write(ref_24k_path, wav_24k.cpu().squeeze().numpy(), model.config.mimi.sample_rate)
    
    # 1. Mimi Encode Parity
    print("Generating Mimi encoding reference...")
    # input expected: [B, C, T]
    mimi_input = wav_24k.unsqueeze(0)
    
    T = mimi_input.shape[-1]
    hop = model.mimi.frame_size
    if T % hop != 0:
        actual_pad = hop - (T % hop)
        mimi_input = torch.nn.functional.pad(mimi_input, (0, actual_pad))
    
    # Python's encode_to_latent does JUST pad_for_conv1d(mimi_input, frame_size, frame_size)
    # which only pads to a multiple of frame_size on the RIGHT.
    x_padded = mimi_input.clone()
    
    # Trace first layer
    layer0_out = model.mimi.encoder.model[0](x_padded, model_state=None)
    print(f"layer0_out shapes: {layer0_out.shape}, max: {layer0_out.max().item()}")
    
    # Trace first resnet block layers
    resnet1 = model.mimi.encoder.model[1]
    r1_l0_out = resnet1.block[0](layer0_out)
    r1_l1_out = resnet1.block[1](r1_l0_out, model_state=None)
    r1_l2_out = resnet1.block[2](r1_l1_out)
    r1_l3_out = resnet1.block[3](r1_l2_out, model_state=None)
    layer1_out = layer0_out + r1_l3_out # The final sum
    
    layer2_out = model.mimi.encoder.model[2](layer1_out) # ELU
    layer3_out = model.mimi.encoder.model[3](layer2_out, model_state=None) # Conv stride 4
    
    print(f"layer3_out shapes: {layer3_out.shape}, max: {layer3_out.max().item()}")
    
    seanet_out = model.mimi.encoder(x_padded, model_state=None)
    # encoder_transformer expects [B, C, T] and returns [B, C, T]
    (mimi_emb,) = model.mimi.encoder_transformer(seanet_out, model_state=None)
    mimi_latents = model.mimi._to_framerate(mimi_emb)
    
    safetensors.torch.save_file({
        "mimi_input": mimi_input.cpu(),
        "mimi_input_padded": x_padded.cpu(),
        "seanet_out": seanet_out.cpu(),
        "transformer_out": mimi_emb.cpu(),
        "mimi_latents": mimi_latents.cpu(),
        "layer0_out": layer0_out.cpu(),
        "layer1_out": layer1_out.cpu(),
        "r1_l1_out": r1_l1_out.cpu(),
        "r1_l3_out": r1_l3_out.cpu(),
        "layer2_out": layer2_out.cpu(),
        "layer3_out": layer3_out.cpu()
    }, "ref_mimi_latents.safetensors")
    print("Saved ref_mimi_latents.safetensors (with intermediates)")

    # Save input for debugging
    safetensors.torch.save_file({"mimi_input": mimi_input.cpu()}, "ref_mimi_input.safetensors")
    print("Saved ref_mimi_input.safetensors")
    
    # 2. Voice Conditioning Parity
    print("Generating Voice State reference...")
    conditioning = model._encode_audio(mimi_input)
    
    safetensors.torch.save_file({"voice_conditioning": conditioning.cpu()}, "ref_voice_conditioning.safetensors")
    print("Saved ref_voice_conditioning.safetensors")
    
    # 3. Audio Generation Output (Default Params)
    print("Generating audio (default params, fixed seed)...")
    text = "Hello world."
    
    # We re-load state to be safe/clean
    model_state = model.get_state_for_audio_prompt(mimi_input.squeeze(0)) # Expects [C, T]
    
    generated_audio = model.generate_audio(
        model_state=model_state,
        text_to_generate=text
    )
    
    output_path = "ref_output.wav"
    sf.write(output_path, generated_audio.cpu().squeeze().numpy(), model.sample_rate)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    generate_refs()
