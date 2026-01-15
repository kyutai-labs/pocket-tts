"""
Extract multiple FlowLM generated latents for parity testing.

This captures the generated latents and audio at each step to compare with Rust.
"""

import torch
import safetensors.torch
import os
import logging
import queue
import threading
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.default_parameters import DEFAULT_VARIANT
from pocket_tts.modules.stateful_module import init_states, increment_steps

def extract_latent_refs():
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    
    print("Loading model...")
    model = TTSModel.load_model(DEFAULT_VARIANT, temp=0.0)
    model.eval()
    
    # Load reference audio and get voice state
    ref_wav_path = "ref_24k.wav"
    if not os.path.exists(ref_wav_path):
        print(f"Error: {ref_wav_path} not found.")
        return
        
    wav, sr = audio_read(ref_wav_path)
    wav_24k = convert_audio(wav, sr, model.config.mimi.sample_rate, 1).to(model.device)
    mimi_input = wav_24k.unsqueeze(0)
    
    # Get voice state same way as generate
    model_state = model.get_state_for_audio_prompt(mimi_input.squeeze(0))
    
    # Generate using actual stream to capture latents
    text = "Hello world."
    
    # Capture latents during generation using the internal stream
    latents_captured = []
    audio_chunks = []
    
    for chunk in model.generate_audio_stream(
        model_state=model_state,
        text_to_generate=text,
        frames_after_eos=None,
        copy_state=True,
    ):
        audio_chunks.append(chunk.cpu())
    
    # Concatenate audio
    full_audio = torch.cat(audio_chunks, dim=-1)
    print(f"Generated {len(audio_chunks)} audio chunks")
    print(f"Full audio shape: {full_audio.shape}")
    
    # Also capture first N frames explicitly step-by-step for parity
    # Re-initialize state
    model_state2 = model.get_state_for_audio_prompt(mimi_input.squeeze(0))
    mimi_state = init_states(model.mimi, batch_size=1, sequence_length=1000)
    
    from pocket_tts.models.tts_model import prepare_text_prompt
    prepared_text, frames_after_eos = prepare_text_prompt(text)
    
    from pocket_tts.conditioners.base import TokenizedText
    prep = model.flow_lm.conditioner.prepare(prepared_text)
    text_embeddings = model.flow_lm.conditioner(TokenizedText(prep.tokens))
    audio_conditioning = model._audio_conditioning_for_prompt(mimi_input)
    text_embeddings = torch.cat([text_embeddings, audio_conditioning], dim=1)
    
    # Prompt transformer
    model.flow_lm.transformer(text_embeddings, model_state2)
    increment_steps(model.flow_lm, model_state2, increment=text_embeddings.shape[1])
    
    # Generate N latents
    num_frames = min(16, len(audio_chunks))
    latents = []
    audio_frames = []
    
    backbone_input = torch.full(
        (1, 1, model.flow_lm.ldim),
        fill_value=float("NaN"),
        device=model.flow_lm.device,
        dtype=model.flow_lm.dtype,
    )
    
    for i in range(num_frames):
        # Run FlowLM
        next_latent, is_eos = model.flow_lm._sample_next_latent(
            backbone_input,
            text_embeddings,
            model_state=model_state2,
            lsd_decode_steps=model.lsd_decode_steps,
            temp=model.temp,
            noise_clamp=model.noise_clamp,
            eos_threshold=model.eos_threshold,
        )
        increment_steps(model.flow_lm, model_state2, increment=1)
        
        latents.append(next_latent.cpu().contiguous())
        
        # Decode
        mimi_decoding_input = next_latent * model.flow_lm.emb_std + model.flow_lm.emb_mean
        transposed = mimi_decoding_input.transpose(-1, -2)
        quantized = model.mimi.quantizer(transposed)
        audio_frame = model.mimi.decode_from_latent(quantized, mimi_state)
        increment_steps(model.mimi, mimi_state, increment=16)
        
        audio_frames.append(audio_frame.cpu().contiguous())
        
        backbone_input = next_latent.unsqueeze(1)
        print(f"Frame {i}: latent max={next_latent.abs().max().item():.4f}, is_eos={is_eos.item()}")
    
    # Stack latents
    all_latents = torch.stack(latents, dim=1).squeeze(2)  # [1, N, ldim]
    all_audio = torch.cat(audio_frames, dim=-1)  # [1, 1, N*1920]
    
    print(f"\nAll latents shape: {all_latents.shape}")
    print(f"All audio shape: {all_audio.shape}")
    
    safetensors.torch.save_file({
        "generated_latents": all_latents.contiguous(),
        "generated_audio": all_audio.contiguous(),
        "text_embeddings": text_embeddings.cpu().contiguous(),
    }, "ref_generated_latents.safetensors")
    
    print("\nSaved ref_generated_latents.safetensors")

if __name__ == "__main__":
    extract_latent_refs()

