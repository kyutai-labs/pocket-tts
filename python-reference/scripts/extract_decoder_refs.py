"""
Extract decoder intermediate outputs for parity testing.

This script captures outputs at each stage of the mimi decoder:
1. Input to decoder (quantized latent from FlowLM)
2. After upsample
3. After decoder_transformer
4. Final audio output from SEANet decoder
"""

import torch
import safetensors.torch
import os
import logging
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.default_parameters import DEFAULT_VARIANT
from pocket_tts.modules.stateful_module import init_states, increment_steps

def extract_decoder_refs():
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    
    print("Loading model...")
    model = TTSModel.load_model(DEFAULT_VARIANT, temp=0.0)
    model.eval()
    
    # Load reference audio and get voice state
    ref_wav_path = "ref_24k.wav"
    if not os.path.exists(ref_wav_path):
        print(f"Error: {ref_wav_path} not found. Run extract_refs.py first.")
        return
        
    wav, sr = audio_read(ref_wav_path)
    wav_24k = convert_audio(wav, sr, model.config.mimi.sample_rate, 1).to(model.device)
    mimi_input = wav_24k.unsqueeze(0)
    
    # Get voice state
    model_state = model.get_state_for_audio_prompt(mimi_input.squeeze(0))
    
    # Generate one frame using FlowLM
    text = "Hello world."
    from pocket_tts.conditioners.base import TokenizedText
    
    prepared = model.flow_lm.conditioner.prepare(text)
    text_embeddings = model.flow_lm.conditioner(TokenizedText(prepared.tokens))
    
    # Run text through transformer
    model.flow_lm.transformer(text_embeddings, model_state)
    increment_steps(model.flow_lm, model_state, increment=text_embeddings.shape[1])
    
    # Generate first latent
    backbone_input = torch.full(
        (1, 1, model.flow_lm.ldim),
        fill_value=float("NaN"),
        device=model.flow_lm.device,
        dtype=model.flow_lm.dtype,
    )
    
    next_latent, is_eos = model._run_flow_lm_and_increment_step(
        model_state=model_state, backbone_input_latents=backbone_input
    )
    
    print(f"Generated latent shape: {next_latent.shape}")
    print(f"is_eos: {is_eos}")
    
    # Now decode this latent through mimi decoder step by step
    mimi_state = init_states(model.mimi, batch_size=1, sequence_length=1000)
    
    # Denormalize
    mimi_decoding_input = next_latent * model.flow_lm.emb_std + model.flow_lm.emb_mean
    transposed = mimi_decoding_input.transpose(-1, -2)
    
    # Quantize
    quantized = model.mimi.quantizer(transposed)
    print(f"Quantized shape: {quantized.shape}")
    
    # Step 1: Upsample (if present)
    emb = quantized
    if hasattr(model.mimi, 'upsample') and model.mimi.upsample is not None:
        after_upsample = model.mimi.upsample(emb, mimi_state)
        print(f"After upsample shape: {after_upsample.shape}")
    else:
        after_upsample = emb
        print("No upsample (frame rates match)")
    
    # Step 2: Decoder transformer
    (after_decoder_tr,) = model.mimi.decoder_transformer(after_upsample, mimi_state)
    print(f"After decoder_transformer shape: {after_decoder_tr.shape}")
    
    # Step 3: SEANet decoder
    final_audio = model.mimi.decoder(after_decoder_tr, mimi_state)
    print(f"Final audio shape: {final_audio.shape}")
    
    # Increment state for proper streaming
    increment_steps(model.mimi, mimi_state, increment=16)
    
    # Save all intermediate outputs
    safetensors.torch.save_file({
        "latent_from_flowlm": next_latent.contiguous().cpu(),
        "denormalized": mimi_decoding_input.contiguous().cpu(),
        "quantized": quantized.contiguous().cpu(),
        "after_upsample": after_upsample.contiguous().cpu(),
        "after_decoder_transformer": after_decoder_tr.contiguous().cpu(),
        "final_audio": final_audio.contiguous().cpu(),
    }, "ref_decoder_intermediates.safetensors")
    
    print("\nSaved ref_decoder_intermediates.safetensors")
    print("Contains: latent_from_flowlm, denormalized, quantized, after_upsample, after_decoder_transformer, final_audio")

if __name__ == "__main__":
    extract_decoder_refs()
