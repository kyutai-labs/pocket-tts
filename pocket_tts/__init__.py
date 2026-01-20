from beartype import BeartypeConf
from beartype.claw import beartype_this_package

beartype_this_package(conf=BeartypeConf(is_color=False))

from pocket_tts.models.tts_model import TTSModel  # noqa: E402
from pocket_tts.data.audio import load_wav  # noqa: E402
from pocket_tts.data.audio_output import save_audio  # noqa: E402
from pocket_tts.rust_audio import (  # noqa: E402
    normalize_audio,
    apply_gain,
    resample_audio,
    apply_fade,
    compute_audio_metrics,
)

# Public methods:
# TTSModel.device
# TTSModel.sample_rate
# TTSModel.load_model
# TTSModel.generate_audio
# TTSModel.generate_audio_stream
# TTSModel.get_state_for_audio_prompt

# Public audio I/O:
# load_wav - Load audio file (industry-standard alias for audio_read)
# save_audio - Save audio with guaranteed save functionality

# Public Rust-accelerated functions:
# normalize_audio - High-performance audio normalization
# apply_gain - Apply gain to audio samples
# resample_audio - Resample audio (linear or sinc interpolation)
# apply_fade - Apply fade in/out to audio
# compute_audio_metrics - Compute RMS, peak, and dynamic range
from pocket_tts.rust_audio import (  # noqa: E402
    normalize_audio,
    apply_gain,
    resample_audio,
    apply_fade,
    compute_audio_metrics,
)

# NumPy replacement via rust-numpy (drop-in replacement with fallback to NumPy)
try:
    from pocket_tts.numpy_rs import (  # noqa: E402
        arange,
        array,
        clip,
        min,
        max,
        mean,
        median,
        sum,
        sqrt,
        log,
        std,
        var,
        reshape,
        transpose,
        concatenate,
        vstack,
        hstack,
        zeros,
        ones,
        eye,
        linspace,
        interp,
        dot,
        matmul,
        abs,
        power,
        frombuffer,
        size,
        percentile,
        int16,
        int32,
        float32,
        float64,
        int8,
        uint8,
        _RUST_NUMPY_AVAILABLE,
    )
except ImportError:
    _RUST_NUMPY_AVAILABLE = False

# Public methods:
# TTSModel.device
# TTSModel.sample_rate
# TTSModel.load_model
# TTSModel.generate_audio
# TTSModel.generate_audio_stream
# TTSModel.get_state_for_audio_prompt

# Public audio I/O:
# load_wav - Load audio file (industry-standard alias for audio_read)
# save_audio - Save audio with guaranteed save functionality

__all__ = [
    "TTSModel",
    "load_wav",
    "save_audio",
    "normalize_audio",
    "apply_gain",
    "resample_audio",
    "apply_fade",
    "compute_audio_metrics",
    # NumPy replacement functions
    "arange",
    "array",
    "clip",
    "min",
    "max",
    "mean",
    "median",
    "sum",
    "sqrt",
    "log",
    "std",
    "var",
    "reshape",
    "transpose",
    "concatenate",
    "vstack",
    "hstack",
    "zeros",
    "ones",
    "eye",
    "linspace",
    "interp",
    "dot",
    "matmul",
    "abs",
    "power",
    "frombuffer",
    "size",
    "percentile",
    "int16",
    "int32",
    "float32",
    "float64",
    "int8",
    "uint8",
    "_RUST_NUMPY_AVAILABLE",
]
