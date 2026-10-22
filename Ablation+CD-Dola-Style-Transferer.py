# =============================================================================
# GEMMA 3 27B — ONE-MODEL COMBINED BIAS + ATTENTION ABLATION
# =============================================================================
#
# PHASE 1:
#   Existing CSV contains the 1,808 sentences.
#
# PHASE 2:
#   ABLATION ON
#   Combined:
#       High Bias
#       Low Bias
#       High Attention
#       Low Attention
#
# PHASE 3:
#   ABLATION OFF
#   Same exact sentences / combinations.
#
# IMPORTANT:
#   - ONE Gemma 3 27B model instance.
#   - 62 decoder-layer wrappers.
#   - Model weights are never modified.
#   - Threshold is ONLY used to select the bias neuron files.
#   - ALL CSV sentences are processed for EVERY combination.
#   - Bias file selection requires EXACT experiment type + threshold.
#   - Attention files are selected independently of bias type/threshold.
#   - Bias neuron files are NOT assumed to be nested.
#   - Drive checkpoint is written after EVERY batch.
#   - One dot is printed for EVERY sentence.
#   - A new progress line is printed after EVERY 100 sentences.
#
# =============================================================================


# =============================================================================
# 0. INSTALL / IMPORTS
# =============================================================================

import os
import gc
import re
import json
import time
import warnings
from pathlib import Path
from collections import defaultdict
import shutil
import math

import torch
import torch.nn.functional as F
import pandas as pd

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
)

warnings.filterwarnings("ignore")

# ============================================================
# CUDA Configuration
# ============================================================

if torch.cuda.is_available():

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

MODEL_NAME = "google/gemma-3-27b-it"

CSV_PATH = (
    "/content/BABEDataset+Models-All_Colab_Batches.csv"
)

# -----------------------------------------------------------------------------
# BIAS ROOT
#
# Bias neuron files are now located directly under this directory.
#
# Example:
#
# /content/content - Bias Type Neurons/
#     Xb_specific_neurons.Gemma3-27B-Framing-10.pt
#     Xa_specific_neurons.Gemma3-27B-Framing-10.pt
#     Xb_specific_neurons.Gemma3-27B-Emotional-10.pt
#     Xa_specific_neurons.Gemma3-27B-Emotional-10.pt
#     ...
#
# -----------------------------------------------------------------------------

COMBINED_ABLATION_ROOT = (
    "/content/combined_ablation_sets"
)

# -----------------------------------------------------------------------------
# Local output
# -----------------------------------------------------------------------------

OUTPUT_ROOT = (
    "/content/ablation_inference_results"
)

# -----------------------------------------------------------------------------
# Google Drive output
# -----------------------------------------------------------------------------

DRIVE_OUTPUT_ROOT = (
    "/content/drive/MyDrive/"
    "ablation_inference_result"
)


# -----------------------------------------------------------------------------
# Experiment dimensions
# -----------------------------------------------------------------------------

BIAS_TYPES = [
    # "Framing",
    # "Emotional",
    "Divisive",
]

THRESHOLDS = [
    6,
]

TOP_K_VALUES = [
    14000
]


# -----------------------------------------------------------------------------
# Inference configuration
# -----------------------------------------------------------------------------

BATCH_SIZE = 32

MAX_NEW_TOKENS = 300

DO_SAMPLE = False

# TEMPERATURE = 1.0

# TOP_P = 1.0

# -----------------------------------------------------------------------------
# CD-DOLA
# -----------------------------------------------------------------------------

CD_DOLA = True

# Candidate style layers used by CD-DOLA.
# Replace these with the layers you selected from your neuron-density analysis.
CD_DOLA_CANDIDATE_LAYERS = [
    29,
    39,
    27,
    28
]

# Mature/final layer used by CD-DOLA.
CD_DOLA_MATURE_LAYER = 61

# -----------------------------------------------------------------------------
# Resume behavior
# -----------------------------------------------------------------------------

RESUME = True

# Progress message frequency.
PROGRESS_EVERY = 100


# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------

PROMPT2_TEMPLATE = """
Framing bias: Influencing opinions based on how information is presented rather than on objective facts.
Emotional bias: Using negative emotions to provoke a reader's opinions in a way that counters objective reality.
Divisive bias: Using divisive, tribalistic framing that discourages empathy

Sentence: {sentence}

Use the original scores and reasonings to rewrite the sentence so that the scores are as close to 1 as possible without removing important contextual or factual information.
"""

PROMPT3_TEMPLATE = """
Framing bias:
Influencing opinions based on how information is presented rather than on objective facts.

Emotional bias:
Using negative emotions to provoke a reader's opinions in a way that counters objective reality.

Divisive bias:
Using divisive, tribalistic framing that discourages empathy.

Sentence:

{sentence}

Evaluate the sentence.

Score each category from 1-10.

Return EXACTLY in this format:

Framing: [Score]
Emotional: [Score]
Divisive: [Score]

Reasons:
- Framing:
- Emotional:
- Divisive:

Do not rewrite the sentence.
Do not provide any additional scores.
"""


# =============================================================================
# 2. TIMESTAMP HELPER
# =============================================================================

def timestamp():
    """
    Return current local timestamp.
    """

    return time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def log(message):
    """
    Print a timestamped message.
    """

    print(
        f"[{timestamp()}] {message}"
    )


# =============================================================================
# 3. GPU INFORMATION
# =============================================================================

print("=" * 80)
print(
    "GEMMA 3 27B — ONE-MODEL COMBINED "
    "BIAS + ATTENTION ABLATION"
)
print("=" * 80)

log("Experiment script started.")

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is not available. "
        "This experiment requires a CUDA GPU."
    )

GPU_COUNT = torch.cuda.device_count()

print(
    f"GPU count: {GPU_COUNT}"
)

for i in range(GPU_COUNT):

    props = (
        torch.cuda.get_device_properties(i)
    )

    total_gb = (
        props.total_memory
        / (1024 ** 3)
    )

    print(
        f"GPU {i}: {props.name} | "
        f"{total_gb:.2f} GB"
    )

print(
    f"BF16 supported: "
    f"{torch.cuda.is_bf16_supported()}"
)

print()

DEVICE = torch.device(
    "cuda:0"
)

DTYPE = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

print(
    f"Using device: {DEVICE}"
)

print(
    f"Using dtype: {DTYPE}"
)


# =============================================================================
# 4. GOOGLE DRIVE
# =============================================================================

try:

    from google.colab import drive

    print()
    print("=" * 80)
    print("MOUNTING GOOGLE DRIVE")
    print("=" * 80)

    drive.mount(
        "/content/drive",
        force_remount=True,
    )

    if os.path.exists(
        "/content/drive"
    ):

        print(
            "Google Drive available."
        )

        os.makedirs(
            DRIVE_OUTPUT_ROOT,
            exist_ok=True,
        )

        print(
            "Drive checkpoint directory:"
        )

        print(
            DRIVE_OUTPUT_ROOT
        )

        print(
            "Drive checkpointing: ENABLED"
        )

        print(
            "Checkpoint frequency: EVERY BATCH"
        )

except Exception as e:

    print(
        f"Google Drive mount skipped: {e}"
    )


# =============================================================================
# 5. PRINT CONFIGURATION
# =============================================================================

print()
print("=" * 80)
print("CONFIGURATION")
print("=" * 80)

print(
    f"Types:       {BIAS_TYPES}"
)

print(
    f"Thresholds:  {THRESHOLDS}"
)

print(
    f"Top-k:       {TOP_K_VALUES}"
)

print(
    f"CSV:         {CSV_PATH}"
)

print()
print("Combined ablation root:")
print(COMBINED_ABLATION_ROOT)

print()
print("Local output:")
print(OUTPUT_ROOT)

print()
print("Drive output:")
print(DRIVE_OUTPUT_ROOT)

print()
print("MODEL COUNT: 1")
print("Phase 2 = ABLATION ON")
print("Phase 3 = ABLATION OFF")
print(
    "Threshold affects neuron-file selection ONLY."
)
print(
    "ALL CSV sentences are processed "
    "for every combination."
)
print(
    "One dot printed per sentence."
)
print(
    f"Progress printed every "
    f"{PROGRESS_EVERY} sentences."
)
print(
    "Drive checkpoint saved after every batch."
)


# =============================================================================
# 6. CREATE OUTPUT DIRECTORIES
# =============================================================================

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True,
)

os.makedirs(
    DRIVE_OUTPUT_ROOT,
    exist_ok=True,
)


# =============================================================================
# 7. LOAD PHASE 1 RESULTS / SENTENCES
# =============================================================================

print()
print("=" * 80)
print("LOADING PHASE 1 RESULTS")
print("=" * 80)

if not os.path.exists(
    CSV_PATH
):

    raise FileNotFoundError(
        f"CSV file not found:\n{CSV_PATH}"
    )

df = pd.read_csv(
    CSV_PATH
)

print(
    f"Loaded {len(df):,} rows."
)

if "sentence" not in df.columns:

    raise KeyError(
        "CSV does not contain required "
        "'sentence' column.\n"
        f"Columns found: {list(df.columns)}"
    )

print(
    "Sentence column: sentence"
)

sentences = (
    df["sentence"].fillna("").astype(str)
    + "\n\n"
    + df["output_1"].fillna("").astype(str)
).tolist()

# =========================================================================
# LIMIT EXPERIMENT TO FIRST 1,000 SENTENCES
# =========================================================================

MAX_SENTENCES = 1809

sentences = sentences[:MAX_SENTENCES]

print(
    f"\nSentences used for EVERY combination: "
    f"{len(sentences):,}"
)



# =============================================================================
# 8. TORCH LOAD HELPER
# =============================================================================

def safe_torch_load(path):
    """
    Load .pt files on CPU.

    This prevents neuron files from consuming GPU memory during
    discovery / validation.
    """

    try:

        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        return torch.load(
            path,
            map_location="cpu",
        )


# =============================================================================
# 9. NEURON FILE METADATA
# =============================================================================

def inspect_neuron_file(path):
    """
    Inspect a .pt neuron file.

    Returns metadata without moving neuron tensors to GPU.
    """

    path = Path(path)

    data = safe_torch_load(
        path
    )

    if not isinstance(
        data,
        dict,
    ):

        return {
            "path": path,
            "data": data,
            "is_dict": False,
            "has_neurons": False,
            "filter_type": None,
            "threshold": None,
        }

    return {
        "path": path,
        "data": data,
        "is_dict": True,
        "has_neurons": (
            "neurons" in data
        ),
        "filter_type": data.get(
            "filter_type"
        ),
        "threshold": data.get(
            "threshold"
        ),
    }


# =============================================================================
# 10. EXTRACT K VALUES
# =============================================================================

def get_available_k_values(data):
    """
    Determine which top-k values exist in a neuron file.

    Supports:

        neurons = {
            5000: ...,
            6000: ...,
        }

    and string keys such as:

        "5000"
    """

    if not isinstance(
        data,
        dict,
    ):

        return []

    neurons = data.get(
        "neurons"
    )

    if isinstance(
        neurons,
        dict,
    ):

        values = []

        for key in neurons.keys():

            try:

                values.append(
                    int(key)
                )

            except Exception:

                pass

        return sorted(
            set(values)
        )

    return []


# =============================================================================
# 11. EXACT BIAS FILE DISCOVERY
# =============================================================================
#
# Bias files are NOT assumed to be nested.
#
# We search directly inside BIAS_ROOT.
#
# Exact filename:
#
#   Xb_specific_neurons.Gemma3-27B-Framing-10.pt
#   Xa_specific_neurons.Gemma3-27B-Framing-10.pt
#
# Metadata is still validated:
#
#   filter_type == experiment_type
#   threshold   == threshold
#
# =============================================================================

# =============================================================================
# 12. ATTENTION FILE DISCOVERY
# =============================================================================

# =============================================================================
# 13. LOAD COMBINED ABLATION NEURONS
# =============================================================================

def load_combined_ablation_file(path):
    """
    Load the precomputed combined-ablation neuron file.

    The file already contains the FINAL neurons to ablate.

    No bias/attention set operations are performed.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Combined ablation file does not exist:\n"
            f"{path}"
        )

    print()
    print("Loading combined ablation neurons:")
    print(path)

    data = safe_torch_load(path)

    if not isinstance(data, dict):
        raise TypeError(
            f"Combined ablation file is not a dictionary.\n"
            f"Type = {type(data)}\n"
            f"File = {path}"
        )

    if "neurons" not in data:
        raise KeyError(
            f"Combined ablation file does not contain "
            f"'neurons'.\n"
            f"Keys found: {list(data.keys())}\n"
            f"File = {path}"
        )

    neurons = normalize_neuron_indices(
        data["neurons"]
    )

    print()
    print("Combined ablation neurons loaded.")

    print(
        f"Layers containing neurons: "
        f"{len(neurons)}"
    )

    print(
        f"Total neurons to ablate: "
        f"{sum(len(v) for v in neurons.values()):,}"
    )

    return neurons

# =============================================================================
# 14. FIND COMBINED ABLATION FILE
# =============================================================================

def find_combined_ablation_file(
    combined_root,
    experiment_type,
    threshold,
    top_k,
):
    """
    Find the precomputed combined-ablation file.

    Example:

        Gemma3-27B_Framing_threshold-6_topk-5000.pt

    The neurons inside this file are used DIRECTLY.
    """

    combined_root = Path(combined_root)

    if not combined_root.exists():
        raise FileNotFoundError(
            f"Combined ablation root does not exist:\n"
            f"{combined_root}"
        )

    filename = (
        f"Gemma3-27B_"
        f"{experiment_type}_"
        f"threshold-{threshold}_"
        f"topk-{top_k}.pt"
    )

    path = combined_root / filename

    if not path.exists():
        raise FileNotFoundError(
            "\n"
            "COMBINED ABLATION FILE NOT FOUND\n"
            f"Expected file:\n"
            f"  {path}\n"
        )

    print()
    print("Using combined ablation file:")
    print(path)

    return path


# =============================================================================
# 14. EXTRACT SPECIFIC TOP-K NEURONS
# =============================================================================

def get_neurons_for_k(
    neuron_data,
    k,
):
    """
    Extract neuron set for a requested k.

    Handles integer and string dictionary keys.
    """

    if not isinstance(
        neuron_data,
        dict,
    ):

        raise TypeError(
            "Neuron data must be a dictionary."
        )

    neurons = neuron_data.get(
        "neurons"
    )

    if neurons is None:

        raise KeyError(
            "Neuron data does not contain "
            "'neurons'."
        )

    # -------------------------------------------------------------------------
    # Standard format:
    #
    # neurons = {
    #     5000: [...],
    #     6000: [...],
    # }
    # -------------------------------------------------------------------------

    if isinstance(
        neurons,
        dict,
    ):

        if k in neurons:

            return neurons[k]

        if str(k) in neurons:

            return neurons[str(k)]

        for key, value in neurons.items():

            try:

                if int(key) == int(k):

                    return value

            except Exception:

                pass

        available = (
            get_available_k_values(
                neuron_data
            )
        )

        raise KeyError(
            f"Requested k={k} not found.\n"
            f"Available k={available}"
        )

    # -------------------------------------------------------------------------
    # Single neuron-set format.
    # -------------------------------------------------------------------------

    if isinstance(
        neurons,
        (list, tuple),
    ):

        return neurons

    if torch.is_tensor(
        neurons
    ):

        return neurons

    raise TypeError(
        f"Unsupported neurons structure: "
        f"{type(neurons)}"
    )


# =============================================================================
# 15. NORMALIZE NEURON INDICES
# =============================================================================

def normalize_neuron_indices(
    neurons,
):
    """
    Convert neuron specification into:

        {
            layer_index: set(neuron_indices)
        }

    Supported forms:

        {
            0: [1, 2, 3],
            1: [4, 5]
        }

    or:

        [
            (0, 1),
            (0, 2),
            (1, 4)
        ]

    or tensors.
    """

    result = defaultdict(
        set
    )

    if neurons is None:

        return dict(
            result
        )

    # -------------------------------------------------------------------------
    # Dictionary format.
    # -------------------------------------------------------------------------

    if isinstance(
        neurons,
        dict,
    ):

        for layer, indices in neurons.items():

            try:

                layer_int = int(
                    layer
                )

            except Exception:

                continue

            if torch.is_tensor(
                indices
            ):

                indices = (
                    indices
                    .detach()
                    .cpu()
                    .tolist()
                )

            if isinstance(
                indices,
                (list, tuple, set),
            ):

                for idx in indices:

                    try:

                        result[
                            layer_int
                        ].add(
                            int(idx)
                        )

                    except Exception:

                        pass

            else:

                try:

                    result[
                        layer_int
                    ].add(
                        int(indices)
                    )

                except Exception:

                    pass

        return dict(
            result
        )

    # -------------------------------------------------------------------------
    # List / tuple of (layer, neuron).
    # -------------------------------------------------------------------------

    if isinstance(
        neurons,
        (list, tuple),
    ):

        for item in neurons:

            if (
                isinstance(
                    item,
                    (list, tuple),
                )
                and len(item) >= 2
            ):

                try:

                    layer = int(
                        item[0]
                    )

                    neuron = int(
                        item[1]
                    )

                    result[
                        layer
                    ].add(
                        neuron
                    )

                except Exception:

                    pass

        return dict(
            result
        )

    # -------------------------------------------------------------------------
    # Tensor format.
    # -------------------------------------------------------------------------

    if torch.is_tensor(
        neurons
    ):

        arr = (
            neurons
            .detach()
            .cpu()
        )

        if (
            arr.ndim == 2
            and arr.shape[1] >= 2
        ):

            for row in arr:

                try:

                    layer = int(
                        row[0]
                    )

                    neuron = int(
                        row[1]
                    )

                    result[
                        layer
                    ].add(
                        neuron
                    )

                except Exception:

                    pass

        return dict(
            result
        )

    return dict(
        result
    )


# =============================================================================
# 16. MLP ABLATION WRAPPER
# =============================================================================

class GemmaMLPAblationWrapper(torch.nn.Module):

    def __init__(self, original_mlp):
        super().__init__()

        # IMPORTANT:
        # original_mlp MUST be the real Gemma MLP.
        if is_ablation_wrapper(original_mlp):
            raise TypeError(
                "GemmaMLPAblationWrapper received another "
                "GemmaMLPAblationWrapper instead of the original "
                "Gemma MLP."
            )

        required_attributes = [
            "gate_proj",
            "up_proj",
            "down_proj",
            "act_fn",
        ]

        missing = [
            name
            for name in required_attributes
            if not hasattr(original_mlp, name)
        ]

        if missing:
            raise TypeError(
                "The supplied module does not look like a "
                f"Gemma MLP. Missing: {missing}. "
                f"Module type: {type(original_mlp)}"
            )

        self.original_mlp = original_mlp

        self.enabled = False

        # FINAL PRECOMPUTED NEURONS TO ABLATE.
        self.ablation_neurons = {}

        self.layer_index = -1

    def configure(
        self,
        enabled,
        ablation_neurons=None,
    ):

        self.enabled = bool(enabled)

        self.ablation_neurons = (
            ablation_neurons or {}
        )

    def disable(self):

        self.enabled = False

        self.ablation_neurons = {}

    def forward(self, x):

        # ---------------------------------------------------------------------
        # NORMAL GEMMA MLP
        # ---------------------------------------------------------------------

        gate = self.original_mlp.gate_proj(x)

        up = self.original_mlp.up_proj(x)

        activated_gate = (
            self.original_mlp.act_fn(gate)
        )

        # ---------------------------------------------------------------------
        # DIRECT ABLATION
        #
        # The neurons in self.ablation_neurons are ALREADY the final
        # neurons to ablate.
        #
        # No subtraction.
        # No intersection.
        # No high/low bias logic.
        # No attention logic.
        # ---------------------------------------------------------------------

        if self.enabled:

            neuron_indices = (
                self.ablation_neurons.get(
                    self.layer_index,
                    set(),
                )
            )

            if neuron_indices:

                hidden_size = (
                    activated_gate.shape[-1]
                )

                valid_indices = sorted(
                    idx
                    for idx in neuron_indices
                    if 0 <= idx < hidden_size
                )

                if valid_indices:

                    activated_gate = (
                        activated_gate.clone()
                    )

                    activated_gate[
                        ...,
                        valid_indices
                    ] = 0

        # ---------------------------------------------------------------------
        # NORMAL GEMMA MLP CONTINUATION
        # ---------------------------------------------------------------------

        hidden = (
            activated_gate * up
        )

        return self.original_mlp.down_proj(
            hidden
        )

# =============================================================================
# 17. FIND GEMMA DECODER LAYERS
# =============================================================================

def get_decoder_layers(model):
    """
    Locate Gemma decoder layers.
    """

    candidates = [
        (
            "language_model.model.layers",
            lambda m: getattr(
                getattr(
                    getattr(
                        m,
                        "language_model",
                        None,
                    ),
                    "model",
                    None,
                ),
                "layers",
                None,
            ),
        ),
        (
            "model.language_model.layers",
            lambda m: getattr(
                getattr(
                    m,
                    "model",
                    None,
                ),
                "language_model.layers",
                None,
            ),
        ),
        (
            "model.layers",
            lambda m: getattr(
                getattr(
                    m,
                    "model",
                    None,
                ),
                "layers",
                None,
            ),
        ),
        (
            "language_model.layers",
            lambda m: getattr(
                getattr(
                    m,
                    "language_model",
                    None,
                ),
                "layers",
                None,
            ),
        ),
    ]

    for name, getter in candidates:

        try:

            layers = getter(
                model
            )

        except Exception:

            layers = None

        if layers is not None:

            try:

                if len(layers) > 0:

                    print(
                        f"Decoder layers found at: "
                        f"{name}"
                    )

                    return layers

            except Exception:

                pass

    # -------------------------------------------------------------------------
    # Recursive fallback.
    # -------------------------------------------------------------------------

    for module_name, module in (
        model.named_modules()
    ):

        try:

            if (
                hasattr(
                    module,
                    "__len__",
                )
                and len(module) == 62
            ):

                first = module[0]

                class_name = (
                    first.__class__.__name__
                )

                if (
                    "DecoderLayer"
                    in class_name
                ):

                    print(
                        "Decoder layers found by "
                        f"recursive search: "
                        f"{module_name}"
                    )

                    return module

        except Exception:

            continue

    raise RuntimeError(
        "Could not locate Gemma decoder layers."
    )


# =============================================================================
# 18. INSTALL WRAPPERS
# =============================================================================
def is_ablation_wrapper(module):
    """
    Detect an ablation wrapper robustly, including wrappers created by
    an earlier execution of this Colab cell/class definition.
    """

    return (
        module.__class__.__name__ == "GemmaMLPAblationWrapper"
        and hasattr(module, "original_mlp")
    )


def unwrap_original_mlp(module, layer_index):
    """
    Recover the real Gemma MLP from any number of previously-created
    ablation wrappers.

    This intentionally does NOT use isinstance(), because Colab can
    redefine GemmaMLPAblationWrapper between cell executions.
    """

    current = module
    unwrap_count = 0

    while is_ablation_wrapper(current):

        if not hasattr(current, "original_mlp"):
            raise RuntimeError(
                f"Layer {layer_index}: detected an ablation wrapper "
                "without original_mlp."
            )

        current = current.original_mlp
        unwrap_count += 1

        if unwrap_count > 100:
            raise RuntimeError(
                f"Layer {layer_index}: excessive nested wrappers "
                "detected."
            )

    return current, unwrap_count


def verify_gemma_mlp(module, layer_index):
    """
    Verify that a module is the real Gemma MLP.
    """

    required_attributes = [
        "gate_proj",
        "up_proj",
        "down_proj",
        "act_fn",
    ]

    missing = [
        name
        for name in required_attributes
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            f"Layer {layer_index}: recovered module is not a "
            f"Gemma MLP.\n"
            f"Module type: {type(module)}\n"
            f"Module class: {module.__class__.__name__}\n"
            f"Missing attributes: {missing}"
        )

    return True


def install_ablation_wrappers(model):
    """
    Install exactly ONE current GemmaMLPAblationWrapper per decoder layer.

    Robust to Colab re-execution:
        - detects wrappers from older class definitions
        - removes arbitrary wrapper nesting
        - preserves the original Gemma MLP
        - installs exactly one current wrapper
    """

    layers = get_decoder_layers(model)

    wrappers = []

    print()
    print("=" * 80)
    print("INSTALLING ABLATION WRAPPERS")
    print("=" * 80)

    for i, layer in enumerate(layers):

        current_mlp = layer.mlp

        # ---------------------------------------------------------------------
        # Recover the REAL original MLP regardless of how many old wrappers
        # exist and regardless of which Python class definition created them.
        # ---------------------------------------------------------------------

        original_mlp, unwrap_count = (
            unwrap_original_mlp(
                current_mlp,
                i,
            )
        )

        if unwrap_count:

            print(
                f"Layer {i}: removed "
                f"{unwrap_count} existing wrapper(s)."
            )

        # ---------------------------------------------------------------------
        # Verify that what remains is genuinely the Gemma MLP.
        # ---------------------------------------------------------------------

        verify_gemma_mlp(
            original_mlp,
            i,
        )

        # ---------------------------------------------------------------------
        # Install exactly ONE wrapper from the CURRENT class definition.
        # ---------------------------------------------------------------------

        wrapper = GemmaMLPAblationWrapper(
            original_mlp
        )

        wrapper.layer_index = i

        layer.mlp = wrapper

        wrappers.append(
            wrapper
        )

    # -------------------------------------------------------------------------
    # Structural verification.
    # -------------------------------------------------------------------------

    if len(wrappers) != len(layers):

        raise RuntimeError(
            f"Expected {len(layers)} wrappers but "
            f"created {len(wrappers)}."
        )

    for i, layer in enumerate(layers):

        current = layer.mlp

        # Current wrapper must be present.
        if not isinstance(
            current,
            GemmaMLPAblationWrapper,
        ):

            raise RuntimeError(
                f"Layer {i}: current MLP is not a "
                "GemmaMLPAblationWrapper."
            )

        # Its underlying module must NOT be a wrapper.
        if is_ablation_wrapper(
            current.original_mlp
        ):

            raise RuntimeError(
                f"Layer {i}: nested ablation wrapper remains "
                "inside the current wrapper."
            )

        # Underlying module must be a real Gemma MLP.
        verify_gemma_mlp(
            current.original_mlp,
            i,
        )

    print()
    print(
        f"Installed exactly one wrapper on "
        f"{len(wrappers)} layers."
    )

    print(
        "Wrapper structure verified:"
    )

    print(
        "    Current Wrapper -> Original Gemma MLP"
    )

    print(
        "No nested wrappers detected."
    )

    print(
        "No model weights were modified."
    )

    return wrappers

# =============================================================================
# 19. LOAD PROCESSOR
# =============================================================================

print()
print("=" * 80)
print("LOADING PROCESSOR")
print("=" * 80)

processor = (
    AutoProcessor.from_pretrained(
        MODEL_NAME,
        padding_side="left",
    )
)

print(
    "Processor loaded."
)


# =============================================================================
# 20. GPU MEMORY CHECK
# =============================================================================

def print_gpu_memory(
    label,
):
    """
    Print current CUDA memory usage.
    """

    print()
    print(
        f"GPU MEMORY — {label}"
    )

    for i in range(
        torch.cuda.device_count()
    ):

        allocated = (
            torch.cuda.memory_allocated(i)
            / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved(i)
            / (1024 ** 3)
        )

        free, total = (
            torch.cuda.mem_get_info(i)
        )

        free_gb = (
            free
            / (1024 ** 3)
        )

        total_gb = (
            total
            / (1024 ** 3)
        )

        print(
            f"GPU {i}: "
            f"free={free_gb:.2f} GB | "
            f"total={total_gb:.2f} GB | "
            f"allocated={allocated:.2f} GB | "
            f"reserved={reserved:.2f} GB"
        )


# =============================================================================
# 21. LOAD OR REUSE SINGLE GEMMA 3 27B MODEL
# =============================================================================

print()
print("=" * 80)
print("CHECKING FOR EXISTING GEMMA 3 27B MODEL")
print("=" * 80)

# -----------------------------------------------------------------------------
# Check whether a model object already exists in the current Python runtime.
# -----------------------------------------------------------------------------

existing_model = globals().get("model", None)

if existing_model is not None:

    print()
    print("Existing `model` object found.")
    print(
        f"Existing model type: "
        f"{type(existing_model)}"
    )

    # -------------------------------------------------------------------------
    # Basic validation.
    # -------------------------------------------------------------------------

    if not isinstance(
        existing_model,
        torch.nn.Module,
    ):

        raise TypeError(
            "An object named `model` already exists, "
            "but it is not a torch.nn.Module."
        )

    model = existing_model

    model.eval()

    print()
    print(
        "REUSING EXISTING MODEL."
    )

    print(
        "No new model will be downloaded."
    )

    print(
        f"Model device: {model.device}"
    )

else:

    print()
    print(
        "No existing `model` object found."
    )

    print(
        "Loading Gemma 3 27B..."
    )

    # -------------------------------------------------------------------------
    # Clean unused CUDA cache before loading.
    # -------------------------------------------------------------------------

    gc.collect()

    torch.cuda.empty_cache()

    print_gpu_memory(
        "BEFORE MODEL LOAD"
    )

    # -------------------------------------------------------------------------
    # Load model only when one does not already exist.
    # -------------------------------------------------------------------------

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
    )

    model.eval()

    print()
    print(
        "NEW MODEL LOADED."
    )

    print(
        f"Model device: {model.device}"
    )

print_gpu_memory(
    "MODEL READY"
)

print()
print(
    "Model instance ready for experiment."
)

print(
    "Model instances loaded by this script: 1"
)

# =============================================================================
# 22. INSTALL ABLATION WRAPPERS
# =============================================================================

wrappers = (
    install_ablation_wrappers(
        model
    )
)

# =============================================================================
# DEFUNCT
# 22A. CD-DOLA LAYER OUTPUT CAPTURE
# =============================================================================

class CDDoLAHiddenStateCapture:

    def __init__(self):
        self.hidden_states = {}

    def clear(self):
        self.hidden_states.clear()

    def make_hook(self, layer_index):

        def hook(module, inputs, output):

            # Gemma decoder layers normally return their hidden state
            # as the first element of the output tuple.
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            self.hidden_states[layer_index] = hidden

        return hook


cd_dola_capture = CDDoLAHiddenStateCapture()

cd_dola_hooks = []

if CD_DOLA:

    decoder_layers = get_decoder_layers(model)

    # Capture only the layers needed by CD-DOLA.
    capture_layers = sorted(
        set(
            CD_DOLA_CANDIDATE_LAYERS
            # + [CD_DOLA_MATURE_LAYER]
        )
    )

    for layer_index in capture_layers:

        hook = decoder_layers[
            layer_index
        ].register_forward_hook(
            cd_dola_capture.make_hook(
                layer_index
            )
        )

        cd_dola_hooks.append(hook)

    print()
    print(
        "CD-DOLA hidden-state hooks installed."
    )

    print(
        f"Candidate layers: "
        f"{CD_DOLA_CANDIDATE_LAYERS}"
    )

    print(
        f"Mature layer: "
        f"{CD_DOLA_MATURE_LAYER}"
    )

# =============================================================================
# DEFUNCT
# 22B. CONVERT DECODER HIDDEN STATE TO LOGITS
# =============================================================================

def hidden_state_to_logits(
    hidden_state,
):
    """
    Convert a decoder-layer hidden state into vocabulary logits
    using Gemma's final normalization and language-model head.
    """

    normalized = model.model.language_model.norm(
        hidden_state
    )

    logits = model.lm_head(
        normalized
    )

    return logits

# =============================================================================
# DEFUNCT
# 22C. CD-DOLA PREMATURE-LAYER SELECTION
# =============================================================================

def select_cd_dola_layer(
    candidate_logits,
    mature_logits,
    candidate_layers,
):
    """
    Select the premature layer using the CD-DOLA implementation's
    Jensen-Shannon divergence criterion.

    Parameters
    ----------
    candidate_logits:
        Tensor of shape:
            [num_candidate_layers, batch, vocab]

    mature_logits:
        Tensor of shape:
            [batch, vocab]

    candidate_layers:
        List of actual decoder-layer indices corresponding to
        candidate_logits.

    Returns
    -------
    selected_layer:
        Actual decoder layer index selected for this decoding step.

    selected_index:
        Index into candidate_layers.
    """

    # -------------------------------------------------------------------------
    # Convert mature and premature logits to probability distributions.
    # -------------------------------------------------------------------------

    softmax_mature_layer = torch.softmax(
        mature_logits,
        dim=-1,
    )

    softmax_premature_layers = torch.softmax(
        candidate_logits,
        dim=-1,
    )

    # -------------------------------------------------------------------------
    # Mixture distribution:
    #
    # M = 0.5 * (P_mature + P_premature)
    #
    # Shape:
    #     [num_candidate_layers, batch, vocab]
    # -------------------------------------------------------------------------

    M = 0.5 * (
        softmax_mature_layer.unsqueeze(0)
        +
        softmax_premature_layers
    )

    # -------------------------------------------------------------------------
    # Jensen-Shannon divergence:
    #
    # JSD(P || Q)
    #     = 0.5 KL(P || M)
    #     + 0.5 KL(Q || M)
    #
    # Add a small epsilon to avoid log(0).
    # -------------------------------------------------------------------------

    eps = torch.finfo(
        softmax_mature_layer.dtype
    ).eps

    mature_prob = (
        softmax_mature_layer.unsqueeze(0)
        .clamp_min(eps)
    )

    premature_prob = (
        softmax_premature_layers
        .clamp_min(eps)
    )

    mixture_prob = (
        M.clamp_min(eps)
    )

    kl_mature = (
        mature_prob
        * (
            mature_prob.log()
            -
            mixture_prob.log()
        )
    ).sum(
        dim=-1
    )

    kl_premature = (
        premature_prob
        * (
            premature_prob.log()
            -
            mixture_prob.log()
        )
    ).sum(
        dim=-1
    )

    js_divs = 0.5 * (
        kl_mature
        +
        kl_premature
    )

    # -------------------------------------------------------------------------
    # The authors average the JSD over the batch before selecting the layer.
    #
    # Result:
    #     [num_candidate_layers]
    # -------------------------------------------------------------------------

    js_divs = js_divs.mean(
        dim=-1
    )

    # -------------------------------------------------------------------------
    # Select the candidate layer with maximum JSD.
    # -------------------------------------------------------------------------

    selected_index = int(
        js_divs.argmax()
        .detach()
        .cpu()
        .item()
    )

    selected_layer = candidate_layers[
        selected_index
    ]

    return (
        selected_layer,
        selected_index,
    )


# =============================================================================
# DEFUNCT
# 22D. CD-DOLA CONTRASTIVE LOGITS
# =============================================================================

def compute_cd_dola_logits(
    candidate_logits,
    mature_logits,
    selected_index,
):
    """
    Compute the CD-DOLA contrastive logits using the authors'
    log-probability formulation.

    Parameters
    ----------
    candidate_logits:
        Tensor of shape:
            [num_candidate_layers, batch, vocab]

    mature_logits:
        Tensor of shape:
            [batch, vocab]

    selected_index:
        Index of the candidate layer selected by 22C.

    Returns
    -------
    cd_dola_logits:
        Tensor of shape:
            [batch, vocab]
    """

    # -------------------------------------------------------------------------
    # Select the premature-layer logits chosen by the JSD criterion.
    # -------------------------------------------------------------------------

    premature_logits = (
        candidate_logits[
            selected_index
        ]
    )

    # -------------------------------------------------------------------------
    # Convert both mature and premature logits to log probabilities.
    #
    # This is the contrastive operation used by the authors:
    #
    #     log P_mature(token)
    #       -
    #     log P_premature(token)
    # -------------------------------------------------------------------------

    mature_log_probs = (
        mature_logits.log_softmax(
            dim=-1
        )
    )

    premature_log_probs = (
        premature_logits.log_softmax(
            dim=-1
        )
    )

    # -------------------------------------------------------------------------
    # Contrast mature vs premature prediction.
    # -------------------------------------------------------------------------

    diff_logits = (
        mature_log_probs
        -
        premature_log_probs
    )

    # -------------------------------------------------------------------------
    # Normalize the contrastive scores into a log-probability
    # distribution before decoding.
    # -------------------------------------------------------------------------

    cd_dola_logits = (
        diff_logits.log_softmax(
            dim=-1
        )
    )

    return cd_dola_logits

# =============================================================================
# 23. TOTAL COMBINATIONS
# =============================================================================

TOTAL_COMBINATIONS = (
    len(BIAS_TYPES)
    *
    len(THRESHOLDS)
    *
    len(TOP_K_VALUES)
)

print()
print("=" * 80)
print("TOTAL COMBINATIONS")
print("=" * 80)

print(
    f"TOTAL COMBINATIONS: "
    f"{TOTAL_COMBINATIONS}"
)

print(
    f"TOTAL SENTENCES PER COMBINATION: "
    f"{len(sentences):,}"
)

print(
    "TOTAL MODEL INSTANCES LOADED: 1"
)


# =============================================================================
# 24. OUTPUT PATH HELPERS
# =============================================================================

def sanitize_filename(
    value,
):

    value = str(
        value
    )

    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    )

    return value.strip(
        "_"
    )


def combination_filename(
    experiment_type,
    threshold,
    top_k,
):
    return (
        f"Gemma3-27B_"
        f"{sanitize_filename(experiment_type)}_"
        f"threshold-{threshold}_"
        f"topk-{top_k}.csv"
    )


def local_output_path(
    experiment_type,
    threshold,
    top_k,
):
    filename = combination_filename(
        experiment_type,
        threshold,
        top_k,
    )

    return Path(OUTPUT_ROOT) / filename


def drive_output_path(
    experiment_type,
    threshold,
    top_k,
):
    filename = combination_filename(
        experiment_type,
        threshold,
        top_k,
    )

    return Path(DRIVE_OUTPUT_ROOT) / filename


# =============================================================================
# 25. PREPARE NEURONS FOR A COMBINATION
# =============================================================================

def prepare_combination_neurons(
    high_bias_data,
    low_bias_data,
    high_attention_data,
    low_attention_data,
    top_k,
):
    """
    Extract and normalize all four neuron sets
    for the same top-k.
    """

    high_bias_raw = (
        get_neurons_for_k(
            high_bias_data,
            top_k,
        )
    )

    low_bias_raw = (
        get_neurons_for_k(
            low_bias_data,
            top_k,
        )
    )

    high_attention_raw = (
        get_neurons_for_k(
            high_attention_data,
            top_k,
        )
    )

    low_attention_raw = (
        get_neurons_for_k(
            low_attention_data,
            top_k,
        )
    )

    high_bias = (
        normalize_neuron_indices(
            high_bias_raw
        )
    )

    low_bias = (
        normalize_neuron_indices(
            low_bias_raw
        )
    )

    high_attention = (
        normalize_neuron_indices(
            high_attention_raw
        )
    )

    low_attention = (
        normalize_neuron_indices(
            low_attention_raw
        )
    )

    return (
        high_bias,
        low_bias,
        high_attention,
        low_attention,
    )


# =============================================================================
# 26. CONFIGURE ALL LAYERS
# =============================================================================

def configure_ablation(
    enabled,
    ablation_neurons=None,
):
    """
    Configure all wrappers using the final precomputed
    neuron set.

    The supplied neurons are already the exact neurons
    to ablate. No additional operations are performed.
    """

    for wrapper in wrappers:

        if enabled:

            wrapper.configure(
                enabled=True,
                ablation_neurons=ablation_neurons,
            )

        else:

            wrapper.disable()

# =============================================================================
# 27. PROCESSOR INPUT CREATION
# =============================================================================

def build_prompt(
    sentence,
    prompt_template,
):

    return prompt_template.format(
        sentence=sentence
    )


def tokenize_batch(
    batch_sentences,
    prompt_template,
):
    """
    Tokenize a batch of text-only Gemma 3 inputs.

    Uses apply_chat_template() to construct the prompts,
    then explicitly moves returned tensors to the GPU.
    """

    batch_messages = []

    for sentence in batch_sentences:

        prompt = build_prompt(
            sentence,
            prompt_template,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        batch_messages.append(
            messages
        )

    # -------------------------------------------------------------------------
    # First create the formatted chat text.
    # -------------------------------------------------------------------------

    formatted_prompts = (
        processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    )

    # -------------------------------------------------------------------------
    # Tokenize the formatted prompts explicitly.
    # -------------------------------------------------------------------------

    inputs = processor(
        text=formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # -------------------------------------------------------------------------
    # Move only tensor inputs to GPU.
    # -------------------------------------------------------------------------

    for key, value in inputs.items():

        if torch.is_tensor(value):

            inputs[key] = value.to(
                DEVICE
            )

    return inputs

# =============================================================================
# 28. GENERATE - ABLATION ONLY
# =============================================================================

@torch.inference_mode()
def generate_batch(
    batch_sentences,
    prompt_template,
):
    """
    Generate ordinary ablation-only responses for one batch.

    Returns tokenizer-level representations with literal backslashes
    marking token boundaries.
    """

    inputs = (
        tokenize_batch(
            batch_sentences,
            prompt_template,
        )
    )

    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "use_cache": True,
        "pad_token_id": (
            processor
            .tokenizer
            .pad_token_id
        ),
    }

    if DO_SAMPLE:

        generation_kwargs.update(
            {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            }
        )

    outputs = model.generate(
        **inputs,
        **generation_kwargs,
    )

    input_length = (
        inputs[
            "input_ids"
        ].shape[1]
    )

    generated_only = (
        outputs[
            :,
            input_length:
        ]
    )

    return [
        [
            token
            for token_id, token in zip(
                token_ids.tolist(),
                processor.tokenizer.convert_ids_to_tokens(
                    token_ids.tolist()
                ),
            )
            if token_id not in processor.tokenizer.all_special_ids
        ]
        for token_ids in generated_only
    ]

# =============================================================================
# 28A. CD-DOLA GENERATION
# =============================================================================
@torch.no_grad()
def generate_cd_dola_batch(
    batch_sentences,
):
    """
    CD-DoLa decoding for Gemma 3.

    Uses:
        BATCH_SIZE
        MAX_NEW_TOKENS
        DO_SAMPLE
        CD_DOLA_CANDIDATE_LAYERS
        CD_DOLA_MATURE_LAYER

    Returns:
        generated_tokens:
            LongTensor [batch_size, generated_length]

        selected_layers:
            List[List[int]]
            One selected premature layer per generated token.
    """

    device = model.device

    special_token_ids = set(
        processor.tokenizer.all_special_ids
    )

    # -------------------------------------------------------------------------
    # Prepare inputs
    # -------------------------------------------------------------------------

    model_inputs = tokenize_batch(
        batch_sentences,
        PROMPT2_TEMPLATE,
    )

    model_inputs = {
        k: v.to(device)
        for k, v in model_inputs.items()
        if torch.is_tensor(v)
    }

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", None)

    batch_size = input_ids.shape[0]

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------

    generated_tokens = [[] for _ in range(batch_size)]
    selected_layers = [[] for _ in range(batch_size)]

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Initial forward pass
    #
    # This processes the entire prompt and initializes the KV cache.
    # We request hidden states because CD-DoLa needs the representations
    # from the candidate layers and mature layer.
    # -------------------------------------------------------------------------

    outputs = model(
        **model_inputs,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )

    past_key_values = outputs.past_key_values

    # Last prompt position gives the distribution for the first new token.
    #
    # hidden_states:
    #   [0]     = embedding output
    #   [1]     = output of decoder layer 0
    #   ...
    #   [62]    = output of decoder layer 61
    #
    # We use layer + 1 because hidden_states[0] is the embedding output.
    # -------------------------------------------------------------------------

    # Gemma 3's final RMSNorm is model.model.language_model.norm.
    final_norm = model.model.language_model.norm

    lm_head = model.lm_head

    # -------------------------------------------------------------------------
    # Generation loop
    # -------------------------------------------------------------------------

    for step in range(MAX_NEW_TOKENS):

        hidden_states = outputs.hidden_states

        # ---------------------------------------------------------------------
        # Mature-layer logits
        # ---------------------------------------------------------------------

        mature_hidden = hidden_states[
            CD_DOLA_MATURE_LAYER + 1
        ][:, -1, :]

        # The mature layer is the final decoder layer, so apply Gemma's
        # final RMSNorm before the LM head.
        # mature_hidden = final_norm(mature_hidden)

        mature_logits = lm_head(mature_hidden)

        # ---------------------------------------------------------------------
        # Candidate-layer logits
        # ---------------------------------------------------------------------

        candidate_logits = []

        for layer_idx in CD_DOLA_CANDIDATE_LAYERS:

            h = hidden_states[layer_idx + 1][:, -1, :]
            h = final_norm(h)
            # Premature-layer hidden states are projected directly through
            # the LM head, matching the DoLa early-exit construction.
            candidate_logits.append(
                lm_head(h)
            )

        candidate_logits = torch.stack(
            candidate_logits,
            dim=0,
        )
        # Shape:
        #   [num_candidates, batch_size, vocab_size]

        # ---------------------------------------------------------------------
        # Dynamic premature-layer selection
        #
        # This follows the author's JSD-based selection procedure.
        # ---------------------------------------------------------------------

        mature_probs = F.softmax(
            mature_logits,
            dim=-1,
        )

        premature_probs = F.softmax(
            candidate_logits,
            dim=-1,
        )

        # M = 0.5 * (mature + premature)
        M = 0.5 * (
            mature_probs.unsqueeze(0)
            + premature_probs
        )

        mature_log_probs = F.log_softmax(
            mature_logits,
            dim=-1,
        )

        premature_log_probs = F.log_softmax(
            candidate_logits,
            dim=-1,
        )

        # Author's implementation:
        #
        # kl1 = KL(mature || M)
        # kl2 = KL(premature || M)
        #
        # followed by averaging over vocabulary and batch.
        # ---------------------------------------------------------------------

        kl1 = F.kl_div(
            mature_log_probs.unsqueeze(0),
            M,
            reduction="none",
        ).mean(-1)

        kl2 = F.kl_div(
            premature_log_probs,
            M,
            reduction="none",
        ).mean(-1)

        js_divs = 0.5 * (kl1 + kl2)

        # Average across the batch, exactly as in the author's code.
        js_divs = js_divs.mean(-1)

        selected_candidate_idx = int(
            js_divs.argmax().item()
        )

        selected_layer = CD_DOLA_CANDIDATE_LAYERS[
            selected_candidate_idx
        ]

        # Record the same selected layer for every active example in
        # this batch. The author's implementation selects one layer for
        # the whole batch at a generation step.
        # for b in range(batch_size):
        #    if not finished[b]:
        #        selected_layers[b].append(selected_layer)

        # ---------------------------------------------------------------------
        # Contrast mature and selected premature distributions
        # ---------------------------------------------------------------------

        premature_logits = candidate_logits[
            selected_candidate_idx
        ]

        mature_log_probs = F.log_softmax(
            mature_logits,
            dim=-1,
        )

        premature_log_probs = F.log_softmax(
            premature_logits,
            dim=-1,
        )

        # DoLa contrast:
        #
        # log p_mature - log p_premature
        # ---------------------------------------------------------------------

        diff_logits = (
            mature_log_probs
            - premature_log_probs
        )

        # Convert contrastive scores into the final token distribution.
        diff_logits = F.log_softmax(
            diff_logits,
            dim=-1,
        )

        # -------------------------------------------------------------------------
        # Relative-top filtering
        #
        # Match the authors' DoLa implementation.
        #
        # Only tokens whose mature-layer probability is sufficiently close
        # to the mature-layer maximum are allowed to participate in the
        # contrastive distribution.
        # -------------------------------------------------------------------------

        relative_top = 0.1
        relative_top_value = -1000.0

        probs_max = mature_log_probs.max(
            dim=-1,
            keepdim=True,
        ).values

        probs_thresh = (
            probs_max + math.log(relative_top)
        )

        relative_top_mask = (
            mature_log_probs < probs_thresh
        )

        diff_logits = torch.where(
            relative_top_mask,
            torch.full_like(
                diff_logits,
                relative_top_value,
            ),
            diff_logits,
        )

        # ---------------------------------------------------------------------
        # Decode
        # ---------------------------------------------------------------------

        if DO_SAMPLE:
            probs = diff_logits.exp()

            next_token = torch.multinomial(
                probs,
                num_samples=1,
            ).squeeze(-1)

        else:
            next_token = diff_logits.argmax(
                dim=-1
            )

        # ---------------------------------------------------------------------
        # Record tokens
        # ---------------------------------------------------------------------
        for b in range(batch_size):

            if not finished[b]:

                token_id = int(
                    next_token[b].item()
                )

                if token_id not in special_token_ids:

                    generated_tokens[b].append(
                        token_id
                    )

                    selected_layers[b].append(
                        selected_layer
                    )

        # ---------------------------------------------------------------------
        # Stop on EOS
        # ---------------------------------------------------------------------

        eos_token_id = model.generation_config.eos_token_id

        if eos_token_id is not None:

            if isinstance(eos_token_id, int):
                eos_ids = [eos_token_id]
            else:
                eos_ids = list(eos_token_id)

            newly_finished = torch.zeros_like(
                finished
            )

            for eos_id in eos_ids:
                newly_finished |= (
                    next_token == eos_id
                )

            finished |= newly_finished

        if finished.all():
            break

        # ---------------------------------------------------------------------
        # Feed generated token through the model using the KV cache
        #
        # This produces the hidden states needed to choose the layer for
        # the NEXT token.
        # ---------------------------------------------------------------------

        next_input_ids = next_token.unsqueeze(-1)

        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (batch_size, 1),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=-1,
            )

        outputs = model(
            input_ids=next_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        past_key_values = outputs.past_key_values

    # -------------------------------------------------------------------------
    # Convert generated token IDs to tokenizer-token lists.
    #
    # Return the token lists themselves.
    # Formatting for dataframe storage (backslash-delimited) and
    # reconstruction for Phase 3 are handled outside this function.
    # -------------------------------------------------------------------------

    token_lists = []

    special_token_ids = set(
        processor.tokenizer.all_special_ids
    )

    for tokens in generated_tokens:

        token_lists.append(
            [
                token
                for token_id, token in zip(
                    tokens,
                    processor.tokenizer.convert_ids_to_tokens(
                        tokens
                    ),
                )
                if token_id not in special_token_ids
            ]
        )

    return (
        token_lists,
        selected_layers,
    )

def format_token_sequence(
    token_list,
    delimiter="",
):
    if delimiter == "":
        return processor.tokenizer.convert_tokens_to_string(
            token_list
        ).strip()

    return delimiter.join(
        processor.tokenizer.convert_tokens_to_string(
            [token]
        )
        for token in token_list
    )

# =============================================================================
# 29. DRIVE CHECKPOINT
# =============================================================================

def save_checkpoint_to_drive(
    rows,
    drive_path,
):
    """
    Save current accumulated rows to Google Drive.

    Uses a temporary file and atomic replacement.
    """

    drive_path = Path(
        drive_path
    )

    drive_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = Path(
        str(drive_path)
        + ".tmp"
    )

    checkpoint_df = pd.DataFrame(
        rows
    )

    checkpoint_df.to_csv(
        temp_path,
        index=False,
    )

    os.replace(
        temp_path,
        drive_path,
    )


# =============================================================================
# 30. RESUME CHECK
# =============================================================================

def is_output_complete(
    path,
):
    """
    Determine whether an existing output contains all sentences.
    """

    path = Path(
        path
    )

    if not path.exists():

        return False

    if path.stat().st_size <= 0:

        return False

    try:

        existing = pd.read_csv(
            path,
            usecols=["sentence"],
        )

        return (
            len(existing)
            >=
            len(sentences)
        )

    except Exception:

        return False

# =============================================================================
# 30A. PARSE PHASE 3 SCORES
# =============================================================================

def extract_phase3_scores(response):
    """
    Extract Framing, Emotional, and Divisive scores from
    the Phase 3 response.

    Expected format:

        Framing: [Score]
        Emotional: [Score]
        Divisive: [Score]
    """

    if not isinstance(response, str):
        return {
            "Framing": None,
            "Emotional": None,
            "Divisive": None,
        }

    framing_match = re.search(
        r"Framing\s*:\s*\[?\s*(\d+(?:\.\d+)?)\s*\]?",
        response,
        re.IGNORECASE,
    )

    emotional_match = re.search(
        r"Emotional\s*:\s*\[?\s*(\d+(?:\.\d+)?)\s*\]?",
        response,
        re.IGNORECASE,
    )

    divisive_match = re.search(
        r"Divisive\s*:\s*\[?\s*(\d+(?:\.\d+)?)\s*\]?",
        response,
        re.IGNORECASE,
    )

    return {
        "Framing": (
            float(framing_match.group(1))
            if framing_match
            else None
        ),

        "Emotional": (
            float(emotional_match.group(1))
            if emotional_match
            else None
        ),

        "Divisive": (
            float(divisive_match.group(1))
            if divisive_match
            else None
        ),
    }

# =============================================================================
# 31. RUN ONE PHASE
# =============================================================================
def run_combination(
    experiment_type,
    threshold,
    top_k,
    ablation_neurons,
    input_sentences,
):
    """
    Run Phase 2 and Phase 3 and save both results
    into ONE CSV.

    Phase 2 response becomes the input sentence
    for Phase 3.
    """

    local_path = local_output_path(
        experiment_type,
        threshold,
        top_k,
    )

    drive_path = drive_output_path(
        experiment_type,
        threshold,
        top_k,
    )

    # ------------------------------------------------------------
    # Configure Phase 2
    # ------------------------------------------------------------

    configure_ablation(
        enabled=True,
        ablation_neurons=ablation_neurons,
    )


    rows = []

    start_time = time.time()

    total = len(input_sentences)

    running_score_total = 0.0
    running_score_count = 0

    running_cd_dola_score_total = 0.0
    running_cd_dola_score_count = 0

    print()
    print("=" * 80)
    print("PHASE 2 + PHASE 3 — SINGLE CSV")
    print("=" * 80)

    print(f"Type       = {experiment_type}")
    print(f"Threshold  = {threshold}")
    print(f"Top-k      = {top_k}")
    print(f"Sentences  = {total:,}")

    print(f"Local      = {local_path}")
    print(f"Drive      = {drive_path}")

    print("=" * 80)

    # ------------------------------------------------------------
    # Process batches
    # ------------------------------------------------------------

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch_sentences = input_sentences[start:end]

        # ========================================================
        # PHASE 2 — ABLATION ON
        # ========================================================

        phase2_tokens = generate_batch(
            batch_sentences,
            PROMPT2_TEMPLATE,
        )

        if CD_DOLA:

            (
                phase_2_cd_dola_tokens,
                cd_dola_layers,
            ) = generate_cd_dola_batch(
                batch_sentences,
            )
        else:
            phase_2_cd_dola_tokens = None
            cd_dola_layers = None

        # ========================================================
        # PHASE 3 — ABLATION OFF
        #
        # Phase 3 receives the Phase 2 responses.
        # ========================================================

        configure_ablation(
            enabled=False,
        )

        phase2_responses = [
            format_token_sequence(
                token_list,
                delimiter="",
            )
            for token_list in phase2_tokens
        ]

        phase3_tokens = generate_batch(
            phase2_responses,
            PROMPT3_TEMPLATE,
        )

        #-------

        if CD_DOLA:
            phase2_cd_dola_responses = [
                format_token_sequence(
                    token_list,
                    delimiter="",
                )
                for token_list in phase_2_cd_dola_tokens
            ]

            phase3_cd_dola_tokens = generate_batch(
                phase2_cd_dola_responses,
                PROMPT3_TEMPLATE,
            )
        else:
            phase2_cd_dola_responses = None
            phase3_cd_dola_tokens = None

        # ========================================================
        # Restore Phase 2 ablation for next batch
        # ========================================================

        configure_ablation(
            enabled=True,
            ablation_neurons=ablation_neurons,
        )

        # ========================================================
        # Store both Phase 2 representations and Phase 3 response
        # in the SAME ROW
        # ========================================================

        for i, (original_sentence, phase3_token) in enumerate(
            zip(batch_sentences, phase3_tokens)
        ):

            # ============================================================
            # NON-CD-DOLA OUTPUTS
            # ============================================================

            phase_2_token_string = format_token_sequence(
                phase2_tokens[i],
                delimiter="\\",
            )

            phase_3_response = format_token_sequence(
                phase3_tokens[i],
                delimiter="",
            )

            # ============================================================
            # CD-DOLA OUTPUTS
            # ============================================================

            if CD_DOLA:

                phase_2_cd_dola_token_string = (
                    format_token_sequence(
                        phase_2_cd_dola_tokens[i],
                        delimiter="\\",
                    )
                )

                cd_dola_layer_string = "\\".join(
                    str(layer)
                    for layer in cd_dola_layers[i]
                )

                phase_3_cd_dola_response = (
                    format_token_sequence(
                        phase3_cd_dola_tokens[i],
                        delimiter="",
                    )
                )

            else:

                phase_2_cd_dola_token_string = None
                cd_dola_layer_string = None
                phase_3_cd_dola_response = None

            # ============================================================
            # SCORE NON-DOLA
            # ============================================================

            scores = extract_phase3_scores(
                phase_3_response
            )

            current_score = scores.get(
                experiment_type
            )

            if current_score is not None:

                running_score_total += current_score
                running_score_count += 1

            # ============================================================
            # SCORE CD-DOLA
            # ============================================================

            if CD_DOLA:

                cd_dola_scores = extract_phase3_scores(
                    phase_3_cd_dola_response
                )

                current_cd_dola_score = (
                    cd_dola_scores.get(
                        experiment_type
                    )
                )

                if current_cd_dola_score is not None:

                    running_cd_dola_score_total += (
                        current_cd_dola_score
                    )

                    running_cd_dola_score_count += 1

            # ============================================================
            # SAVE ROW
            # ============================================================

            rows.append(
                {
                    "sentence": original_sentence,

                    "phase_2_tok_str":
                        phase_2_token_string,

                    "phase_2_cd_dola_tok_str":
                        phase_2_cd_dola_token_string,

                    "cd_dola_layers":
                        cd_dola_layer_string,

                    "phase_3_response":
                        phase_3_response,

                    "phase_3_cd_dola_response":
                        phase_3_cd_dola_response,

                    "bias_type":
                        experiment_type,

                    "threshold":
                        threshold,

                    "top_k":
                        top_k,

                    "model":
                        "Gemma3-27B",
                }
            )

            print(
                ".",
                end="",
                flush=True,
            )

        processed = end

        # ========================================================
        # PRINT RUNNING AVERAGES EVERY 128 SENTENCES
        #
        # The averages are cumulative:
        #
        #   128  -> sentences 1-128
        #   256  -> sentences 1-256
        #   384  -> sentences 1-384
        #   ...
        #
        # Both Non-DoLa and CD-DoLa are reported separately.
        #
        # Only the CURRENT bias type is reported.
        # ========================================================

        if (
            processed % 32 == 0
            or processed == total
        ):

            print()
            print("=" * 80)

            if processed % 32 == 0:

                print(
                    f"RUNNING {experiment_type.upper()} "
                    f"AVERAGES — SENTENCES 1-{processed:,}"
                )

            else:

                print(
                    f"FINAL {experiment_type.upper()} "
                    f"AVERAGES — SENTENCES 1-{processed:,}"
                )

            print(
                f"Bias type: {experiment_type}"
            )

            print(
                f"Threshold: {threshold}"
            )

            print(
                f"Top-k: {top_k}"
            )

            print("-" * 80)

            # ====================================================
            # NON-DOLA
            # ====================================================

            if running_score_count > 0:

                running_average = (
                    running_score_total
                    / running_score_count
                )

                print(
                    f"Non-DoLa {experiment_type}: "
                    f"{running_average:.4f} "
                    f"(n={running_score_count:,})"
                )

            else:

                print(
                    f"Non-DoLa {experiment_type}: "
                    "No valid scores found."
                )

            # ====================================================
            # CD-DOLA
            # ====================================================

            if CD_DOLA:

                if running_cd_dola_score_count > 0:

                    running_cd_dola_average = (
                        running_cd_dola_score_total
                        / running_cd_dola_score_count
                    )

                    print(
                        f"CD-DoLa {experiment_type}: "
                        f"{running_cd_dola_average:.4f} "
                        f"(n={running_cd_dola_score_count:,})"
                    )

                else:

                    print(
                        f"CD-DoLa {experiment_type}: "
                        "No valid scores found."
                    )

            print("=" * 80)

        # ========================================================
        # Progress
        # ========================================================

        if (
            processed % PROGRESS_EVERY == 0
            or processed == total
        ):

            elapsed = (
                time.time()
                - start_time
            )

            rate = (
                processed / elapsed
                if elapsed > 0
                else 0
            )

            remaining = total - processed

            eta = (
                remaining / rate
                if rate > 0
                else 0
            )

            print()

            log(
                f"Progress: "
                f"{processed:,}/{total:,} "
                f"({processed / total * 100:6.2f}%) | "
                f"{rate:.2f} sentences/s | "
                f"ETA {eta / 60:.1f} min"
            )

        # ========================================================
        # SAVE ONE CSV AFTER EVERY BATCH
        # ========================================================

        save_checkpoint_to_drive(
            rows,
            drive_path,
        )

        save_checkpoint_to_drive(
            rows,
            local_path,
        )

        log(
            f"Checkpoint saved: "
            f"{processed:,}/{total:,}"
        )

    # ------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------

    result_df = pd.DataFrame(rows)

    result_df.to_csv(
        local_path,
        index=False,
    )

    shutil.copy2(
        local_path,
        drive_path,
    )

    # ------------------------------------------------------------
    # Disable ablation
    # ------------------------------------------------------------

    configure_ablation(
        enabled=False,
    )


    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 80)

    log(
        "Completed Phase 2 + Phase 3."
    )

    print(
        f"Rows: {len(result_df):,}"
    )

    print(
        f"Local file: {local_path}"
    )

    print(
        f"Drive file: {drive_path}"
    )

    print(
        f"Time: {elapsed / 60:.2f} min"
    )

    print("=" * 80)

    torch.cuda.empty_cache()
    gc.collect()

    return rows

# =============================================================================
# 32. MAIN EXPERIMENT LOOP
# =============================================================================

combination_number = 0

experiment_start = time.time()

for experiment_type in BIAS_TYPES:

    for threshold in THRESHOLDS:

        print()
        print("=" * 80)

        print(
            f"TYPE = {experiment_type}"
        )

        print(
            f"THRESHOLD = {threshold}"
        )

        print(
            f"Sentences = {len(sentences):,}"
        )

        print(
            "Using PRECOMPUTED combined-ablation neurons."
        )

        print(
            "No additional neuron selection or "
            "set operations will be performed."
        )

        print("=" * 80)

        for top_k in TOP_K_VALUES:

            combination_number += 1

            print()
            print("=" * 80)

            print(
                f"COMBINATION "
                f"{combination_number}/"
                f"{TOTAL_COMBINATIONS}"
            )

            print(
                f"Timestamp : {timestamp()}"
            )

            print(
                f"Type      : {experiment_type}"
            )

            print(
                f"Threshold : {threshold}"
            )

            print(
                f"Top-k     : {top_k}"
            )

            print("=" * 80)

            # =================================================================
            # FIND PRECOMPUTED COMBINED ABLATION FILE
            # =================================================================

            combined_file = (
                find_combined_ablation_file(
                    combined_root=COMBINED_ABLATION_ROOT,
                    experiment_type=experiment_type,
                    threshold=threshold,
                    top_k=top_k,
                )
            )

            # =================================================================
            # LOAD FINAL ABLATION NEURONS
            # =================================================================

            ablation_neurons = (
                load_combined_ablation_file(
                    combined_file
                )
            )

            # =================================================================
            # RUN PHASE 2 + PHASE 3
            # =================================================================

            run_combination(
                experiment_type=experiment_type,
                threshold=threshold,
                top_k=top_k,
                ablation_neurons=ablation_neurons,
                input_sentences=sentences,
            )

            # =================================================================
            # CLEANUP
            # =================================================================

            configure_ablation(
                enabled=False,
            )

            torch.cuda.empty_cache()

            gc.collect()

# =============================================================================
# 33. FINAL CLEANUP
# =============================================================================

configure_ablation(
    enabled=False,
)


torch.cuda.empty_cache()

gc.collect()


# =============================================================================
# 34. FINAL SUMMARY
# =============================================================================

total_elapsed = (
    time.time()
    - experiment_start
)

print()
print()
print("=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)

print(
    f"End timestamp: {timestamp()}"
)

print(
    "Model: Gemma 3 27B"
)

print(
    "Model instances loaded: 1"
)

print(
    f"Bias types: {BIAS_TYPES}"
)

print(
    f"Thresholds: {THRESHOLDS}"
)

print(
    f"Top-k values: {TOP_K_VALUES}"
)

print(
    f"Combinations: {TOTAL_COMBINATIONS}"
)

print(
    f"Sentences per combination: "
    f"{len(sentences):,}"
)

print(
    "Phase 2: Ablation ON"
)

print(
    "Phase 3: Ablation OFF"
)

print()
print(
    "Local output directory:"
)

print(
    OUTPUT_ROOT
)

print()
print(
    "Google Drive output directory:"
)

print(
    DRIVE_OUTPUT_ROOT
)

print(
    f"Total elapsed time: "
    f"{total_elapsed / 3600:.2f} hours"
)


print("=" * 80)

