# ============================================================
# Cell 1
# Build tokenized corpora for neuron discovery
# ============================================================

import re
import torch
import pandas as pd
from transformers import AutoTokenizer


# ============================================================
# Configuration
# ============================================================

CSV_PATH = "/content/BABEDataset+Models-All_Colab_Batches.csv"

MODEL_NAME = "google/gemma-3-27b-it"

# Xa = AFTER / rewritten style
# Xb = BEFORE / original style

XA_OUTPUT = "/content/Xa.gemma"
XB_OUTPUT = "/content/Xb.gemma"


# ============================================================
# Load CSV
# ============================================================

print("Loading CSV...")

df = pd.read_csv(CSV_PATH)

print(f"Loaded {len(df):,} rows")


# ============================================================
# Extract paired corpora
# ============================================================
#
# Xb = original sentence
# Xa = rewritten sentence
#
# All examples are included.
# ============================================================

Xa_sentences = []
Xb_sentences = []

print("Extracting examples...")


for _, row in df.iterrows():

    orig = row["sentence"]
    rewrite = row["output_2"]

    if pd.isna(orig) or pd.isna(rewrite):
        continue

    # --------------------------------------------------------
    # Xb = BEFORE / original
    # Xa = AFTER / rewritten
    # --------------------------------------------------------

    Xb_sentences.append(
        str(orig)
    )

    Xa_sentences.append(
        str(rewrite)
    )


print(
    f"Xa examples (after):  "
    f"{len(Xa_sentences):,}"
)

print(
    f"Xb examples (before): "
    f"{len(Xb_sentences):,}"
)


# ============================================================
# Sanity check
# ============================================================

if len(Xa_sentences) != len(Xb_sentences):

    raise RuntimeError(
        "Xa and Xb do not contain the same "
        "number of examples."
    )


# ============================================================
# Load tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)


# ============================================================
# Save tokenized sentences
# ============================================================

def save_tokenized_sentences(
    texts,
    output_path,
):

    tokenized = []

    print(
        f"\nTokenizing -> {output_path}"
    )

    for i, text in enumerate(texts):

        ids = tokenizer.encode(
            text,
            add_special_tokens=True,
        )

        tokenized.append(
            torch.tensor(
                ids,
                dtype=torch.long,
            )
        )

        if (i + 1) % 1000 == 0:

            print(
                f"  {i+1:,}/{len(texts):,}"
            )

    torch.save(
        tokenized,
        output_path,
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Sentences: {len(tokenized):,}"
    )


# ============================================================
# Save Xa
# ============================================================

save_tokenized_sentences(
    Xa_sentences,
    XA_OUTPUT,
)


# ============================================================
# Save Xb
# ============================================================

save_tokenized_sentences(
    Xb_sentences,
    XB_OUTPUT,
)


# ============================================================
# Complete
# ============================================================

print()
print("Xa = AFTER / rewritten")
print("Xb = BEFORE / original")
print()
print("✅ Cell 1 complete.")

# ============================================================
# Cell 2
# Load Gemma 3 27B and initialize activation statistics
# ============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "google/gemma-3-27b-it"

TOKEN_FILES = {
    "Xa": "/content/Xa.gemma",
    "Xb": "/content/Xb.gemma",
}

OUTPUT_DIR = "/content"

BATCH_SIZE = 1


# ============================================================
# Load model
# ============================================================

print("🚀 Loading Gemma 3 27B...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("✅ Gemma 3 27B loaded")


# ============================================================
# Model configuration
# ============================================================

text_config = model.config.text_config

num_layers = text_config.num_hidden_layers
intermediate_size = text_config.intermediate_size

print()
print("========== MODEL INFO ==========")
print(f"Model: {MODEL_NAME}")
print(f"Layers: {num_layers}")
print(f"MLP hidden size: {intermediate_size}")
print(f"Activation: {text_config.hidden_activation}")
print("================================")
print()


# ============================================================
# Locate Gemma language model
# ============================================================

language_model = model.model.language_model

print("Language model type:")
print(type(language_model))

print()

print("Decoder layer type:")
print(type(language_model.layers[0]))

print()

print("MLP type:")
print(type(language_model.layers[0].mlp))

print()


# ============================================================
# Statistics device
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required.")

stats_device = torch.device("cuda")

print(f"Statistics device: {stats_device}")


# ============================================================
# Activation statistics
# ============================================================
#
# We follow the authors' activation.py conceptually.
#
# activation_sum:
#     Sum of post-activation MLP neuron values.
#
# positive_count:
#     Number of token positions where activation > 0.
#
# The authors call this statistic "over_zero".
#
# Shape:
#
#     [num_layers, intermediate_size]
# ============================================================

activation_sum = torch.zeros(
    num_layers,
    intermediate_size,
    dtype=torch.float64,
    device=stats_device,
)

positive_count = torch.zeros(
    num_layers,
    intermediate_size,
    dtype=torch.long,
    device=stats_device,
)


# ============================================================
# Token counter
# ============================================================

total_tokens = 0


# ============================================================
# Report
# ============================================================

print()
print("Activation statistics initialized.")

print(
    f"activation_sum shape: "
    f"{tuple(activation_sum.shape)}"
)

print(
    f"positive_count shape: "
    f"{tuple(positive_count.shape)}"
)

print(
    f"total_tokens: {total_tokens}"
)

# ============================================================
# Cell 3
# Hook Gemma MLP activations
#
# This cell:
#
#   1. Defines the score type to filter on.
#   2. Defines thresholds 10 -> 1.
#   3. Determines which sentence indices pass each threshold.
#   4. Installs the Gemma MLP activation collectors.
#
# Cell 4 will run the model separately for every threshold.
#
# ============================================================

import re
import torch
import pandas as pd
from types import MethodType


# ============================================================
# Configuration
# ============================================================

CSV_PATH = "/content/BABEDataset+Models-All_Colab_Batches.csv"

# Score type to filter on.
#
# Valid examples:
#
#     "Framing"
#     "Emotional"
#     "Divisive"
#
TYPE = "Emotional"

# Run thresholds from 10 down to 1.
THRESHOLDS = [6]


# ============================================================
# Load CSV
# ============================================================

print("Loading CSV for sentence filtering...")

df = pd.read_csv(
    CSV_PATH
)

print(
    f"Loaded {len(df):,} rows"
)


# ============================================================
# Score parser
# ============================================================

score_pattern = re.compile(
    r"(Framing|Emotional|Divisive)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def parse_scores(text):

    if pd.isna(text):

        return {}

    matches = score_pattern.findall(
        str(text)
    )

    return {
        score_type.lower(): float(score)
        for score_type, score in matches
    }


# ============================================================
# Validate TYPE
# ============================================================

VALID_TYPES = {
    "framing",
    "emotional",
    "divisive",
}

if TYPE.lower() not in VALID_TYPES:

    raise ValueError(
        f"Invalid TYPE: {TYPE}\n"
        f"Valid types: "
        f"{sorted(VALID_TYPES)}"
    )


# ============================================================
# Determine allowed sentence indices
# ============================================================
#
# Cell 1 saves only rows where BOTH:
#
#     sentence
#     output_2
#
# are present.
#
# Therefore we maintain a separate sequential token index.
#
# token_idx = position in Xa.gemma / Xb.gemma
#
# This guarantees that the indices correspond exactly to
# the tokenized sentence files.
#
# ============================================================

print()
print(
    f"Filtering by TYPE = {TYPE}"
)

allowed_indices_by_threshold = {
    threshold: []
    for threshold in THRESHOLDS
}


token_idx = 0


for _, row in df.iterrows():

    # --------------------------------------------------------
    # Match Cell 1's sentence inclusion logic
    # --------------------------------------------------------

    orig = row["sentence"]
    rewrite = row["output_2"]

    if pd.isna(orig) or pd.isna(rewrite):

        continue


    # --------------------------------------------------------
    # This token_idx is the actual index in:
    #
    #     Xa.gemma
    #     Xb.gemma
    #
    # --------------------------------------------------------

    scores = parse_scores(
        row["output_1"]
    )

    score = scores.get(
        TYPE.lower()
    )


    # --------------------------------------------------------
    # Add this token index to every threshold it satisfies.
    # --------------------------------------------------------

    if score is not None:

        for threshold in THRESHOLDS:

            if score >= threshold:

                allowed_indices_by_threshold[
                    threshold
                ].append(
                    token_idx
                )


    # --------------------------------------------------------
    # Advance token index ONLY when Cell 1 would have
    # created a tokenized sentence.
    # --------------------------------------------------------

    token_idx += 1


# ============================================================
# Verify against token-file size
# ============================================================

EXPECTED_TOKEN_SENTENCES = 1807

if token_idx != EXPECTED_TOKEN_SENTENCES:

    raise RuntimeError(
        "Sentence/token index mismatch!\n"
        f"Cell 3 found {token_idx:,} valid "
        f"sentence/rewrite pairs, but the token files "
        f"contain {EXPECTED_TOKEN_SENTENCES:,} sentences.\n\n"
        "This means Cell 1 and Cell 3 are not using "
        "the same sentence inclusion rules."
    )


# ============================================================
# Report
# ============================================================

for threshold in THRESHOLDS:

    indices = (
        allowed_indices_by_threshold[
            threshold
        ]
    )

    print(
        f"Threshold {threshold:2d}: "
        f"{len(indices):,} sentences"
    )

# ============================================================
# MLP forward hook
# ============================================================

def factory(layer_idx):

    def gemma_mlp_forward(self, x):

        global total_tokens

        # ====================================================
        # Standard Gemma gated MLP
        # ====================================================

        gate = self.gate_proj(x)

        up = self.up_proj(x)

        activated_gate = self.act_fn(
            gate
        )


        # ====================================================
        # Get attention mask
        # ====================================================

        mask = current_attention_mask


        # ====================================================
        # Flatten activations
        # ====================================================

        act = activated_gate.reshape(
            -1,
            activated_gate.shape[-1]
        )


        # ====================================================
        # Flatten attention mask
        # ====================================================

        mask = mask.reshape(-1)


        # ====================================================
        # Keep ONLY real tokens
        # ====================================================

        valid = (
            mask.to(
                device=act.device
            )
            .bool()
        )

        act = act[valid]


        # ====================================================
        # Nothing to collect
        # ====================================================

        if act.numel() == 0:

            return self.down_proj(
                activated_gate * up
            )


        # ====================================================
        # Numerical stability
        # ====================================================

        act64 = act.double()


        # ====================================================
        # Sum of activations
        # ====================================================

        activation_sum[layer_idx] += (
            act64.sum(dim=0)
        )


        # ====================================================
        # Number of positive activations
        # ====================================================

        positive_count[layer_idx] += (
            (act > 0)
            .sum(dim=0)
            .to(torch.long)
        )


        # ====================================================
        # Count real tokens
        #
        # Only layer 0 counts tokens once.
        # ====================================================

        if layer_idx == 0:

            total_tokens += (
                act.shape[0]
            )


        # ====================================================
        # Continue normal Gemma MLP
        # ====================================================

        hidden = (
            activated_gate * up
        )

        return self.down_proj(
            hidden
        )

    return gemma_mlp_forward


# ============================================================
# Global attention mask
# ============================================================

current_attention_mask = None


# ============================================================
# Attach collectors
# ============================================================

print()
print(
    "🪝 Attaching Gemma MLP activation collectors..."
)


for layer_idx in range(
    num_layers
):

    mlp = (
        language_model
        .layers[layer_idx]
        .mlp
    )

    mlp.forward = MethodType(
        factory(layer_idx),
        mlp,
    )


print(
    f"✅ Collectors attached to "
    f"{num_layers} layers."
)


# ============================================================
# Summary
# ============================================================

print()
print("========== CELL 3 CONFIGURATION ==========")

print(
    f"Filter type: {TYPE}"
)

print(
    f"Thresholds:   {THRESHOLDS}"
)

print()
print("Sentence counts:")

for threshold in THRESHOLDS:

    print(
        f"  {TYPE} >= {threshold:2d}: "
        f"{len(allowed_indices_by_threshold[threshold]):,}"
    )

print("===========================================")

print()
print("✅ Cell 3 complete.")

# ============================================================
# Cell 4
# Run Xa / Xb corpora through Gemma
#
# Thresholds:
#
#     10, 9, 8, ..., 1
#
# For every threshold:
#
#     sentence token IDs
#             ↓
#     FILTER by allowed_indices
#             ↓
#     concatenate into one token stream
#             ↓
#     fixed-length blocks
#             ↓
#     attention mask
#             ↓
#     Gemma forward pass
#             ↓
#     activation statistics
#
# ============================================================

import os
import torch
from tqdm.auto import tqdm


# ============================================================
# Configuration
# ============================================================

DATASETS = {
    "Xa": "/content/Xa.gemma",
    "Xb": "/content/Xb.gemma",
}

BLOCK_LEN = 2048


# ============================================================
# Output filename
# ============================================================

def get_output_file(
    style,
    threshold,
):

    return (
        f"/content/"
        f"activation.{style}.train."
        f"Gemma3-27B-{TYPE}-{threshold}.pt"
    )


# ============================================================
# Determine block length
# ============================================================

if BLOCK_LEN is None:

    BLOCK_LEN = getattr(
        model.config,
        "max_position_embeddings",
        None,
    )

    if BLOCK_LEN is None:

        BLOCK_LEN = getattr(
            model.config.text_config,
            "max_position_embeddings",
            None,
        )

    if BLOCK_LEN is None:

        raise RuntimeError(
            "Could not determine model context length. "
            "Set BLOCK_LEN manually."
        )


print(
    f"Fixed block length: {BLOCK_LEN:,}"
)


# ============================================================
# Load tokenized corpus
# ============================================================

def load_tokenized_corpus(path):

    print()
    print(f"Loading: {path}")

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Tokenized corpus not found:\n"
            f"  {path}\n\n"
            f"Run Cell 1 first."
        )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(data, list):

        raise TypeError(
            "Expected tokenized corpus to be "
            "a list of tensors."
        )

    print(
        f"Sequences: {len(data):,}"
    )

    return data


# ============================================================
# Filter tokenized corpus
# ============================================================

def filter_tokenized_corpus(
    tokenized_corpus,
    allowed_indices,
):

    print(
        "Applying sentence filter..."
    )

    n_sentences = len(
        tokenized_corpus
    )

    selected = []

    for idx in allowed_indices:

        if idx < 0 or idx >= n_sentences:

            raise IndexError(
                f"Allowed index {idx} is outside "
                f"the tokenized corpus range "
                f"[0, {n_sentences - 1}]."
            )

        selected.append(
            tokenized_corpus[idx]
        )

    print(
        f"Selected sequences: "
        f"{len(selected):,}"
    )

    print(
        f"Excluded sequences: "
        f"{n_sentences - len(selected):,}"
    )

    return selected


# ============================================================
# Concatenate token IDs
# ============================================================

def concatenate_token_stream(
    tokenized_corpus
):

    print(
        "Concatenating token IDs..."
    )

    pieces = []

    for ids in tokenized_corpus:

        if not isinstance(
            ids,
            torch.Tensor,
        ):

            raise TypeError(
                "Each corpus element must be "
                "a torch.Tensor."
            )

        if ids.numel() > 0:

            pieces.append(
                ids.to(
                    dtype=torch.long
                )
            )

    if len(pieces) == 0:

        raise RuntimeError(
            "Corpus contains no tokens."
        )

    ids = torch.cat(
        pieces,
        dim=0,
    )

    print(
        f"Total tokens in corpus: "
        f"{ids.numel():,}"
    )

    return ids


# ============================================================
# Reset statistics
# ============================================================

def reset_statistics():

    global total_tokens

    activation_sum.zero_()

    positive_count.zero_()

    total_tokens = 0


# ============================================================
# Process one corpus
# ============================================================

def process_corpus(
    style,
    tokenized_corpus,
    allowed_indices,
    threshold,
):

    global total_tokens
    global current_attention_mask

    print()
    print("=" * 70)
    print(
        f"Processing {style} | "
        f"{TYPE} >= {threshold}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Reset statistics
    # --------------------------------------------------------

    reset_statistics()

    # --------------------------------------------------------
    # Apply sentence filter
    # --------------------------------------------------------

    tokenized_corpus = (
        filter_tokenized_corpus(
            tokenized_corpus,
            allowed_indices,
        )
    )

    # --------------------------------------------------------
    # Concatenate selected sentences
    # --------------------------------------------------------

    ids = concatenate_token_stream(
        tokenized_corpus
    )

    n_total = ids.numel()

    # --------------------------------------------------------
    # Number of blocks
    # --------------------------------------------------------

    n_blocks = (
        (n_total + BLOCK_LEN - 1)
        // BLOCK_LEN
    )

    print()
    print(
        f"Total tokens:    {n_total:,}"
    )

    print(
        f"Blocks:          {n_blocks:,}"
    )

    print(
        f"Block length:    {BLOCK_LEN:,}"
    )

    # --------------------------------------------------------
    # Model input device
    # --------------------------------------------------------

    input_device = (
        language_model
        .embed_tokens
        .weight
        .device
    )

    print(
        f"Model input device: "
        f"{input_device}"
    )

    # --------------------------------------------------------
    # Process blocks
    # --------------------------------------------------------

    for start in tqdm(
        range(
            0,
            n_total,
            BLOCK_LEN,
        ),
        desc=(
            f"{style} "
            f"{TYPE}>={threshold}"
        ),
    ):

        end = min(
            start + BLOCK_LEN,
            n_total,
        )

        block = ids[
            start:end
        ]

        actual_len = block.numel()

        # ----------------------------------------------------
        # Pad final block
        # ----------------------------------------------------

        if actual_len < BLOCK_LEN:

            pad_len = (
                BLOCK_LEN - actual_len
            )

            pad_token_id = (
                tokenizer.pad_token_id
            )

            if pad_token_id is None:

                pad_token_id = (
                    tokenizer.eos_token_id
                )

            padding = torch.full(
                (pad_len,),
                pad_token_id,
                dtype=torch.long,
            )

            block = torch.cat(
                [
                    block,
                    padding,
                ],
                dim=0,
            )

        # ----------------------------------------------------
        # Attention mask
        # ----------------------------------------------------

        attention_mask = torch.zeros(
            BLOCK_LEN,
            dtype=torch.long,
        )

        attention_mask[
            :actual_len
        ] = 1

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        batch = (
            block
            .unsqueeze(0)
            .to(input_device)
        )

        attention_mask = (
            attention_mask
            .unsqueeze(0)
            .to(input_device)
        )

        # ----------------------------------------------------
        # Tell Cell 3 which positions are real tokens.
        # ----------------------------------------------------

        current_attention_mask = (
            attention_mask
        )

        with torch.no_grad():

            model(
                input_ids=batch,
                attention_mask=attention_mask,
                use_cache=False,
            )

    # --------------------------------------------------------
    # Verify token count
    # --------------------------------------------------------

    if total_tokens != n_total:

        raise RuntimeError(
            "Token counter mismatch!\n"
            f"Collector counted: {total_tokens:,}\n"
            f"Expected:          {n_total:,}"
        )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    output_path = get_output_file(
        style,
        threshold,
    )

    output = {

        "activation_sum":
            activation_sum.cpu(),

        "positive_count":
            positive_count.cpu(),

        "over_zero":
            positive_count.cpu(),

        "n_tokens":
            total_tokens,

        "block_len":
            BLOCK_LEN,

        "model_name":
            MODEL_NAME,

        "style":
            style,

        "filter_type":
            TYPE,

        "threshold":
            threshold,
    }

    torch.save(
        output,
        output_path,
    )

    print()
    print(
        f"Saved: {output_path}"
    )

    print(
        f"n_tokens: "
        f"{total_tokens:,}"
    )


# ============================================================
# Load both tokenized corpora ONCE
# ============================================================

Xa_corpus = load_tokenized_corpus(
    DATASETS["Xa"]
)

Xb_corpus = load_tokenized_corpus(
    DATASETS["Xb"]
)


# ============================================================
# Run every threshold
# ============================================================

for threshold in THRESHOLDS:

    allowed_indices = (
        allowed_indices_by_threshold[
            threshold
        ]
    )

    print()
    print("#" * 80)
    print(
        f"# THRESHOLD {threshold}"
    )
    print("#" * 80)

    process_corpus(
        "Xa",
        Xa_corpus,
        allowed_indices,
        threshold,
    )

    process_corpus(
        "Xb",
        Xb_corpus,
        allowed_indices,
        threshold,
    )


# ============================================================
# Cleanup
# ============================================================

del Xa_corpus
del Xb_corpus

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# Complete
# ============================================================

print()
print("=" * 70)
print("✅ Cell 4 complete")
print("=" * 70)

# ============================================================
# Cell 5
# Identify top-k neurons for Xa and Xb
#
# Runs separately for every threshold:
#
#     10, 9, 8, ..., 1
#
# Ranking statistic:
#
#     Xa: over_zero(Xa) - over_zero(Xb)
#     Xb: over_zero(Xb) - over_zero(Xa)
#
# ============================================================

import torch


# ============================================================
# Configuration
# ============================================================

K_VALUES = list(
    [14000]
)


# ============================================================
# Activation filename
# ============================================================

def get_activation_file(
    style,
    threshold,
):

    return (
        f"/content/"
        f"activation.{style}.train."
        f"Gemma3-27B-{TYPE}-{threshold}.pt"
    )


# ============================================================
# Top-k output filename
# ============================================================

def get_top_k_file(
    threshold,
):

    return (
        f"/content/"
        f"top_k_neurons."
        f"Gemma3-27B-{TYPE}-{threshold}.pt"
    )


# ============================================================
# Process every threshold
# ============================================================

for threshold in THRESHOLDS:

    print()
    print("=" * 80)
    print(
        f"PROCESSING THRESHOLD: "
        f"{TYPE} >= {threshold}"
    )
    print("=" * 80)


    # ========================================================
    # Load activation statistics
    # ========================================================

    Xa_file = get_activation_file(
        "Xa",
        threshold,
    )

    Xb_file = get_activation_file(
        "Xb",
        threshold,
    )

    print()
    print(
        f"Loading Xa: {Xa_file}"
    )

    Xa_data = torch.load(
        Xa_file,
        map_location="cpu",
        weights_only=True,
    )

    print(
        f"Loading Xb: {Xb_file}"
    )

    Xb_data = torch.load(
        Xb_file,
        map_location="cpu",
        weights_only=True,
    )


    # ========================================================
    # Extract over_zero
    # ========================================================

    Xa_over_zero = (
        Xa_data["positive_count"]
        .to(torch.long)
    )

    Xb_over_zero = (
        Xb_data["positive_count"]
        .to(torch.long)
    )


    # ========================================================
    # Verify shapes
    # ========================================================

    if (
        Xa_over_zero.shape
        != Xb_over_zero.shape
    ):

        raise RuntimeError(
            "Xa and Xb over_zero tensors "
            "have different shapes."
        )


    num_layers, intermediate_size = (
        Xa_over_zero.shape
    )

    num_neurons = (
        num_layers
        * intermediate_size
    )


    # ========================================================
    # Compute directional differences
    # ========================================================

    Xa_minus_Xb = (
        Xa_over_zero
        - Xb_over_zero
    )

    Xb_minus_Xa = (
        Xb_over_zero
        - Xa_over_zero
    )


    # ========================================================
    # Flatten
    # ========================================================

    Xa_scores_flat = (
        Xa_minus_Xb.reshape(-1)
    )

    Xb_scores_flat = (
        Xb_minus_Xa.reshape(-1)
    )


    # ========================================================
    # Containers
    # ========================================================

    top_k_neurons = {
        "Xa": {},
        "Xb": {},
    }

    top_k_scores = {
        "Xa": {},
        "Xb": {},
    }


    # ========================================================
    # Select top-k
    # ========================================================

    for k in K_VALUES:

        if k > num_neurons:

            raise ValueError(
                f"k={k} exceeds the total "
                f"number of neurons "
                f"({num_neurons})."
            )


        # ----------------------------------------------------
        # Xa
        # ----------------------------------------------------

        Xa_topk = torch.topk(
            Xa_scores_flat,
            k=k,
            largest=True,
        )

        Xa_global_indices = (
            Xa_topk.indices
        )

        Xa_scores = (
            Xa_topk.values
        )


        # ----------------------------------------------------
        # Xb
        # ----------------------------------------------------

        Xb_topk = torch.topk(
            Xb_scores_flat,
            k=k,
            largest=True,
        )

        Xb_global_indices = (
            Xb_topk.indices
        )

        Xb_scores = (
            Xb_topk.values
        )


        # ----------------------------------------------------
        # Convert flattened indices
        # ----------------------------------------------------

        Xa_layers = (
            Xa_global_indices
            // intermediate_size
        )

        Xa_neurons = (
            Xa_global_indices
            % intermediate_size
        )

        Xb_layers = (
            Xb_global_indices
            // intermediate_size
        )

        Xb_neurons = (
            Xb_global_indices
            % intermediate_size
        )


        # ----------------------------------------------------
        # Store [layer, neuron]
        # ----------------------------------------------------

        Xa_pairs = torch.stack(
            [
                Xa_layers,
                Xa_neurons,
            ],
            dim=1,
        ).to(torch.long)

        Xb_pairs = torch.stack(
            [
                Xb_layers,
                Xb_neurons,
            ],
            dim=1,
        ).to(torch.long)


        top_k_neurons["Xa"][k] = (
            Xa_pairs
        )

        top_k_neurons["Xb"][k] = (
            Xb_pairs
        )

        top_k_scores["Xa"][k] = (
            Xa_scores
        )

        top_k_scores["Xb"][k] = (
            Xb_scores
        )


    # ========================================================
    # Save
    # ========================================================

    OUTPUT_FILE = get_top_k_file(
        threshold
    )

    torch.save(
        {
            "K_VALUES": K_VALUES,

            "top_k_neurons":
                top_k_neurons,

            "top_k_scores":
                top_k_scores,

            "num_layers":
                num_layers,

            "intermediate_size":
                intermediate_size,

            "filter_type":
                TYPE,

            "threshold":
                threshold,
        },
        OUTPUT_FILE,
    )


    # ========================================================
    # Summary
    # ========================================================

    print()
    print(
        "========== TOP-K NEURONS =========="
    )

    print(
        f"Layers:            {num_layers:,}"
    )

    print(
        f"Neurons per layer: {intermediate_size:,}"
    )

    print(
        f"Total neurons:     {num_neurons:,}"
    )

    print()

    for k in K_VALUES:

        Xa_pairs = (
            top_k_neurons["Xa"][k]
        )

        Xb_pairs = (
            top_k_neurons["Xb"][k]
        )

        print(
            f"k={k:5d} | "
            f"Xa={Xa_pairs.shape[0]:5d} | "
            f"Xb={Xb_pairs.shape[0]:5d}"
        )

    print(
        "==================================="
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


# ============================================================
# Complete
# ============================================================

print()
print("=" * 70)
print("✅ Cell 5 complete")
print("=" * 70)

# ============================================================
# Cell 6
# Remove intersection between Xa and Xb neuron sets
#
# Runs separately for every threshold:
#
#     10, 9, 8, ..., 1
#
# For each k:
#
#     intersection = Xa ∩ Xb
#
#     Xa-specific = Xa \ Xb
#
#     Xb-specific = Xb \ Xa
#
# ============================================================

import os
import torch


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = "/content/neuron_sets"


# ============================================================
# Create output directory
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Top-k filename
# ============================================================

def get_top_k_file(
    threshold,
):

    return (
        f"/content/"
        f"top_k_neurons."
        f"Gemma3-27B-{TYPE}-{threshold}.pt"
    )


# ============================================================
# Output filenames
# ============================================================

def get_neuron_set_file(
    set_type,
    threshold,
):

    return os.path.join(
        OUTPUT_DIR,
        (
            f"{set_type}_neurons."
            f"Gemma3-27B-{TYPE}-{threshold}.pt"
        ),
    )


# ============================================================
# Process every threshold
# ============================================================

for threshold in THRESHOLDS:

    print()
    print("=" * 80)
    print(
        f"PROCESSING THRESHOLD: "
        f"{TYPE} >= {threshold}"
    )
    print("=" * 80)


    # ========================================================
    # Load top-k neurons
    # ========================================================

    TOP_K_FILE = get_top_k_file(
        threshold
    )

    print(
        f"Loading: {TOP_K_FILE}"
    )

    data = torch.load(
        TOP_K_FILE,
        map_location="cpu",
        weights_only=True,
    )


    # ========================================================
    # Extract
    # ========================================================

    top_k_neurons = (
        data["top_k_neurons"]
    )

    K_VALUES = (
        data["K_VALUES"]
    )

    num_layers = (
        data["num_layers"]
    )

    intermediate_size = (
        data["intermediate_size"]
    )


    # ========================================================
    # Output containers
    # ========================================================

    intersection_neurons = {}

    Xa_specific_neurons = {}

    Xb_specific_neurons = {}


    # ========================================================
    # Process every k
    # ========================================================

    for k in K_VALUES:

        Xa_pairs = (
            top_k_neurons["Xa"][k]
        )

        Xb_pairs = (
            top_k_neurons["Xb"][k]
        )


        # ----------------------------------------------------
        # Convert to sets
        # ----------------------------------------------------

        Xa_set = {
            (
                int(pair[0]),
                int(pair[1]),
            )
            for pair in Xa_pairs
        }

        Xb_set = {
            (
                int(pair[0]),
                int(pair[1]),
            )
            for pair in Xb_pairs
        }


        # ----------------------------------------------------
        # Intersection
        # ----------------------------------------------------

        intersection = (
            Xa_set & Xb_set
        )


        # ----------------------------------------------------
        # Xa-specific
        # ----------------------------------------------------

        Xa_only = (
            Xa_set - Xb_set
        )


        # ----------------------------------------------------
        # Xb-specific
        # ----------------------------------------------------

        Xb_only = (
            Xb_set - Xa_set
        )


        # ----------------------------------------------------
        # Convert back to tensors
        # ----------------------------------------------------

        intersection_neurons[k] = (
            torch.tensor(
                sorted(intersection),
                dtype=torch.long,
            ).reshape(-1, 2)
        )

        Xa_specific_neurons[k] = (
            torch.tensor(
                sorted(Xa_only),
                dtype=torch.long,
            ).reshape(-1, 2)
        )

        Xb_specific_neurons[k] = (
            torch.tensor(
                sorted(Xb_only),
                dtype=torch.long,
            ).reshape(-1, 2)
        )


    # ========================================================
    # Save files
    # ========================================================

    INTERSECTION_FILE = (
        get_neuron_set_file(
            "intersection",
            threshold,
        )
    )

    XA_FILE = (
        get_neuron_set_file(
            "Xa_specific",
            threshold,
        )
    )

    XB_FILE = (
        get_neuron_set_file(
            "Xb_specific",
            threshold,
        )
    )


    torch.save(
        {
            "neurons":
                intersection_neurons,

            "representation":
                "[layer, neuron]",

            "K_VALUES":
                K_VALUES,

            "filter_type":
                TYPE,

            "threshold":
                threshold,
        },
        INTERSECTION_FILE,
    )


    torch.save(
        {
            "neurons":
                Xa_specific_neurons,

            "representation":
                "[layer, neuron]",

            "K_VALUES":
                K_VALUES,

            "filter_type":
                TYPE,

            "threshold":
                threshold,
        },
        XA_FILE,
    )


    torch.save(
        {
            "neurons":
                Xb_specific_neurons,

            "representation":
                "[layer, neuron]",

            "K_VALUES":
                K_VALUES,

            "filter_type":
                TYPE,

            "threshold":
                threshold,
        },
        XB_FILE,
    )


    # ========================================================
    # Verify and report
    # ========================================================

    print()
    print(
        "========== NEURON SETS =========="
    )

    for k in K_VALUES:

        n_xa = (
            Xa_specific_neurons[k]
            .shape[0]
        )

        n_xb = (
            Xb_specific_neurons[k]
            .shape[0]
        )

        n_intersection = (
            intersection_neurons[k]
            .shape[0]
        )


        # ----------------------------------------------------
        # Sanity checks
        # ----------------------------------------------------

        assert (
            n_xa + n_intersection == k
        )

        assert (
            n_xb + n_intersection == k
        )


        print(
            f"k={k:5d} | "
            f"Xa-only={n_xa:5d} | "
            f"Xb-only={n_xb:5d} | "
            f"intersection={n_intersection:5d}"
        )


    print(
        "================================="
    )


    # ========================================================
    # Report
    # ========================================================

    print()
    print(
        f"Xa-specific:  {XA_FILE}"
    )

    print(
        f"Xb-specific:  {XB_FILE}"
    )

    print(
        f"Intersection: {INTERSECTION_FILE}"
    )


# ============================================================
# Complete
# ============================================================

print()
print("=" * 70)
print("✅ Cell 6 complete")
print("=" * 70)
