"""Turning a text prompt into the chunks the model generates one at a time.

The model is trained on single sentences, so long inputs are split on sentence
boundaries and regrouped into chunks that fit `max_tokens`.
"""

import logging

logger = logging.getLogger(__name__)


def prepare_text_prompt(
    text: str, pad_with_spaces_for_short_inputs: bool, remove_semicolons: bool
) -> tuple[str, int]:
    text = text.strip()
    if text == "":
        raise ValueError("Text prompt cannot be empty")
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
    if remove_semicolons:
        text = text.replace(";", ",")
    number_of_words = len(text.split())
    if number_of_words <= 4:
        frames_after_eos_guess = 3
    else:
        frames_after_eos_guess = 1

    # Make sure it starts with an uppercase letter
    if not text[0].isupper():
        text = text[0].upper() + text[1:]

    # Let's make sure it ends with some kind of punctuation
    # If it ends with a letter or digit, we add a period.
    if text[-1].isalnum():
        text = text + "."

    # The model does not perform well when there are very few tokens, so
    # we can add empty spaces at the beginning to increase the token count.
    if pad_with_spaces_for_short_inputs and len(text.split()) < 5:
        text = " " * 8 + text

    return text, frames_after_eos_guess


def _is_decimal_period_boundary(
    list_of_tokens: list[int], segment_start_idx: int, tokenizer
) -> bool:
    """Return True when segment_start_idx begins right after a decimal period."""
    prefix = tokenizer.sp.decode(list_of_tokens[:segment_start_idx])
    suffix = tokenizer.sp.decode(list_of_tokens[segment_start_idx:])
    return (
        len(prefix) >= 2
        and prefix[-1] == "."
        and prefix[-2].isdigit()
        and bool(suffix)
        and suffix[0].isdigit()
    )


def _find_boundary_indices(
    list_of_tokens: list[int],
    boundary_tokens: list[int],
    tokenizer=None,
    skip_decimal_periods: bool = False,
) -> list[int]:
    """Find token indices where text should be split based on boundary tokens.

    Returns a list of boundary positions used to slice segments. Each consecutive
    pair (indices[i], indices[i+1]) defines one segment. The first element is
    always 0 and the last is always len(list_of_tokens).
    """
    boundary_set = set(boundary_tokens)
    indices = [0]
    previous_was_boundary = False
    for idx, token in enumerate(list_of_tokens):
        if token in boundary_set:
            previous_was_boundary = True
        else:
            if previous_was_boundary:
                if (
                    skip_decimal_periods
                    and tokenizer is not None
                    and _is_decimal_period_boundary(list_of_tokens, idx, tokenizer)
                ):
                    previous_was_boundary = False
                    continue
                indices.append(idx)
            previous_was_boundary = False
    indices.append(len(list_of_tokens))
    return indices


def _segments_from_boundaries(
    list_of_tokens: list[int], boundary_indices: list[int], tokenizer
) -> list[tuple[int, str]]:
    """Decode token segments between boundary indices into (token_count, text) pairs."""
    segments = []
    for i in range(len(boundary_indices) - 1):
        start = boundary_indices[i]
        end = boundary_indices[i + 1]
        text = tokenizer.sp.decode(list_of_tokens[start:end])
        segments.append((end - start, text))
    return segments


def split_into_best_sentences(
    tokenizer,
    text_to_generate: str,
    max_tokens: int,
    pad_with_spaces_for_short_inputs: bool,
    remove_semicolons: bool,
) -> list[str]:
    text_to_generate, _ = prepare_text_prompt(
        text_to_generate, pad_with_spaces_for_short_inputs, remove_semicolons
    )
    text_to_generate = text_to_generate.strip()
    tokens = tokenizer(text_to_generate)
    list_of_tokens = tokens.tokens[0].tolist()

    _, *end_of_sentence_tokens = tokenizer(".!...?").tokens[0].tolist()
    sentence_boundaries = _find_boundary_indices(
        list_of_tokens, end_of_sentence_tokens, tokenizer, skip_decimal_periods=True
    )
    nb_tokens_and_sentences = _segments_from_boundaries(
        list_of_tokens, sentence_boundaries, tokenizer
    )

    # Sub-split oversized sentences on commas, semicolons, and colons to prevent skipped words
    _, *fallback_tokens = tokenizer(",;:").tokens[0].tolist()
    refined_segments = []
    for nb_tokens, text in nb_tokens_and_sentences:
        if nb_tokens <= max_tokens:
            refined_segments.append((nb_tokens, text))
        else:
            sub_tokens = tokenizer(text.strip()).tokens[0].tolist()
            sub_boundaries = _find_boundary_indices(sub_tokens, fallback_tokens)
            sub_segments = _segments_from_boundaries(sub_tokens, sub_boundaries, tokenizer)
            if len(sub_segments) > 1:
                refined_segments.extend(sub_segments)
            else:
                refined_segments.append((nb_tokens, text))

    max_nb_tokens_in_a_chunk = max_tokens
    chunks = []
    current_chunk = ""
    current_nb_of_tokens_in_chunk = 0
    for nb_tokens, sentence in refined_segments:
        if current_chunk == "":
            current_chunk = sentence
            current_nb_of_tokens_in_chunk = nb_tokens
            continue

        if current_nb_of_tokens_in_chunk + nb_tokens > max_nb_tokens_in_a_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_nb_of_tokens_in_chunk = nb_tokens
        else:
            current_chunk += " " + sentence
            current_nb_of_tokens_in_chunk += nb_tokens

    if current_chunk != "":
        chunks.append(current_chunk.strip())

    for chunk in chunks:
        chunk_tokens = tokenizer(chunk.strip()).tokens[0].tolist()
        if len(chunk_tokens) > max_tokens:
            logger.warning(
                "Chunk has %d tokens (max %d), generation may skip words: '%.50s...'",
                len(chunk_tokens),
                max_tokens,
                chunk,
            )

    return chunks
