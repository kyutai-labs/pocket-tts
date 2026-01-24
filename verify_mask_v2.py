import torch
import torch.nn.functional as F


def test_mask():
    print("Testing SDPA Mask Logic...")
    # Q: [1, 1, 1, 1]
    q = torch.ones(1, 1, 1, 1)
    # K: [10, -10]
    k = torch.tensor([[[[10.0], [-10.0]]]])  # [1, 1, 2, 1]
    v = torch.tensor([[[[1.0], [2.0]]]])  # [1, 1, 2, 1]

    # Scores:
    # 0: 10.0
    # 1: -10.0
    # Unmasked Softmax: ~1.0 at index 0, ~0.0 at index 1.
    # Result should be 1.0 (Value at 0).

    # We want to MASK Index 0. So we force attention to Index 1.
    # Result should be 2.0.

    # Mask A: [True, False]
    mask_a = torch.tensor([[[[True, False]]]], dtype=torch.bool)

    # Mask B: [False, True]
    mask_b = torch.tensor([[[[False, True]]]], dtype=torch.bool)

    out_a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_a)
    out_b = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_b)

    print(f"Outcome A (Mask=[True, False]): {out_a.item()}")
    print(f"Outcome B (Mask=[False, True]): {out_b.item()}")

    if abs(out_a.item() - 2.0) < 0.1:
        print("A -> 2.0 means Index 0 was MASKED (Ignored). So True = Ignore.")
    elif abs(out_a.item() - 1.0) < 0.1:
        print("A -> 1.0 means Index 0 was KEPT (Attended). So True = Keep.")

    if abs(out_b.item() - 1.0) < 0.1:
        print("B -> 1.0 means Index 1 was MASKED. So True = Ignore.")


if __name__ == "__main__":
    test_mask()
