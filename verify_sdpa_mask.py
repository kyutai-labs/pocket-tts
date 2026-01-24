import torch
import torch.nn.functional as F


def test_sdpa_mask():
    print(f"PyTorch Version: {torch.__version__}")

    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 2, 8)
    v = torch.randn(1, 1, 2, 8)

    # Mask: 1st element True, 2nd element False
    # If True = Keep, output should depend mostly on 1st element (if weights are high) or just be valid.
    # If True = Ignore, output should be nan or depend on 2nd element.

    # Let's try a clearer test.
    # q . k -> scores.
    # Masking.

    # Case 1: Boolean Mask [1, 1, 1, 2]
    mask_true_false = torch.tensor([[[[True, False]]]], dtype=torch.bool)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_true_false)

    print("Mask: [True, False]")
    # Check if we attended to the second element (False)
    # If True=Keep, we attended to Index 0. Index 1 should have -inf score.
    # If True=Ignore, we attended to Index 1. Index 0 should have -inf score.

    # We can inspect this by providing manual q, k to enforce specific scores.
    # But checking if `out` is non-nan is a start.

    # Let's try with additive mask simulation
    # If I pass float mask of 0 and -inf.

    # Let's just create a scenario where one is clearly masked.
    q = torch.ones(1, 1, 1, 4)
    k = torch.cat([torch.ones(1, 1, 1, 4) * 10, torch.ones(1, 1, 1, 4) * -10], dim=2)
    v = torch.cat([torch.ones(1, 1, 1, 4) * 1, torch.ones(1, 1, 1, 4) * 2], dim=2)

    # k values: Index 0 is highly similar (score 40), Index 1 is dissimilar (-40).
    # Softmax will peak at Index 0. Output should be close to 1.

    # Mask Index 0 (True).
    mask = torch.tensor([[[[True, False]]]], dtype=torch.bool)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    print(f"Output with Mask [True, False]: {out.mean().item()}")

    if abs(out.mean().item() - 1.0) < 0.1:
        print("Conclusion: True means KEEP (Attend)")
    elif abs(out.mean().item() - 2.0) < 0.1:
        print("Conclusion: True means IGNORE (Mask out)")
    else:
        print("Conclusion: Ambiguous")


if __name__ == "__main__":
    test_sdpa_mask()
