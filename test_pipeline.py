"""
Sanity check script for LeadAttnResNet and AsymmetricLoss.
"""
import torch
from model import LeadAttnResNet
from loss import AsymmetricLoss


def main():
    print("Testing LeadAttnResNet...")
    device = "cpu"
    model = LeadAttnResNet(num_classes=6, d_model=128, nhead=4).to(device)

    # 1. Forward pass
    x = torch.randn(4, 12, 1000)
    logits, attn = model(x, return_attn=True)
    assert logits.shape == (4, 6), f"Expected [4, 6], got {logits.shape}"
    assert attn.shape == (4, 12, 12), f"Expected [4, 12, 12], got {attn.shape}"

    # 2. Loss computation
    y = torch.randint(0, 2, (4, 6)).float()
    loss_fn = AsymmetricLoss()
    loss = loss_fn(logits, y)
    assert not torch.isnan(loss), "Loss was NaN"
    loss.backward()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Sanity check passed! Model params: {n_params:,} ({n_params * 4 / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
