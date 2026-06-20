import argparse

import torch

from fm_minimal import TinyDiTVelocity, TinyUNetVelocity


def check_model(name: str, model: torch.nn.Module, t: torch.Tensor, x: torch.Tensor) -> None:
    model.zero_grad(set_to_none=True)
    output = model(t, x)
    if output.shape != x.shape:
        raise RuntimeError(f"{name}: expected output shape {tuple(x.shape)}, got {tuple(output.shape)}")

    loss = output.square().mean()
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not any(parameter.grad is not None for parameter in trainable):
        raise RuntimeError(f"{name}: no gradients were produced")
    if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in trainable):
        raise RuntimeError(f"{name}: non-finite gradients were produced")

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"{name}: input={tuple(x.shape)} output={tuple(output.shape)} "
        f"loss={loss.item():.6f} parameters={parameters} gradients=ok"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x = torch.randn(
        args.batch,
        args.channels,
        args.image_size,
        args.image_size,
        device=args.device,
    )
    t = torch.rand(args.batch, 1, device=args.device)

    unet = TinyUNetVelocity(in_channels=args.channels).to(args.device)
    dit = TinyDiTVelocity(
        image_size=(args.image_size, args.image_size),
        in_channels=args.channels,
        patch_size=args.patch_size,
    ).to(args.device)

    check_model("TinyUNetVelocity", unet, t, x)
    check_model("TinyDiTVelocity", dit, t, x)


if __name__ == "__main__":
    main()
