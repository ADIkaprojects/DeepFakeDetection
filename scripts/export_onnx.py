from __future__ import annotations

import argparse

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="models/atn.onnx")
    parser.add_argument("--input-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(model, dict) and "state_dict" in model:
        raise RuntimeError("Checkpoint contains state_dict only. Build model class before export.")

    model.eval()
    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    torch.onnx.export(model, dummy, args.output, opset_version=17)
    print(f"Exported ONNX model to {args.output}")


if __name__ == "__main__":
    main()
