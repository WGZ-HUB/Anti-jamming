"""
Inference for condition-guided latent diffusion radar anti-jamming.
by Wang Guozheng
2026.10.20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import torch
from diffusers import DDIMScheduler


ROOT = Path(__file__).resolve().parent
VAE_ROOT = ROOT / "VAE"
TSE_ROOT = ROOT / "TSE"
for module_root in (VAE_ROOT, TSE_ROOT):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from vae.checkpoint import load_checkpoint  # noqa: E402
from vae.data import complex_to_two_channels, pad_or_crop  # noqa: E402
from model.tse.conditioners import TSE  # noqa: E402


SCENARIOS = (
    "ISRJ",
    "NFMJ",
    "ISRJ_NFMJ",
    "FALSE",
    "EXTENDED_COMPOUND",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover a complex radar target echo with the proposed model."
    )
    parser.add_argument("--scenario", choices=SCENARIOS, default="EXTENDED_COMPOUND")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="MAT file containing echo and ref; sig is optional.",
    )
    parser.add_argument("--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--tse-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--ddim-steps", type=int, default=1)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--preserve-scale",
        action="store_true",
        help="Do not apply the dataset's unit-peak output convention.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def load_tse(checkpoint_path: Path, device: torch.device):
    checkpoint = _torch_load(checkpoint_path)
    config = checkpoint["diff_config"]
    model = TSE(
        udit_model=dict(config["diffwrap"]["UDiT"]),
        vit_model=dict(config["diffwrap"]["ViT"]),
    )
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict"))
    if state_dict is None:
        raise KeyError("TSE checkpoint does not contain model weights.")
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, config


def as_matrix(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2:
        raise ValueError(f"'{name}' must be [N, L], got {value.shape}.")
    return value


def load_input(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    data = scipy.io.loadmat(path)
    for required in ("echo", "ref"):
        if required not in data:
            raise KeyError(f"'{required}' is missing from {path}.")
    echo = as_matrix(data["echo"], "echo")
    reference = as_matrix(data["ref"], "ref")
    target = as_matrix(data["sig"], "sig") if "sig" in data else None
    return echo, reference, target


def to_model_input(
    signal: np.ndarray,
    index: int,
    sample_size: int,
    device: torch.device,
) -> torch.Tensor:
    value = complex_to_two_channels(signal[index])
    value = pad_or_crop(value, sample_size=sample_size, random_crop=False)
    return value.unsqueeze(0).to(device)


def to_complex(signal: torch.Tensor) -> np.ndarray:
    value = signal.detach().cpu().squeeze(0)
    return value[0].numpy() + 1j * value[1].numpy()


@torch.inference_mode()
def sample_latent(
    model: torch.nn.Module,
    scheduler: DDIMScheduler,
    mixture_latent: torch.Tensor,
    reference_latent: torch.Tensor,
    steps: int,
    eta: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    mixture = mixture_latent.transpose(1, 2).contiguous()
    reference = reference_latent.transpose(1, 2).contiguous()
    lengths = torch.tensor([mixture.shape[1]], device=device)
    reference_lengths = torch.tensor([reference.shape[1]], device=device)

    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler.set_timesteps(steps, device=device)
    prediction = torch.randn(
        mixture.shape, generator=generator, device=device, dtype=mixture.dtype
    )

    for timestep in scheduler.timesteps:
        model_input = scheduler.scale_model_input(prediction, timestep)
        velocity, _ = model(
            x=model_input,
            timesteps=timestep,
            mixture=mixture,
            reference=reference,
            x_len=lengths,
            ref_len=reference_lengths,
        )
        prediction = scheduler.step(
            model_output=velocity,
            timestep=timestep,
            sample=prediction,
            eta=eta,
            generator=generator,
        ).prev_sample

    return prediction.transpose(1, 2).contiguous()


def evaluate(
    echo: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, float]:
    length = min(len(echo), len(target), len(prediction))
    echo, target, prediction = echo[:length], target[:length], prediction[:length]
    target_energy = np.sum(np.abs(target) ** 2)
    input_error = np.sum(np.abs(echo - target) ** 2)
    output_error = np.sum(np.abs(prediction - target) ** 2)
    input_sinr = 10.0 * np.log10((target_energy + eps) / (input_error + eps))
    output_sinr = 10.0 * np.log10((target_energy + eps) / (output_error + eps))
    rmse = np.sqrt(np.mean(np.abs(prediction - target) ** 2))
    target_zero_mean = target - np.mean(target)
    prediction_zero_mean = prediction - np.mean(prediction)
    corr = np.abs(np.vdot(target_zero_mean, prediction_zero_mean)) / (
        np.linalg.norm(target_zero_mean) * np.linalg.norm(prediction_zero_mean) + eps
    )
    return {
        "input_sinr_db": float(input_sinr),
        "output_sinr_db": float(output_sinr),
        "sinrif_db": float(output_sinr - input_sinr),
        "rmse": float(rmse),
        "corr": float(corr),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    input_path = args.input or ROOT / "examples" / args.scenario / "sample.mat"
    checkpoint_dir = ROOT / "checkpoints" / args.scenario
    vae_path = args.vae_checkpoint or checkpoint_dir / "vae_best.pt"
    tse_path = args.tse_checkpoint or checkpoint_dir / "tse_best.pt"
    output_dir = args.output_dir / args.scenario
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (input_path, vae_path, tse_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    vae, vae_checkpoint = load_checkpoint(vae_path, map_location=device)
    vae.to(device).eval()
    tse, diffusion_config = load_tse(tse_path, device)
    scheduler = DDIMScheduler(**diffusion_config["ddim"]["diffusers"])

    echo_matrix, ref_matrix, sig_matrix = load_input(input_path)
    available = min(len(echo_matrix), len(ref_matrix))
    if args.index < 0 or args.index >= available:
        raise IndexError(f"index={args.index}, available={available}")

    sample_size = int(vae_checkpoint["model_config"]["sample_size"])
    echo_tensor = to_model_input(echo_matrix, args.index, sample_size, device)
    ref_tensor = to_model_input(ref_matrix, args.index, sample_size, device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        echo_latent, _ = vae.encode(echo_tensor, deterministic=True)
        ref_latent, ref_std = vae.encode(ref_tensor, deterministic=True)
        pred_latent = sample_latent(
            tse,
            scheduler,
            echo_latent,
            ref_latent,
            args.ddim_steps,
            args.eta,
            args.seed,
            device,
        )
        # Restore the known transmitted-waveform scale without using the label.
        pred_tensor = vae.decode(pred_latent, signal_std=ref_std)
        if not args.preserve_scale:
            magnitude = torch.sqrt(
                pred_tensor[:, 0].square() + pred_tensor[:, 1].square()
            )
            peak = magnitude.amax(dim=-1, keepdim=True).clamp_min(1e-12)
            pred_tensor = pred_tensor / peak.unsqueeze(1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    echo = to_complex(echo_tensor)
    reference = to_complex(ref_tensor)
    prediction = to_complex(pred_tensor)
    result: dict[str, Any] = {
        "scenario": args.scenario,
        "sample_index": args.index,
        "ddim_steps": args.ddim_steps,
        "seed": args.seed,
        "device": str(device),
        "peak_normalized": not args.preserve_scale,
        "inference_time_ms": elapsed_ms,
    }

    output_payload: dict[str, Any] = {
        "echo": echo,
        "ref": reference,
        "pred_sig": prediction,
    }
    if sig_matrix is not None:
        target_tensor = to_model_input(sig_matrix, args.index, sample_size, device)
        target = to_complex(target_tensor)
        output_payload["sig"] = target
        result.update(evaluate(echo, target, prediction))

    scipy.io.savemat(output_dir / "prediction.mat", output_payload)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved: {output_dir / 'prediction.mat'}")


if __name__ == "__main__":
    main()
