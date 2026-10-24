# Checkpoints

Each scenario contains only the best inference checkpoints:

- `vae_best.pt`: STFT-VAE weights and model configuration.
- `tse_best.pt`: condition-guided latent diffusion weights and diffusion configuration.

Optimizer states, intermediate epochs, training logs, and ablation checkpoints are excluded.
The `.pt` files are large and must be managed with Git LFS or attached to a GitHub Release.
