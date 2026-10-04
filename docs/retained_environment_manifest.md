# Retained experimental-environment manifest

These metadata were captured retrospectively from the retained experimental
workstation and locally stored model, which were preserved unchanged after
data collection; they were not embedded in the original 21 August 2026 run
configuration.

The formal Qwen Full run `full_20260821T051022Z_d81f78bf/config.json` records
only the Ollama tag `qwen2.5:14b`. Absence of version / digest / quantization
fields in that file does not prevent reporting the retained environment; it
means those fields are **not run-embedded metadata**.

Capture time (UTC): `2026-10-04T13:42:43Z`

## Query results

| Item | Value | How obtained |
|---|---|---|
| OS | Ubuntu 20.04.6 LTS (focal) | `lsb_release -a`; `/etc/os-release` `PRETTY_NAME` |
| Python | 3.8.10 | `/usr/bin/python3.8 --version`; `/usr/bin/python3` → `python3.8` |
| ROS 2 | Foxy | `/opt/ros/foxy` present; `ros-foxy-ros-base` `0.9.2-1focal.20230606.035837`; `ros-foxy-rclpy` `1.0.13-1focal.20230527.051412` |
| Ollama client | 0.13.2 | `/opt/ollama/bin/ollama --version` (no running server at capture) |
| Planner tag | `qwen2.5:14b` | Ollama library manifest `registry.ollama.ai/library/qwen2.5/14b`; also the tag stored in the 21 August 2026 `config.json` |
| Quantization | Q4_K_M | GGUF `general.file_type = 15` (`MOSTLY_Q4_K_M`); `general.name = Qwen2.5 14B Instruct`; GGUF version 3 |
| Model manifest SHA256 | `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6` | `sha256sum` of `/opt/ollama/models/manifests/registry.ollama.ai/library/qwen2.5/14b` |
| Weight SHA256 | `2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54` | Manifest layer `application/vnd.ollama.image.model`; `sha256sum` of the 8 988 110 688-byte blob confirmed the same digest |

## Manifest layer (from the retained Ollama manifest)

- config digest: `sha256:db59b814cab753a51167db007fa6b6e0095f678ff24c9f7284753b75b34c6df3`
- model weights: `sha256:2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54` (8 988 110 688 bytes)
- system / template / license layers are present in the same manifest and were not used as the planner weight identity

## What this file is not

- It is not a substitute for fields that were never written into
  `full_20260821T051022Z_d81f78bf/config.json`.
- It does not redistribute model weights.
- It does not claim that Ollama 0.13.2 was recorded at planning time; it
  records the client version found on the unchanged workstation at capture.
