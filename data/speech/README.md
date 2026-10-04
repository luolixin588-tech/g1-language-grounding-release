# Speech / STT runtime logs

Copied (not moved) from the retained workstation ROS logs that cover the
2026-08-24 / 2026-08-25 physical-trial window. Logger: `g1_audio_bridge`.

These are raw ear-node lines, not a saved Extended Data Table 2
aggregation. The manuscript Table 2 means (STT 160.58 ms, E2E 851.10 ms)
are **not** stored as a table in this directory.

## Files

| File | Original source | Session start (local) | Latency lines | Recognized lines |
|---|---|---|---:|---:|
| `raw_log_01.txt` | `~/.ros/log/python_3417980_1787580797685.log` | 2026-08-24 22:13 | 75 | 55 |
| `raw_log_02.txt` | `~/.ros/log/python_4155467_1787644803083.log` | 2026-08-25 16:00 | 92 | 53 |

In each public copy, one local install path was rewritten:

`/home/luolixin/unitree_ros2_ws/install/my_g1_audio` → `<local-install-dir>`

Original ROS files were not modified.

## Latency fields

Each recognition attempt logs:

```
Latency: stt_ms=..., endpoint_ms=..., e2e_ms=...
```

- `stt_ms`: Faster-Whisper wall-clock
- `endpoint_ms`: VAD endpoint wait
- `e2e_ms`: last speech frame to text ready

Some attempts have a latency line without a `Recognized(...)` line (empty
or discarded transcript). Descriptive means of the released lines:

| File | STT mean / median (ms) | E2E mean / median (ms) |
|---|---|---|
| `raw_log_01.txt` | 123.58 / 74.60 | 802.55 / 739.00 |
| `raw_log_02.txt` | 146.75 / 81.25 | 807.67 / 733.35 |

Do not treat these two-session means as Table 2.

The controller-side physical log `data/physical_trials/rawdate.log` can
overlap this window; it is not a second independent STT dataset.
