# Calibration data

Copy the exact C4 shard used by GEMQ into this directory before running the
Qwen3 workflow:

```text
c4-train.00000-of-01024.json
```

The Qwen3 scripts default to C4, 128 blocks, sequence length 2048, and seed 0.
The generated checkpoint metadata records a SHA-256 digest of the resulting
token IDs so a GEMQ run can be checked against the same samples.
