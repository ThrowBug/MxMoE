# Qwen3.5-35B-A3B text-only MxMoE

From the MxMoE repository root, run `bash scripts/Qwen3.5-35B-A3B/run_all.sh`.
This uses the same local C4 JSON file and 128 × 2048, seed-0 calibration recipe
as the Qwen3 scripts. `MODEL` can select a local copy of the same checkpoint.
The default stages are also individually runnable in their prescribed order:
`trace.sh`, `collect_loss.sh`, `allocate.sh`, `quantize.sh`.

Trace and loss collection use the original, unquantized model. During final
quantization, each decoder layer first replays its current inputs with original
layer weights to collect both fixed-linear GPTQ statistics and routed-expert
inputs. It then quantizes those weights and replays the quantized layer once to
advance the inputs to the next layer, matching the Qwen3 layerwise schedule.
Full attention `q/k/v/o`, linear attention `in_proj_qkv/out_proj`, and
shared-expert `gate/up/down` default to W4 GPTQ.
Set `ATTN_WBITS`, `LINEAR_ATTN_WBITS`, or `SHARED_EXPERT_WBITS` to `16` to
leave the corresponding group unquantized. The other linear-attention weights,
router, and shared-expert gate remain at model precision. These fixed settings
are part of every artifact's identity; keep them identical for all four stages.
The routed experts alone receive asymmetric W1/W2/W3 G128-A16 mixed-bit GPTQ.
`NOMINAL_BITS` defaults to 2.0 and the allocator optimizes accuracy only.

Override `TRACE_FILE`, `LOSS_DIR`, `QCONFIG`, or `OUTPUT` to place artifacts
elsewhere. The allocator checks that trace and loss artifacts share the model,
calibration data, and fixed-bit settings. Completed artifacts are not
overwritten. A failed
`collect_loss.sh` may leave `.partial` files; those are not consumed by the
allocator and are replaced on the next collection attempt. Saved model weights
are dequantized BF16 fake-quant weights, as in the existing Qwen3 workflow,
not packed low-bit runtime weights.

Each stage runs one decoder layer on the selected CUDA device at a time, while
the text-only model and intermediate activations live primarily in host memory.
Expect substantial host RAM and GPU time for a 35B model. The full 35B C4 run
must be smoke-tested on the target machine before treating its results as
validated; repository unit tests do not load the checkpoint.
