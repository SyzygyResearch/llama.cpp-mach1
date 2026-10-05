# llama.cpp-mach1

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that runs **Mach-1** models.
Get the model from [SyzygyResearch/Mach-1-Additive-35B-GGUF](https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B-GGUF).

## Quick start

Download a build from [Releases](https://github.com/SyzygyResearch/llama.cpp-mach1/releases) — pick the archive for your hardware (**NVIDIA → cuda, AMD/Intel → vulkan**), unpack it, and run from inside:

```sh
./llama-cli -hf SyzygyResearch/Mach-1-Additive-35B-GGUF
```

The archive carries the binaries and their libraries, and the model downloads on first run.

## Build from source

Pick the backend for your hardware:

```sh
git clone https://github.com/SyzygyResearch/llama.cpp-mach1
cd llama.cpp-mach1

# NVIDIA (requires the CUDA toolkit)
cmake -B build -DGGML_CUDA=ON
# AMD / Intel / Apple Silicon (requires the Vulkan SDK, incl. glslc)
cmake -B build -DGGML_VULKAN=ON

cmake --build build --config Release -j
```

## Run

```sh
# straight from Hugging Face
./build/bin/llama-cli -hf SyzygyResearch/Mach-1-Additive-35B-GGUF

# interactive chat with a local file
./build/bin/llama-cli -m Mach-1-Additive-35B.mach1.gguf

# single-turn / scripted use
./build/bin/llama-cli -m Mach-1-Additive-35B.mach1.gguf -st -p "your prompt"

# OpenAI-compatible server
./build/bin/llama-server -m Mach-1-Additive-35B.mach1.gguf
```

GPU offload is automatic in GPU builds (no `-ngl` flag needed).

## Vision

The multimodal variant pairs the same language GGUF with a projector file — get both from [SyzygyResearch/Mach-1-Additive-35B-Multimodal-GGUF](https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B-Multimodal-GGUF):

```sh
./build/bin/llama-mtmd-cli \
  -m Mach-1-Additive-35B.mach1.gguf \
  --mmproj mmproj-Mach-1-Additive-35B-f16.gguf \
  --image photo.jpg -p "Describe this image."
```

`llama-server` takes the same `--mmproj` flag and accepts images through the OpenAI-compatible `image_url` content part.

## Qwen3.8-Flash-Next

[SyzygyResearch/Mach-1-Additive-Qwen3.8-Flash-Next-GGUF](https://huggingface.co/SyzygyResearch/Mach-1-Additive-Qwen3.8-Flash-Next-GGUF) is Qwen3.8-Flash-Next at 1.7 bits per weight: one 130 GB file holding 26.7 GB of weights and the model's 102 GB n-gram embedding table. Build with CUDA, then:

```sh
./build/bin/llama-cli -hf SyzygyResearch/Mach-1-Additive-Qwen3.8-Flash-Next-GGUF -ngl 99 -fa on --mlock \
  --temp 1.0 --top-p 0.95 --top-k 20

./build/bin/llama-server -m Mach-1-Additive-Qwen3.8-Flash-Next.mach1.gguf -ngl 99 -fa on --mlock -c 32768 --jinja \
  --temp 1.0 --top-p 0.95 --top-k 20
```

- The weights go to the GPU, so it needs 32 GB of VRAM or more.
- The n-gram table stays in host RAM and is read a few rows per token, so the machine needs about 105 GB of free RAM.
- `--mlock` keeps the table resident. Without it the table is memory-mapped, and pages the OS drops are read back from disk during generation. Locking needs `ulimit -l unlimited` (or root).
- Sample at temperature 1.0, top_p 0.95, top_k 20, or the model can loop.
- The fast decode path is CUDA. AMD builds with HIP run the codec ops on the CPU.

## Notes

- Mach-1 checkpoints need these builds — stock llama.cpp cannot load them, and `llama-quantize` refuses them by design (the weights are already packed code streams).
- Codec details and the backend support matrix are in [docs/mach1.md](docs/mach1.md).
- Everything else works as in [upstream llama.cpp](https://github.com/ggml-org/llama.cpp); see its README for the full tool and server documentation.

## License

MIT, same as upstream llama.cpp.
