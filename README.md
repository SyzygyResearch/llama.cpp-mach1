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

## Notes

- Mach-1 checkpoints need these builds — stock llama.cpp cannot load them, and `llama-quantize` refuses them by design (the weights are already packed code streams).
- Codec details and the backend support matrix are in [docs/mach1.md](docs/mach1.md).
- Everything else works as in [upstream llama.cpp](https://github.com/ggml-org/llama.cpp); see its README for the full tool and server documentation.

## License

MIT, same as upstream llama.cpp.
