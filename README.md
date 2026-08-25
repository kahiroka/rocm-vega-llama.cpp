# ROCm 7.2 + llama.cpp on mixed AMD Vega GPUs (gfx900 + gfx906)

This repo documents a reproducible Docker setup for running current `llama.cpp` on a mixed AMD Vega system with both:

- Radeon RX Vega 64 / Vega10 / `gfx900`
- Radeon Instinct MI50 (Radeon VII-class Vega20) / `gfx906`

The goal was not just to make ROCm *enumerate* these unsupported GPUs, but to get current ROCm 7.2.1 + rocBLAS + llama.cpp to execute on both architectures in the same process.

## Why this exists

Modern ROCm releases can still see older Vega GPUs through the HIP runtime, but current prebuilt rocBLAS/Tensile payloads no longer include all legacy architectures. In practice, that means:

- `rocminfo` may show `gfx900` and `gfx906`
- HIP compilation may work
- `llama.cpp --list-devices` may work
- but first real GEMM/inference can fail because the matching rocBLAS/Tensile kernels are absent

This setup combines:

1. a ROCm 7.2.1 image patched for `gfx906`
2. `gfx900` rocBLAS/Tensile payloads copied from ROCm 6.3.4
3. a `llama.cpp` build targeting both `gfx900;gfx906`

## Tested hardware

The initial test machine used:

- AMD Ryzen 5 3600
- 64 GB DDR4
- Radeon RX Vega 64 8 GB HBM2 (`gfx900`)
- AMD MI50 16 GB HBM2, presenting as Radeon VII (`gfx906`)
- Linux host with Docker and access to `/dev/kfd` and `/dev/dri`

The interesting result was that a mixed Vega64 + MI50 setup could run Gemma 4 26B-A4B under ROCm 7.2.1 once both rocBLAS payloads were present.

## Important caveat

These GPUs are **not officially supported by ROCm 7.2**. This is an experimental compatibility setup. Use it at your own risk.

## 1. Build the mixed ROCm image

Create `Dockerfile.rocm72-gfx900-gfx906`:

```dockerfile
# Stage 1: obtain legacy gfx900 rocBLAS/Tensile payloads
FROM rocm/dev-ubuntu-22.04:6.3.4 AS rocm6-libs

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends rocblas && \
    rm -rf /var/lib/apt/lists/*

# Stage 2: start with a ROCm 7.2.1 image that already restores gfx906
FROM mixa3607/rocm-gfx906:7.2.1-complete

COPY --from=rocm6-libs \
    /opt/rocm/lib/rocblas/library/ \
    /tmp/rocm6-rocblas/

RUN set -eux; \
    ROCBLAS_LIB="/opt/rocm/lib/rocblas/library"; \
    echo "=== Existing gfx906 files ==="; \
    find "$ROCBLAS_LIB" -maxdepth 1 -type f | grep gfx906 | head || true; \
    echo "=== Adding gfx900 files from ROCm 6.3.4 ==="; \
    find /tmp/rocm6-rocblas -maxdepth 1 -type f -name '*gfx900*' \
      -exec cp -v {} "$ROCBLAS_LIB/" \;; \
    echo "=== Result ==="; \
    echo "gfx900 files:"; \
    find "$ROCBLAS_LIB" -maxdepth 1 -type f | grep gfx900 | wc -l; \
    echo "gfx906 files:"; \
    find "$ROCBLAS_LIB" -maxdepth 1 -type f | grep gfx906 | wc -l; \
    rm -rf /tmp/rocm6-rocblas

WORKDIR /workspace
CMD ["/bin/bash"]
```

Build it:

```bash
sudo docker build \
  -t rocm72-donnager-gfx900-gfx906 \
  -f Dockerfile.rocm72-gfx900-gfx906 \
  .
```

On the tested build, the final image contained roughly:

```text
gfx900: 128 files
gfx906: 156 files
```

with real `.co`, `.hsaco`, and `.dat` rocBLAS/Tensile payloads present.

## 2. Verify both GPUs enumerate

Run the image with both GPU device nodes:

```bash
sudo docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  rocm72-donnager-gfx900-gfx906 \
  bash
```

Inside:

```bash
rocminfo | grep -E 'Name:.*gfx(900|906)'
```

Expected shape:

```text
Name: gfx900
Name: gfx906
```

## 3. Prove rocBLAS actually executes on gfx900

Device detection alone is not enough. We used a tiny SGEMM test and verified that rocBLAS executed on device 0 (Vega64 / gfx900):

```text
HIP devices: 2
device 0: AMD Radeon RX Vega  arch=gfx900:xnack-
device 1: AMD Radeon VII      arch=gfx906:sramecc+:xnack-
Calling rocblas_sgemm on device 0...
rocblas_sgemm returned: 0
hipDeviceSynchronize: no error
```

A small test program is included in this repo as `test-rocblas.cpp`.

## 4. Build llama.cpp for both gfx900 and gfx906

For persistence, mount your host llama.cpp tree into the container rather than cloning into a disposable `--rm` container.

Example:

```bash
sudo docker run --rm -it \
  -p 8080:8080 \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  -v /home/$USER/Development/llama.cpp:/workspace/llama.cpp \
  -v /home/$USER/Downloads:/models \
  rocm72-donnager-gfx900-gfx906 \
  bash
```

Inside:

```bash
cd /workspace/llama.cpp

cmake -B build-rocm72-dual \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx900;gfx906" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON

cmake --build build-rocm72-dual \
  --target llama-server \
  -j$(nproc)
```

Verify:

```bash
./build-rocm72-dual/bin/llama-server --list-devices
```

Expected:

```text
ROCm0: AMD Radeon RX Vega (8176 MiB, ...)
ROCm1: AMD Radeon VII (16368 MiB, ...)
```

## 5. Mixed-GPU llama.cpp example

Tested with Gemma 4 26B-A4B Q4_K_M:

```bash
./build-rocm72-dual/bin/llama-server \
  -m /models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --device ROCm0,ROCm1 \
  -ngl 999 \
  --split-mode layer \
  --tensor-split 1,4 \
  -c 8192 \
  -np 1 \
  -b 512 \
  -ub 256 \
  --host 0.0.0.0 \
  --port 8080
```

### Why `1,4` mattered

A `--tensor-split 1,2` overloaded the 8 GB Vega64 and caused severe multi-second stalls. The browser could become unresponsive while generation paused, then inference would resume.

Changing to:

```text
--tensor-split 1,4
```

left more headroom on the Vega64 and produced much smoother behavior.

Observed behavior with the stable `1,4` split:

- ~57.6 tok/s at low context
- ~56.5 tok/s around 3.6K context
- ~55.5 tok/s around 4.4K context
- ~54.3 tok/s around 6.6K context
- ~53.8 tok/s near 8K context
- ~52.8 tok/s at 8191/8192 tokens
- no repeated long stalls seen in the stable run

The exact numbers are model/build/hardware dependent, but the important lesson was that **heterogeneous GPUs should not be split purely by VRAM capacity**. The slower/smaller Vega64 worked much better as a minority participant.

## 6. Single MI50 comparison

The same model on the single gfx906 MI50 under the patched ROCm 7.2.1 environment was also healthy:

- roughly 58-59 tok/s at low context
- roughly 52-53 tok/s near a full 8K context

This was dramatically better than the Vulkan path on this particular model/workload.

## 7. Reproduction notes

- Keep the custom ROCm image separate from the host ROCm installation.
- Do not overwrite host rocBLAS/Tensile files while experimenting.
- Use bind mounts for source/build output and models so `docker run --rm` remains safe.
- `HIP_VISIBLE_DEVICES=1` is useful for isolating the MI50/gfx906 in single-GPU tests.
- Avoid copying environment variables intended for Vega APUs such as `HSA_OVERRIDE_GFX_VERSION` unless you specifically need them. A real gfx900 Vega64 and gfx906 MI50 enumerate correctly without those overrides in this setup.

## Credits / prior work

This setup builds on community work that restored older Vega rocBLAS/Tensile payloads to newer ROCm releases, including:

- `mixa3607/rocm-gfx906:7.2.1-complete`
- `daimonionnn/amd-vega-rocm-vulkan-llm-toolkit`
- community ROCm 7.2 / Frigate Vega backport experiments

The combined gfx900 + gfx906 image and the mixed-GPU `llama.cpp` validation documented here were assembled and tested on real Vega64 + MI50 hardware.

## Search terms

For people trying to find this later:

`ROCm 7.2 gfx900`, `ROCm 7.2 gfx906`, `Vega 64 llama.cpp`, `MI50 llama.cpp`, `Radeon VII ROCm`, `mixed AMD GPU llama.cpp`, `Vega10 Vega20 ROCm`, `rocBLAS gfx900`, `rocBLAS gfx906`
