import io
import os
import modal

# Define the container image matching the setup.sh configuration and requirements.txt.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        "pip install torch==2.11.0+cu126 torchvision==0.26.0+cu126 --index-url https://download.pytorch.org/whl/cu126"
    )
    .pip_install(
        "accelerate==1.13.0",
        "diffusers==0.38.0",
        "einops==0.8.2",
        "huggingface_hub==1.14.0",
        "numpy==2.4.3",
        "pillow>=12.2.0",
        "safetensors>=0.7.0",
        "tokenizers==0.22.2",
        "tqdm==4.67.3",
        "transformers==5.8.0",
        "openai==2.36.0",
        "kernels==0.14.0",
    )
    .add_local_dir("./lens", "/root/lens")
)

# Initialize the Modal App
app = modal.App("lens-inference", image=image)

# Persistent volume for HuggingFace model cache to avoid re-downloading weights on every run
hf_volume = modal.Volume.from_name("hf-cache-vol", create_if_missing=True)


@app.function(
    gpu="A100",  # Hopper-or-newer is recommended for native MXFP4 dequantization support
    volumes={"/root/.cache/huggingface": hf_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def run_remote_inference(
    prompt: str,
    repo_id: str = "microsoft/Lens",
    base_resolution: int = 1440,
    aspect_ratio: str = "1:1",
    steps: int = 20,
    cfg: float = 5.0,
    n: int = 1,
    seed: int = None,
    dtype: str = "bfloat16",
    disable_mxfp4: bool = False,
    reasoner: bool = False,
    api_url: str = None,
    api_key: str = None,
    api_model: str = None,
    offload: bool = False,
) -> list[tuple[str, bytes]]:
    """Runs text-to-image inference on the remote Modal GPU container."""
    import torch
    from lens import LensGptOssEncoder, LensPipeline

    prompts = [p.strip() for p in prompt.split("|") if p.strip()]
    if not prompts:
        raise ValueError("No non-empty prompts after splitting on '|'.")

    # Map string dtype to torch dtype
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtypes.get(dtype, torch.bfloat16)

    # Pre-load text encoder to control MXFP4 dequantization
    text_encoder_kwargs = {"subfolder": "text_encoder", "dtype": torch_dtype}
    try:
        from transformers import Mxfp4Config
        text_encoder_kwargs["quantization_config"] = Mxfp4Config(
            dequantize=disable_mxfp4
        )
    except ImportError:
        pass

    print(f"Loading text encoder from {repo_id}...")
    text_encoder = LensGptOssEncoder.from_pretrained(
        repo_id, **text_encoder_kwargs
    )

    print(f"Loading Lens pipeline from {repo_id}...")
    pipe = LensPipeline.from_pretrained(
        repo_id, text_encoder=text_encoder, torch_dtype=torch_dtype
    )

    if offload:
        print("Enabling model CPU offload...")
        pipe.enable_model_cpu_offload()
    else:
        print("Moving pipeline to GPU...")
        pipe.to("cuda")

    if api_url or api_key or api_model:
        pipe.reasoner.openai_base_url = api_url
        pipe.reasoner.openai_api_key = api_key
        pipe.reasoner.openai_model = api_model

    generator = (
        torch.Generator(device=pipe._execution_device).manual_seed(seed)
        if seed is not None
        else None
    )

    print(f"Running generation for {len(prompts)} prompt(s) with steps={steps}, cfg={cfg}, n={n}...")
    out = pipe(
        prompt=prompts,
        base_resolution=base_resolution,
        aspect_ratio=aspect_ratio,
        num_inference_steps=steps,
        guidance_scale=cfg,
        num_images_per_prompt=n,
        generator=generator,
        enable_reasoner=reasoner,
    )

    images = list(out.images)
    expected_images = len(prompts) * n
    if len(images) != expected_images:
        raise RuntimeError(
            f"Pipeline returned {len(images)} images; expected {expected_images}."
        )

    # Save images to in-memory bytes to return back to the local host machine
    results = []
    img_iter = iter(images)
    for p_idx, p in enumerate(prompts):
        for s_idx in range(n):
            img = next(img_iter)
            fname = f"p{p_idx:03d}_s{s_idx:02d}.png"
            
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            
            results.append((fname, img_bytes))
            print(f"Generated {fname} :: {p!r}")

    refined = getattr(pipe, "_last_refined_prompts", prompts)
    if any(r != orig for r, orig in zip(refined, prompts)):
        print("\nRefined prompts:")
        for orig, ref in zip(prompts, refined):
            print(f"  {orig!r}\n    -> {ref!r}")

    return results


@app.local_entrypoint()
def main(
    prompt: str,
    repo_id: str = "microsoft/Lens",
    base_resolution: int = 1440,
    aspect_ratio: str = "1:1",
    steps: int = 20,
    cfg: float = 5.0,
    n: int = 1,
    seed: int = None,
    out: str = "./outputs",
    dtype: str = "bfloat16",
    disable_mxfp4: bool = False,
    reasoner: bool = False,
    api_url: str = None,
    api_key: str = None,
    api_model: str = None,
    offload: bool = False,
):
    """
    Trigger inference remotely on the Modal platform.
    Generated images are downloaded and saved locally in the 'out' directory.
    """
    print(f"Triggering remote inference on Modal for prompt: {prompt!r}")
    results = run_remote_inference.remote(
        prompt=prompt,
        repo_id=repo_id,
        base_resolution=base_resolution,
        aspect_ratio=aspect_ratio,
        steps=steps,
        cfg=cfg,
        n=n,
        seed=seed,
        dtype=dtype,
        disable_mxfp4=disable_mxfp4,
        reasoner=reasoner,
        api_url=api_url,
        api_key=api_key,
        api_model=api_model,
        offload=offload,
    )

    # Save received image bytes to the local output folder
    os.makedirs(out, exist_ok=True)
    for fname, img_bytes in results:
        local_path = os.path.join(out, fname)
        with open(local_path, "wb") as f:
            f.write(img_bytes)
        print(f"Saved generated image to {local_path}")
