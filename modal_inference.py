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
        "fastapi==0.115.0",
        "pydantic==2.9.2",
    )
    .add_local_dir("./lens", "/root/lens")
)

# Initialize the Modal App
app = modal.App("lens-inference", image=image)

# Persistent volume for HuggingFace model cache to avoid re-downloading weights on every run
hf_volume = modal.Volume.from_name("hf-cache-vol", create_if_missing=True)

# Persistent volume for generated output images
outputs_volume = modal.Volume.from_name("lens-outputs-vol", create_if_missing=True)

# Shared dictionary to store task progress in real-time across containers
progress_dict = modal.Dict.from_name("lens-progress-dict", create_if_missing=True)


@app.function(
    gpu="A10G",  # Hopper-or-newer is recommended for native MXFP4 dequantization support
    volumes={
        "/root/.cache/huggingface": hf_volume,
        "/outputs": outputs_volume,
    },
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
    out: str = "/outputs",
    task_id: str = None,
) -> list[str]:
    """Runs text-to-image inference on the remote Modal GPU container."""
    import torch
    from lens import LensGptOssEncoder, LensPipeline

    if task_id:
        progress_dict[task_id] = {
            "status": "starting",
            "progress": 0,
            "step": 0,
            "total_steps": steps,
        }

    try:
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

        if task_id:
            progress_dict[task_id] = {
                "status": "loading",
                "progress": 5,
                "step": 0,
                "total_steps": steps,
            }

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

        # Step callback to write real-time progress to modal.Dict
        def step_callback(pipe, step_idx, timestep, cb_kwargs):
            if task_id:
                # scale progress from 10% to 95% during denoising
                progress = 10 + int((step_idx + 1) / steps * 85)
                progress_dict[task_id] = {
                    "status": "generating",
                    "step": step_idx + 1,
                    "total_steps": steps,
                    "progress": progress,
                }
            return cb_kwargs

        print(f"Running generation for {len(prompts)} prompt(s) with steps={steps}, cfg={cfg}, n={n}...")
        pipeline_output = pipe(
            prompt=prompts,
            base_resolution=base_resolution,
            aspect_ratio=aspect_ratio,
            num_inference_steps=steps,
            guidance_scale=cfg,
            num_images_per_prompt=n,
            generator=generator,
            enable_reasoner=reasoner,
            callback_on_step_end=step_callback,
        )

        images = list(pipeline_output.images)
        expected_images = len(prompts) * n
        if len(images) != expected_images:
            raise RuntimeError(
                f"Pipeline returned {len(images)} images; expected {expected_images}."
            )

        if task_id:
            progress_dict[task_id] = {
                "status": "saving",
                "progress": 96,
                "step": steps,
                "total_steps": steps,
            }

        # Save images directly to the mounted Volume directory
        saved_paths = []
        os.makedirs(out, exist_ok=True)
        img_iter = iter(images)
        for p_idx, p in enumerate(prompts):
            for s_idx in range(n):
                img = next(img_iter)
                fname = f"p{p_idx:03d}_s{s_idx:02d}.png"
                if task_id:
                    fname = f"{task_id}_{fname}"
                dest_path = os.path.join(out, fname)
                img.save(dest_path)
                saved_paths.append(dest_path)
                print(f"Saved {fname} to Volume path {dest_path} :: {p!r}")

        # Commit changes to ensure they are visible on the volume
        outputs_volume.commit()

        refined = getattr(pipe, "_last_refined_prompts", prompts)
        if any(r != orig for r, orig in zip(refined, prompts)):
            print("\nRefined prompts:")
            for orig, ref in zip(prompts, refined):
                print(f"  {orig!r}\n    -> {ref!r}")

        if task_id:
            progress_dict[task_id] = {
                "status": "completed",
                "progress": 100,
                "images": [os.path.basename(sp) for sp in saved_paths],
                "refined_prompts": refined,
            }

        return saved_paths

    except Exception as e:
        import traceback
        traceback.print_exc()
        if task_id:
            progress_dict[task_id] = {
                "status": "failed",
                "progress": 0,
                "error": str(e),
            }
        raise e


# ==============================================================================
# FastAPI Web Application Definition
# ==============================================================================

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lens T2I Studio</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #08090f;
            --panel-bg: rgba(17, 19, 31, 0.65);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-primary: #8b5cf6;
            --accent-secondary: #db2777;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-gradient: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        body {
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(139, 92, 246, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(219, 39, 119, 0.12) 0%, transparent 40%);
            background-attachment: fixed;
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            padding: 1.5rem 2rem;
            border-bottom: 1px solid var(--border-color);
            backdrop-filter: blur(12px);
            background: rgba(8, 9, 15, 0.5);
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo-container {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-symbol {
            width: 2rem;
            height: 2rem;
            background: var(--accent-gradient);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.2rem;
            box-shadow: 0 0 15px rgba(139, 92, 246, 0.4);
        }

        .logo-text {
            font-size: 1.25rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            background: linear-gradient(to right, #fff, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .badge {
            background: rgba(139, 92, 246, 0.15);
            border: 1px solid rgba(139, 92, 246, 0.3);
            padding: 0.25rem 0.75rem;
            border-radius: 99px;
            font-size: 0.75rem;
            font-weight: 500;
            color: #c084fc;
        }

        main {
            flex: 1;
            max-width: 1400px;
            width: 100%;
            margin: 0 auto;
            padding: 2rem;
            display: grid;
            grid-template-columns: 450px 1fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .panel {
            background: var(--panel-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25);
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            height: fit-content;
        }

        .panel-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        textarea {
            width: 100%;
            min-height: 120px;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            resize: vertical;
            outline: none;
            transition: all 0.2s ease;
        }

        textarea:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 10px rgba(139, 92, 246, 0.15);
        }

        .collapsible-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            padding: 0.75rem 0;
            border-top: 1px solid var(--border-color);
            margin-top: 0.5rem;
        }

        .collapsible-content {
            display: none;
            flex-direction: column;
            gap: 1.25rem;
            padding-top: 0.5rem;
        }

        .collapsible-content.open {
            display: flex;
        }

        .chevron {
            transition: transform 0.2s ease;
        }

        .chevron.open {
            transform: rotate(180deg);
        }

        select, input[type="text"], input[type="number"] {
            width: 100%;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.9rem;
            outline: none;
            transition: all 0.2s ease;
        }

        select:focus, input:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 10px rgba(139, 92, 246, 0.15);
        }

        .slider-container {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .slider-container input[type="range"] {
            flex: 1;
            accent-color: var(--accent-primary);
        }

        .slider-val {
            min-width: 3.5rem;
            text-align: right;
            font-size: 0.9rem;
            font-weight: 600;
            color: #c084fc;
        }

        .toggle-group {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(0, 0, 0, 0.12);
            padding: 0.75rem 1rem;
            border-radius: 10px;
            border: 1px solid var(--border-color);
        }

        .switch {
            position: relative;
            display: inline-block;
            width: 44px;
            height: 24px;
        }

        .switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .slider {
            position: absolute;
            cursor: pointer;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: rgba(255, 255, 255, 0.15);
            transition: .3s;
            border-radius: 24px;
        }

        .slider:before {
            position: absolute;
            content: "";
            height: 18px;
            width: 18px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .3s;
            border-radius: 50%;
        }

        input:checked + .slider {
            background: var(--accent-gradient);
        }

        input:checked + .slider:before {
            transform: translateX(20px);
        }

        button {
            width: 100%;
            background: var(--accent-gradient);
            border: none;
            border-radius: 12px;
            padding: 1rem;
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(139, 92, 246, 0.5);
        }

        button:active {
            transform: translateY(1px);
        }

        button:disabled {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-muted);
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        /* Right column - View & Progress */
        .workspace {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            min-height: 600px;
        }

        .display-card {
            background: var(--panel-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25);
            position: relative;
            overflow: hidden;
            min-height: 500px;
        }

        .placeholder-view {
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
            color: var(--text-muted);
            max-width: 400px;
        }

        .placeholder-icon {
            width: 4rem;
            height: 4rem;
            border-radius: 50%;
            background: rgba(139, 92, 246, 0.1);
            border: 1px dashed rgba(139, 92, 246, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: var(--accent-primary);
        }

        .status-container {
            width: 100%;
            max-width: 450px;
            display: none;
            flex-direction: column;
            gap: 1.25rem;
            text-align: center;
        }

        .spinner {
            width: 3.5rem;
            height: 3.5rem;
            border: 3px solid rgba(139, 92, 246, 0.1);
            border-radius: 50%;
            border-top-color: var(--accent-primary);
            animation: spin 1s ease-in-out infinite;
            margin: 0 auto 0.5rem auto;
            position: relative;
        }

        .spinner::before {
            content: '';
            position: absolute;
            top: -6px; left: -6px; right: -6px; bottom: -6px;
            border: 1px solid rgba(219, 39, 119, 0.2);
            border-radius: 50%;
            animation: spin 3s linear infinite reverse;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .status-text {
            font-size: 1.1rem;
            font-weight: 500;
            letter-spacing: 0.5px;
        }

        .status-sub {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .progress-track {
            width: 100%;
            height: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 99px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .progress-bar {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #8b5cf6, #ec4899, #8b5cf6);
            background-size: 200% 200%;
            animation: progress-flow 2s linear infinite;
            border-radius: 99px;
            transition: width 0.4s ease;
        }

        @keyframes progress-flow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .image-viewer {
            display: none;
            width: 100%;
            height: 100%;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 1.5rem;
            position: relative;
        }

        .image-frame {
            border: 1px solid var(--border-color);
            background: rgba(0, 0, 0, 0.4);
            border-radius: 16px;
            padding: 0.5rem;
            max-width: 100%;
            max-height: 550px;
            overflow: hidden;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
        }

        .image-frame img {
            max-width: 100%;
            max-height: 520px;
            border-radius: 12px;
            object-fit: contain;
            display: block;
            opacity: 0;
            transform: scale(0.98);
            transition: all 0.5s ease-out;
        }

        .image-frame img.loaded {
            opacity: 1;
            transform: scale(1);
        }

        .image-actions {
            display: flex;
            gap: 1rem;
        }

        .btn-sec {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 500;
            text-decoration: none;
            box-shadow: none;
            width: auto;
        }

        .btn-sec:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateY(-1px);
            box-shadow: none;
        }

        .error-message {
            display: none;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #ef4444;
            padding: 1rem;
            border-radius: 12px;
            font-size: 0.9rem;
            text-align: center;
            max-width: 450px;
            width: 100%;
        }

        .refined-box {
            display: none;
            background: rgba(139, 92, 246, 0.05);
            border: 1px solid rgba(139, 92, 246, 0.15);
            border-radius: 12px;
            padding: 1rem;
            font-size: 0.85rem;
            color: #c084fc;
            width: 100%;
        }

        .refined-box span {
            font-weight: 600;
            color: var(--text-muted);
            display: block;
            margin-bottom: 0.25rem;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-container">
            <div class="logo-symbol">L</div>
            <div class="logo-text">Lens Studio</div>
            <span class="badge">Cloud T2I</span>
        </div>
        <div style="font-size: 0.85rem; color: var(--text-muted); display: flex; align-items: center; gap: 0.5rem;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 10px #10b981;"></div>
            Active on A100 GPU
        </div>
    </header>

    <main>
        <!-- Left Side Controls Panel -->
        <div class="panel">
            <div class="panel-title">
                <span>🪄</span> Generation Settings
            </div>

            <div class="input-group">
                <label for="prompt">Prompt</label>
                <textarea id="prompt" placeholder="A futuristic cyberpunk library filled with holograms and dynamic floating monitors, cinematic lighting, 8k resolution..."></textarea>
            </div>

            <div class="input-group">
                <label for="repo_id">Model Repository</label>
                <select id="repo_id">
                    <option value="microsoft/Lens">microsoft/Lens (Base Model)</option>
                    <option value="microsoft/Lens-Turbo">microsoft/Lens-Turbo (Fast 4-Step)</option>
                </select>
            </div>

            <div class="collapsible-header" onclick="toggleAdvanced()">
                <span style="font-size: 0.9rem; font-weight: 600; color: var(--text-muted); display: flex; align-items: center; gap: 0.5rem;">
                    <span>⚙️</span> Advanced Options
                </span>
                <span id="chevron" class="chevron">▼</span>
            </div>

            <div id="advanced-content" class="collapsible-content">
                <div class="input-group">
                    <label>Aspect Ratio</label>
                    <select id="aspect_ratio">
                        <option value="1:1">1:1 (Square)</option>
                        <option value="16:9">16:9 (Landscape)</option>
                        <option value="9:16">9:16 (Portrait)</option>
                        <option value="4:3">4:3 (Photo)</option>
                        <option value="3:4">3:4 (Tall)</option>
                        <option value="2:1">2:1 (Panoramic)</option>
                        <option value="1:2">1:2 (Vertical)</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Base Resolution</label>
                    <select id="base_resolution">
                        <option value="1440">1440x1440</option>
                        <option value="1024" selected>1024x1024</option>
                        <option value="768">768x768</option>
                        <option value="512">512x512</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Denoising Steps</label>
                    <div class="slider-container">
                        <input type="range" id="steps" min="1" max="100" value="20" oninput="updateVal('steps')">
                        <span id="steps-val" class="slider-val">20</span>
                    </div>
                </div>

                <div class="input-group">
                    <label>CFG Scale</label>
                    <div class="slider-container">
                        <input type="range" id="cfg" min="1" max="20" step="0.5" value="5.0" oninput="updateVal('cfg')">
                        <span id="cfg-val" class="slider-val">5.0</span>
                    </div>
                </div>

                <div class="input-group">
                    <label>Precision (Dtype)</label>
                    <select id="dtype">
                        <option value="bfloat16">bfloat16 (Fastest / Recommended)</option>
                        <option value="float16">float16</option>
                        <option value="float32">float32</option>
                    </select>
                </div>

                <div class="toggle-group">
                    <span style="font-size: 0.9rem; font-weight: 500;">Enable Prompt Reasoner</span>
                    <label class="switch">
                        <input type="checkbox" id="reasoner">
                        <span class="slider"></span>
                    </label>
                </div>

                <div class="toggle-group">
                    <span style="font-size: 0.9rem; font-weight: 500;">De-quantize GPT-OSS</span>
                    <label class="switch">
                        <input type="checkbox" id="disable_mxfp4">
                        <span class="slider"></span>
                    </label>
                </div>

                <div class="toggle-group">
                    <span style="font-size: 0.9rem; font-weight: 500;">CPU Offload (Reduce VRAM)</span>
                    <label class="switch">
                        <input type="checkbox" id="offload">
                        <span class="slider"></span>
                    </label>
                </div>
            </div>

            <button id="gen-btn" onclick="submitGeneration()">
                <span>🎨</span> Generate Masterpiece
            </button>
        </div>

        <!-- Right Side Workspace Area -->
        <div class="workspace">
            <div class="display-card">
                <!-- Initial Placeholder -->
                <div id="placeholder" class="placeholder-view">
                    <div class="placeholder-icon">🏞️</div>
                    <h3 style="font-size: 1.25rem; font-weight: 600; color: #fff;">Studio Canvas</h3>
                    <p style="font-size: 0.9rem; line-height: 1.4;">Enter a detailed prompt on the left and click generate. The server will spin up a dedicated high-performance GPU remotely to render your image.</p>
                </div>

                <!-- Real-time Progress Tracker -->
                <div id="status-box" class="status-container">
                    <div class="spinner"></div>
                    <div class="status-text" id="status-title">Queuing request...</div>
                    <div class="status-sub" id="status-sub">Allocating GPU resources...</div>
                    <div class="progress-track">
                        <div class="progress-bar" id="progress-bar"></div>
                    </div>
                </div>

                <!-- Image Viewer -->
                <div id="image-view" class="image-viewer">
                    <div class="image-frame">
                        <img id="result-img" alt="Generated result" onload="this.classList.add('loaded')">
                    </div>
                    <div class="image-actions">
                        <a id="btn-dl" class="btn-sec" target="_blank" download="lens-masterpiece.png">📥 Download</a>
                        <a id="btn-full" class="btn-sec" target="_blank">🔍 Full Size</a>
                    </div>
                </div>

                <!-- Error Display -->
                <div id="error-box" class="error-message"></div>
            </div>

            <!-- Refined Prompt Box -->
            <div id="refined-container" class="refined-box">
                <span>🤖 Reasoner Refined Prompt:</span>
                <p id="refined-text"></p>
            </div>
        </div>
    </main>

    <footer style="padding: 2rem; border-top: 1px solid var(--border-color); text-align: center; font-size: 0.8rem; color: var(--text-muted);">
        Powered by Microsoft Lens &middot; Hosted on Modal Serverless Platform &middot; Built with Vanilla CSS & FastAPI
    </footer>

    <script>
        function toggleAdvanced() {
            const content = document.getElementById('advanced-content');
            const chevron = document.getElementById('chevron');
            content.classList.toggle('open');
            chevron.classList.toggle('open');
        }

        function updateVal(id) {
            const range = document.getElementById(id);
            const val = document.getElementById(id + '-val');
            val.textContent = parseFloat(range.value).toFixed(id === 'cfg' ? 1 : 0);
        }

        // Auto-change default steps for Lens-Turbo
        document.getElementById('repo_id').addEventListener('change', function(e) {
            const stepsSlider = document.getElementById('steps');
            const stepsVal = document.getElementById('steps-val');
            if (e.target.value === 'microsoft/Lens-Turbo') {
                stepsSlider.value = 4;
                stepsVal.textContent = "4";
            } else {
                stepsSlider.value = 20;
                stepsVal.textContent = "20";
            }
        });

        let currentTaskId = null;
        let pollInterval = null;

        async function submitGeneration() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {
                alert('Please enter a prompt to continue!');
                return;
            }

            // Disable controls
            document.getElementById('gen-btn').disabled = true;
            document.getElementById('placeholder').style.display = 'none';
            document.getElementById('image-view').style.display = 'none';
            document.getElementById('error-box').style.display = 'none';
            document.getElementById('refined-container').style.display = 'none';
            
            // Show status tracker
            document.getElementById('status-box').style.display = 'flex';
            updateStatus('Queuing task...', 'Requesting isolated serverless GPU container...', 2);

            const payload = {
                prompt: prompt,
                repo_id: document.getElementById('repo_id').value,
                aspect_ratio: document.getElementById('aspect_ratio').value,
                base_resolution: parseInt(document.getElementById('base_resolution').value),
                steps: parseInt(document.getElementById('steps').value),
                cfg: parseFloat(document.getElementById('cfg').value),
                dtype: document.getElementById('dtype').value,
                reasoner: document.getElementById('reasoner').checked,
                disable_mxfp4: document.getElementById('disable_mxfp4').checked,
                offload: document.getElementById('offload').checked
            };

            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    throw new Error('Failed to communicate with ASGI web server');
                }

                const data = await response.json();
                currentTaskId = data.task_id;
                
                // Start polling
                if (pollInterval) clearInterval(pollInterval);
                pollInterval = setInterval(pollProgress, 1000);

            } catch (err) {
                showError('Could not launch generation task: ' + err.message);
            }
        }

        async function pollProgress() {
            if (!currentTaskId) return;

            try {
                const res = await fetch('/api/progress/' + currentTaskId);
                if (!res.ok) return;

                const data = await res.json();
                
                if (data.status === 'starting') {
                    updateStatus('Starting up...', 'Container booting, mounting dependencies...', 5);
                } else if (data.status === 'loading') {
                    updateStatus('Loading model...', 'Downloading weights from Hub & loading into VRAM...', 10);
                } else if (data.status === 'generating') {
                    const pct = data.progress || 10;
                    updateStatus(
                        `Generating image... ${pct}%`,
                        `Executing Denoising Step ${data.step} of ${data.total_steps}`,
                        pct
                    );
                } else if (data.status === 'saving') {
                    updateStatus('Saving images...', 'Denoising complete. Exporting PNG & writing to persistent volume...', 96);
                } else if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    showResult(data.images[0], data.refined_prompts[0], data.refined_prompts[0] !== document.getElementById('prompt').value);
                } else if (data.status === 'failed') {
                    clearInterval(pollInterval);
                    showError(data.error || 'Unknown failure occurred during inference.');
                }
            } catch (e) {
                console.error('Error polling progress:', e);
            }
        }

        function updateStatus(title, desc, percent) {
            document.getElementById('status-title').textContent = title;
            document.getElementById('status-sub').textContent = desc;
            document.getElementById('progress-bar').style.width = percent + '%';
        }

        function showError(msg) {
            document.getElementById('status-box').style.display = 'none';
            const errorBox = document.getElementById('error-box');
            errorBox.textContent = msg;
            errorBox.style.display = 'block';
            document.getElementById('gen-btn').disabled = false;
        }

        function showResult(filename, refinedPrompt, isRefined) {
            document.getElementById('status-box').style.display = 'none';
            document.getElementById('gen-btn').disabled = false;
            
            const imgUrl = '/outputs/' + filename;
            const imgEl = document.getElementById('result-img');
            
            imgEl.classList.remove('loaded');
            imgEl.src = imgUrl;
            
            document.getElementById('btn-dl').href = imgUrl;
            document.getElementById('btn-full').href = imgUrl;
            document.getElementById('image-view').style.display = 'flex';

            if (isRefined) {
                document.getElementById('refined-text').textContent = refinedPrompt;
                document.getElementById('refined-container').style.display = 'block';
            } else {
                document.getElementById('refined-container').style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""


from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    repo_id: str = "microsoft/Lens"
    base_resolution: int = 1024
    aspect_ratio: str = "1:1"
    steps: int = 20
    cfg: float = 5.0
    dtype: str = "bfloat16"
    disable_mxfp4: bool = False
    reasoner: bool = False
    offload: bool = False


@app.function(
    volumes={"/outputs": outputs_volume},
    timeout=600,
)
@modal.asgi_app()
def web_app():
    from fastapi import FastAPI, BackgroundTasks, HTTPException
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uuid

    web_app = FastAPI(title="Lens Text-to-Image")

    @web_app.get("/outputs/{filename}")
    def get_output_image(filename: str):
        from fastapi.responses import FileResponse
        import os

        # Reload the volume to see the latest writes from the GPU container
        outputs_volume.reload()

        file_path = os.path.join("/outputs", filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image not found")

        return FileResponse(file_path)

    @web_app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_CONTENT

    @web_app.post("/api/generate")
    def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
        task_id = str(uuid.uuid4())
        
        # Initialize the progress dict
        progress_dict[task_id] = {
            "status": "queued",
            "progress": 0,
            "step": 0,
            "total_steps": req.steps,
        }

        # Spawn the remote inference task asynchronously using Modal's .spawn() feature.
        # This returns instantly, allowing our web server to respond without blocking.
        run_remote_inference.spawn(
            prompt=req.prompt,
            repo_id=req.repo_id,
            base_resolution=req.base_resolution,
            aspect_ratio=req.aspect_ratio,
            steps=req.steps,
            cfg=req.cfg,
            dtype=req.dtype,
            disable_mxfp4=req.disable_mxfp4,
            reasoner=req.reasoner,
            offload=req.offload,
            out="/outputs",
            task_id=task_id,
        )

        return {"status": "queued", "task_id": task_id}

    @web_app.get("/api/progress/{task_id}")
    def progress(task_id: str):
        if task_id not in progress_dict:
            raise HTTPException(status_code=404, detail="Task not found")
        return progress_dict[task_id]

    return web_app


# ==============================================================================
# Local CLI Execution
# ==============================================================================

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
    out: str = "/outputs",
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
    Generated images are saved to a persistent Modal Volume.
    """
    print(f"Triggering remote inference on Modal for prompt: {prompt!r}")
    saved_paths = run_remote_inference.remote(
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
        out=out,
    )

    print("\nInference complete! Images successfully saved inside Modal Volume:")
    for path in saved_paths:
        print(f"  - {path}")
