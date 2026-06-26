import os, time, glob, torch
from diffusers import ZImagePipeline, ZImageImg2ImgPipeline
from diffusers.hooks import apply_group_offloading
from PIL import Image

USE_STREAM = os.environ.get("ZS", "1") == "1"
REPO = "Tongyi-MAI/Z-Image-Turbo"
LORA_DIR = "/app/data/models/Lora"
WANT = [
    ("ZIT_Skinny_slider_v1.1-mid_2229239-vid_2509564", 1.1),
    ("z-image-base_nsfw_bodybuilder_v2-e20-mid_2359891-vid_2710223", 0.3),
]

def vram(tag):
    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM[{tag}] alloc={torch.cuda.memory_allocated()/2**30:.2f}G free={free/2**30:.2f}G", flush=True)

def find(name):
    hits = glob.glob(f"{LORA_DIR}/**/{name}.safetensors", recursive=True)
    return hits[0] if hits else None

print(f"=== use_stream={USE_STREAM} ===", flush=True)
t0=time.time()
pipe = ZImagePipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16)
pipe.vae.to(dtype=torch.bfloat16)
for m in ("enable_tiling","enable_slicing"):
    getattr(pipe.vae, m)()
onload, offload = torch.device("cuda"), torch.device("cpu")
pipe.vae.to(onload)
pipe.transformer.enable_group_offload(onload_device=onload, offload_device=offload,
    offload_type="block_level", num_blocks_per_group=1, use_stream=USE_STREAM, record_stream=False)
# TE は CPU 常駐(group offload は transformers モデルで壊れる)。encode は CPU で実行。
pipe.text_encoder.to("cpu")
print(f"loaded+offload in {time.time()-t0:.0f}s", flush=True)
vram("after-offload")

names, weights = [], []
for nm, w in WANT:
    p = find(nm)
    if not p:
        print("  lora missing", nm); continue
    pipe.load_lora_weights(p, adapter_name=nm.replace("-","_").replace(".","_")[:60])
    names.append(nm.replace("-","_").replace(".","_")[:60]); weights.append(w)
    print("  lora applied", nm, flush=True)
pipe.set_adapters(names, weights)
vram("after-lora")

PROMPT="Masterpiece, best quality, 1boy, male focus, mature male, bara, tanned skin, black hair, stubble, hairy pecs, fat, chubby, big belly, big pecs, broad shoulders, detailed eyes, perfect face, anime art style, upper body, close up, white shirt, looking up, indoors, office, windows, morning, "*2
print("encode on GPU (TE wholesale)...", flush=True)
te0=time.time()
pipe.text_encoder.to("cuda")
vram("TE-on-gpu")
pe, ne = pipe.encode_prompt(prompt=PROMPT, device=torch.device("cuda"),
    do_classifier_free_guidance=True, negative_prompt="bad anatomy, lowres, blurry")
vram("after-encode-peak")
pipe.text_encoder.to("cpu"); torch.cuda.empty_cache()
print(f"  gpu encode in {time.time()-te0:.0f}s", flush=True)
vram("after-te-offload")

print("base 832x1216...", flush=True)
t1=time.time()
out = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, num_inference_steps=8,
    guidance_scale=1.0, width=832, height=1216, num_images_per_prompt=1,
    generator=torch.Generator("cpu").manual_seed(0))
print(f"  base done in {time.time()-t1:.0f}s", flush=True)
vram("after-base")

img = out.images[0].resize((1664,2432), Image.LANCZOS)
i2i = ZImageImg2ImgPipeline.from_pipe(pipe)
print("hires 1664x2432...", flush=True)
t2=time.time()
pipe.text_encoder.to("cuda")
hpe,hne = pipe.encode_prompt(prompt=PROMPT, device=torch.device("cuda"),
    do_classifier_free_guidance=True, negative_prompt="bad anatomy, lowres, blurry")
pipe.text_encoder.to("cpu"); torch.cuda.empty_cache()
vram("pre-hires")
hr = i2i(prompt_embeds=hpe, negative_prompt_embeds=hne, image=img, strength=0.5,
    num_inference_steps=8, guidance_scale=1.0, width=1664, height=2432,
    num_images_per_prompt=1, generator=torch.Generator("cpu").manual_seed(0))
print(f"  hires done in {time.time()-t2:.0f}s", flush=True)
vram("after-hires")
print("=== SUCCESS ===", flush=True)
