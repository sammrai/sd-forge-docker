import os, time, torch
from diffusers import ZImagePipeline

def vram(tag):
    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM[{tag}] alloc={torch.cuda.memory_allocated()/2**30:.2f}G free={free/2**30:.2f}G", flush=True)

PROMPT=("Photo, male focus, manly, handsome, masculine, 30 year old, Jon Carpenter, "
        "dark hair, light stubble (no beard), muscular build, strong shoulders, strong arms, "
        "vascular, deep pec cleavage, undercut soaking wet hair, bangs in face, looking up, "
        "portrait, looking at viewer, dark eyes, charcoal gradient background")

pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
pipe.vae.to(dtype=torch.bfloat16)
onload, offload = torch.device("cuda"), torch.device("cpu")
pipe.vae.to(onload)
pipe.transformer.enable_group_offload(onload_device=onload, offload_device=offload,
    offload_type="block_level", num_blocks_per_group=1, use_stream=True, record_stream=False)
pipe.text_encoder.to(offload)
vram("after-offload")

ntok = len(pipe.tokenizer(PROMPT).input_ids)
max_seq = min(512, max(64, ntok + 16))
print(f"prompt tokens={ntok} -> max_seq={max_seq}", flush=True)

for tag, ms in (("capped", max_seq), ("full512", 512)):
    pipe.text_encoder.to("cuda")
    vram(f"{tag}-TE-on-gpu")
    try:
        pe, ne = pipe.encode_prompt(prompt=PROMPT, device=onload, do_classifier_free_guidance=True,
            negative_prompt="bad anatomy, lowres", max_sequence_length=ms)
        vram(f"{tag}-after-encode")
        print(f"  {tag}(max_seq={ms}) OK", flush=True)
    except torch.OutOfMemoryError as e:
        print(f"  {tag}(max_seq={ms}) OOM: {str(e)[:80]}", flush=True)
    pipe.text_encoder.to("cpu"); torch.cuda.empty_cache()
    vram(f"{tag}-after-offload")
print("=== DONE ===", flush=True)
