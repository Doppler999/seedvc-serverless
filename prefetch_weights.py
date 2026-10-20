"""Build-time weight bake for the SeedVC serverless image.

Downloads every artifact the Seed-VC V2 (quality) and V1 (fast) engines need
when they load, pins each Hugging Face repo to an immutable revision, and
verifies the SHA-256 of every large (LFS) file. Large artifacts land in the
repo-local HF cache (/workspace/seed-vc/checkpoints) exactly where
hf_utils.load_custom_model_from_hf looks for them; the rest land in $HF_HOME.

V1 (~1.9 GB): the 22.05 kHz base DiT, the 44.1 kHz f0-conditioned DiT used by
`f0_conditioned=true`, both corresponding BigVGAN vocoders and RMVPE.

The runtime image is built with HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1, so a
missing artifact fails loudly at boot instead of being downloaded silently.

Usage: python prefetch_weights.py [--verify-load]
"""
import argparse
import hashlib
import os
import sys

from huggingface_hub import hf_hub_download

SEED_VC_DIR = os.environ.get("SEED_VC_DIR", "/workspace/seed-vc")
CHECKPOINT_DIR = os.path.join(SEED_VC_DIR, "checkpoints")

# (repo_id, revision, filename, sha256 or None)
# LFS blobs: sha256 == the HF blob id, so these double as integrity pins.
V2_ARTIFACTS = [
    # V2 CFM + AR checkpoints (the 856 MB the engine loads)
    ("Plachta/Seed-VC", "257283f9f41585055e8f858fba4fd044e5caed6e", "v2/cfm_small.pth",
     "73ae18615dc8cb01710086629973198226dedc63ebeaab0aee17863d767b3474"),
    ("Plachta/Seed-VC", "257283f9f41585055e8f858fba4fd044e5caed6e", "v2/ar_base.pth",
     "42e2afabf97e5d856580af1f81a9705423b4fb21bb9475cece9925b62355cc5c"),
    # content extractor (narrow + wide)
    ("Plachta/ASTRAL-quantization", "4a2e9679f76eb03753adc8c503e3c23bb9c22f26", "bsq32/bsq32_light.pth",
     "b01d6ba14a7af57ecc02b3eb5897a7e2abdaefab06b3f218d6709dc98b0e83a3"),
    ("Plachta/ASTRAL-quantization", "4a2e9679f76eb03753adc8c503e3c23bb9c22f26", "bsq2048/bsq2048_light.pth",
     "d7953115a60609c72a50bc79e88aa7248ffd4444b3912f89e8375ec1c6485fc6"),
    # speaker/style encoder
    ("funasr/campplus", "e4b6ede7ce16997aff4ae69fbca1f0175e2afede", "campplus_cn_common.bin",
     "3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8"),
    # vocoder
    ("nvidia/bigvgan_v2_22khz_80band_256x", "633ff708ed5b74903e86ff1298cf4a98e921c513",
     "bigvgan_generator.pt",
     "e95ba25972d3de0628d99cd156e9315a9c018899bf739988959ebe3544080ced"),
    ("nvidia/bigvgan_v2_22khz_80band_256x", "633ff708ed5b74903e86ff1298cf4a98e921c513",
     "config.json", None),
    # content extractor SSL backbone
    ("facebook/hubert-large-ll60k", "ff022d095678a2995f3c49bab18a96a9e553f782", "pytorch_model.bin",
     "45a299050945479a68cffe2ab7a63fd08931718ba18f05a42dbb86a5164178e0"),
    ("facebook/hubert-large-ll60k", "ff022d095678a2995f3c49bab18a96a9e553f782", "config.json", None),
    ("facebook/hubert-large-ll60k", "ff022d095678a2995f3c49bab18a96a9e553f782",
     "preprocessor_config.json", None),
    # whisper-small tokenizer (text side of the content extractor)
    ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d", "tokenizer.json", None),
    ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d", "tokenizer_config.json", None),
    ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d", "vocab.json", None),
    ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d", "merges.txt", None),
    ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d", "special_tokens_map.json", None),
    ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d", "added_tokens.json", None),
    ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d", "normalizer.json", None),
    ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d", "config.json", None),
]

# V1 (fast) artifacts, read straight out of seed_vc_wrapper.py at the pinned
# commit: Plachta/Seed-VC DiT checkpoints + configs, RMVPE, 44.1 kHz BigVGAN.
V1_ARTIFACTS = [
    # base (speech) DiT, 22.05 kHz
    ("Plachta/Seed-VC", "257283f9f41585055e8f858fba4fd044e5caed6e",
     "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
     "8ec8841b20bb46df9f7e8e570a6946a4b87b940133c7f0e778487ff33841f720"),
    ("Plachta/Seed-VC", "257283f9f41585055e8f858fba4fd044e5caed6e",
     "config_dit_mel_seed_uvit_whisper_small_wavenet.yml", None),
    # f0-conditioned (singing) DiT, 44.1 kHz
    ("Plachta/Seed-VC", "257283f9f41585055e8f858fba4fd044e5caed6e",
     "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
     "883d19be9fe9c4bb2dce30d545cd77e5ed2b6326c7ed1c4d9a7ce8e555b163f9"),
    ("Plachta/Seed-VC", "257283f9f41585055e8f858fba4fd044e5caed6e",
     "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml", None),
    # F0 extractor for the f0-conditioned path
    ("lj1995/VoiceConversionWebUI", "e6d0c1a17da07c33557852f9dfa2bd44cc75737d", "rmvpe.pt",
     "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193"),
    # 44.1 kHz vocoder (V1 f0 path)
    ("nvidia/bigvgan_v2_44khz_128band_512x", "95a9d1dcb12906c03edd938d77b9333d6ded7dfb",
     "bigvgan_generator.pt",
     "d9fe7ec6bd0b44ed9d66973d5012d8181c1570b01e5c72df51973e241dccd357"),
    ("nvidia/bigvgan_v2_44khz_128band_512x", "95a9d1dcb12906c03edd938d77b9333d6ded7dfb",
     "config.json", None),
]

# hf_utils.load_custom_model_from_hf() always caches under ./checkpoints, so
# every repo it resolves has to be baked there (not in $HF_HOME).
REPO_LOCAL_CACHE_PREFIXES = ("Plachta/", "funasr/", "lj1995/")

ARTIFACTS = V2_ARTIFACTS + V1_ARTIFACTS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_all():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    total = 0
    for repo_id, revision, filename, expected in ARTIFACTS:
        cache_dir = CHECKPOINT_DIR if repo_id.startswith(REPO_LOCAL_CACHE_PREFIXES) else None
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
        size = os.path.getsize(path)
        if expected:
            actual = sha256(path)
            if actual != expected:
                raise SystemExit(f"CHECKSUM MISMATCH {repo_id}/{filename}: {actual} != {expected}")
            print(f"[ok] {repo_id}/{filename} {size / 1e6:.1f} MB sha256={actual}", flush=True)
        else:
            print(f"[ok] {repo_id}/{filename} {size / 1e6:.2f} MB", flush=True)
        total += size
    print(f"[ok] baked {total / 1e6:.0f} MB across {len(ARTIFACTS)} files", flush=True)


def verify_load():
    """Instantiate the V2 wrapper from the baked cache (offline) and load weights."""
    sys.path.insert(0, SEED_VC_DIR)
    os.chdir(SEED_VC_DIR)
    import torch
    import yaml
    from hydra.utils import instantiate
    from omegaconf import DictConfig

    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    wrapper = instantiate(cfg)
    wrapper.load_checkpoints()
    n_params = sum(p.numel() for p in wrapper.parameters())
    print(f"[ok] V2 wrapper instantiated and checkpoints loaded, {n_params / 1e6:.1f} M params", flush=True)
    del wrapper
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # V1 (fast): both DiTs, both vocoders, CAMPPlus and RMVPE load in __init__.
    from seed_vc_wrapper import SeedVCWrapper

    v1 = SeedVCWrapper(device=torch.device("cpu"))
    print(
        f"[ok] V1 wrapper loaded on cpu: sr_base={v1.sr} sr_f0={v1.sr_f0} "
        f"has_model_f0={v1.model_f0 is not None}",
        flush=True,
    )
    del v1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-load", action="store_true",
                    help="instantiate the V2 and V1 wrappers from the baked cache after downloading")
    args = ap.parse_args()
    download_all()
    if args.verify_load:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        verify_load()
    print("[ok] weights baked", flush=True)
