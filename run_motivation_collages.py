#!/usr/bin/env python3
"""
run_motivation_collages.py
---------------------------
Мотивационный эксперимент: визуализация влияния разных методов
модификации начального шума на финальную картинку.

Генерирует 11 коллажей:
  Alpha:     A1, A2, A3
  Base/c:    B1, B2, B3
  Affine v1: V1a, V1b
  Affine v2: W1, W2, W3

Каждый коллаж подписан сверху — что именно варьируется.
Под каждой картинкой — значение параметра + ImageReward.

Запуск:
    python run_motivation_collages.py
    python run_motivation_collages.py --skip_ir  (без ImageReward, быстрее)
"""

import argparse, time
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
import ImageReward as RM

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

FLUX_MODEL = "black-forest-labs/FLUX.1-dev"
DEVICE     = "cuda"
DTYPE      = torch.bfloat16

PROMPT = "a portrait of a man"
SEED   = 42
K      = 10
STEPS  = 28
CFG    = 3.5
H, W   = 1024, 1024

# ──────────────────────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default="results/motivation1")
    p.add_argument("--skip_ir",  action="store_true",
                   help="не считать ImageReward (быстрее)")
    p.add_argument("--thumb",    type=int, default=256,
                   help="размер одной картинки в коллаже (пикс)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# FLUX УТИЛИТЫ
# ──────────────────────────────────────────────────────────────

def get_mu(pipe):
    seq = (H // 16) * (W // 16)
    bs  = pipe.scheduler.config.get("base_image_seq_len", 256)
    ms  = pipe.scheduler.config.get("max_image_seq_len", 4096)
    bsh = pipe.scheduler.config.get("base_shift", 0.5)
    msh = pipe.scheduler.config.get("max_shift", 1.15)
    return (msh - bsh) * (seq - bs) / (ms - bs) + bsh

def prepare_latent(pipe, seed):
    gen = torch.Generator("cpu").manual_seed(seed)
    nc  = pipe.transformer.config.in_channels // 4
    return randn_tensor((1, nc, H//8, W//8),
                        generator=gen, device=pipe.device, dtype=pipe.dtype)

def pack(lat):
    b, c, hh, ww = lat.shape
    lat = lat.view(b, c, hh//2, 2, ww//2, 2).permute(0,2,4,1,3,5)
    return lat.reshape(b, (hh//2)*(ww//2), c*4)

def unpack(lat):
    b, seq, _ = lat.shape
    c = lat.shape[-1] // 4
    hh, ww = H//16, W//16
    lat = lat.view(b, hh, ww, c, 2, 2).permute(0,3,1,4,2,5)
    return lat.reshape(b, c, H//8, W//8)

def decode(pipe, lat):
    lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img = pipe.vae.decode(lat, return_dict=False)[0]
    img = (img/2+0.5).clamp(0,1).cpu().permute(0,2,3,1).float().numpy()
    return Image.fromarray((img[0]*255).round().astype("uint8"))

@torch.no_grad()
def generate(pipe, z_lat, prompt):
    pe, ppe, txt_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None,
        device=DEVICE, num_images_per_prompt=1)
    mu  = get_mu(pipe)
    pipe.scheduler.set_timesteps(STEPS, device=DEVICE, mu=mu)
    g   = torch.full((1,), CFG, device=DEVICE, dtype=DTYPE)
    ids = pipe._prepare_latent_image_ids(1, H//16, W//16, DEVICE, DTYPE)
    lat = z_lat.clone()
    for t in pipe.scheduler.timesteps:
        lp  = pack(lat)
        vp  = pipe.transformer(
            hidden_states=lp, timestep=t.expand(1)/1000,
            guidance=g, encoder_hidden_states=pe,
            pooled_projections=ppe, txt_ids=txt_ids, img_ids=ids)[0]
        vp  = unpack(vp)
        lat = pipe.scheduler.step(vp, t, lat, return_dict=False)[0]
    return decode(pipe, lat)


# ──────────────────────────────────────────────────────────────
# DCT
# ──────────────────────────────────────────────────────────────

def dct2(x):
    hh, ww = x.shape
    v   = np.concatenate([x, x[:, ::-1]], axis=1)
    V   = np.fft.rfft(v, axis=1)[:, :ww]
    k   = np.arange(ww, dtype=np.float64)
    row = (V * np.exp(-1j*np.pi*k/(2*ww))).real
    v2  = np.concatenate([row, row[::-1, :]], axis=0)
    V2  = np.fft.rfft(v2, axis=0)[:hh, :]
    k2  = np.arange(hh, dtype=np.float64)[:, None]
    res = (V2 * np.exp(-1j*np.pi*k2/(2*hh))).real
    res /= np.sqrt(4*hh*ww)
    res[0, :] /= np.sqrt(2)
    res[:, 0] /= np.sqrt(2)
    return res

def idct2(X):
    hh, ww = X.shape
    Y = X.copy()
    Y[0, :] *= np.sqrt(2)
    Y[:, 0] *= np.sqrt(2)
    Y *= np.sqrt(4*hh*ww)
    k   = np.arange(ww, dtype=np.float64)
    k2  = np.arange(hh, dtype=np.float64)[:, None]
    V   = Y * np.exp(1j*np.pi*k/(2*ww))
    v   = np.fft.irfft(V, n=2*ww, axis=1)[:, :ww]
    V2  = v * np.exp(1j*np.pi*k2/(2*hh))
    return np.fft.irfft(V2, n=2*hh, axis=0)[:hh, :] / (4*hh*ww)

def get_low_idx(C, hh, ww, k):
    fy = np.arange(hh, dtype=np.float32)
    fx = np.arange(ww, dtype=np.float32)
    dist = np.sqrt(fy[:,None]**2 + fx[None,:]**2).flatten()
    return np.argsort(dist)[:k]

def apply_mod(z_np, method, params, low_idx, C, hh, ww):
    """Применяем модификацию и нормируем норму."""
    z_mod = z_np.copy().astype(np.float64)
    orig_norms = np.linalg.norm(z_mod.reshape(C, -1), axis=1)
    for ch in range(C):
        flat = dct2(z_mod[ch]).flatten()
        if method == "alpha":
            alpha = params["alpha"]
            flat[low_idx] *= alpha
        elif method == "base":
            c = params["c"]
            flat[low_idx] += c
        elif method == "affine_v1":
            alpha, c = params["alpha"], params["c"]
            flat[low_idx] = alpha * flat[low_idx] + c
        elif method == "affine_v2":
            a, c = params["a"], params["c"]
            flat[low_idx] = a * flat[low_idx] + c
        z_mod[ch] = idct2(flat.reshape(hh, ww))
    # нормировка нормы по каналам
    new_norms = np.linalg.norm(z_mod.reshape(C, -1), axis=1)
    for ch in range(C):
        if new_norms[ch] > 1e-8:
            z_mod[ch] *= orig_norms[ch] / new_norms[ch]
    return z_mod.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# КОЛЛАЖ
# ──────────────────────────────────────────────────────────────

def load_font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        return ImageFont.load_default()

def make_collage(images, labels, title, thumb, n_cols=None):
    """
    images : list of PIL.Image
    labels : list of str (под каждой картинкой)
    title  : str (сверху всего коллажа)
    """
    n = len(images)
    if n_cols is None:
        n_cols = n
    n_rows = (n + n_cols - 1) // n_cols

    pad    = 10
    label_h = 36
    title_h = 48
    cell_w  = thumb + pad
    cell_h  = thumb + label_h + pad

    canvas_w = cell_w * n_cols + pad
    canvas_h = cell_h * n_rows + title_h + pad
    canvas   = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw     = ImageDraw.Draw(canvas)

    f_title = load_font(18)
    f_label = load_font(13)

    # заголовок
    draw.text((pad, 8), title, fill=(30, 30, 30), font=f_title)

    for i, (img, lbl) in enumerate(zip(images, labels)):
        row = i // n_cols
        col = i %  n_cols
        x   = pad + col * cell_w
        y   = title_h + row * cell_h

        thumb_img = img.resize((thumb, thumb), Image.LANCZOS)
        canvas.paste(thumb_img, (x, y))

        # метка под картинкой
        draw.text((x, y + thumb + 4), lbl,
                  fill=(60, 60, 60), font=f_label)

    return canvas


# ──────────────────────────────────────────────────────────────
# ГЕНЕРАЦИЯ ВСЕХ КОЛЛАЖЕЙ
# ──────────────────────────────────────────────────────────────

def run_all(pipe, scorer, z_base_np, low_idx, C, hh, ww,
            save_dir, thumb, skip_ir):

    def gen_image(params_dict, method):
        z_mod = apply_mod(z_base_np, method, params_dict,
                          low_idx, C, hh, ww)
        z_t   = torch.tensor(z_mod[None], device=DEVICE, dtype=DTYPE)
        img   = generate(pipe, z_t, PROMPT)
        ir    = 0.0 if skip_ir else float(scorer.score(PROMPT, img))
        return img, ir

    def save(collage, name):
        p = save_dir / f"{name}.png"
        collage.save(p)
        print(f"  ✓ {name}.png")

    # ── A1: Alpha — простая строка ──────────────────────────
    print("\n[A1] Alpha sweep...")
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5 , 3.0]
    imgs, lbls = [], []
    for a in alphas:
        img, ir = gen_image({"alpha": a}, "alpha")
        imgs.append(img)
        lbls.append(f"α={a:.2f}\nIR={ir:.3f}")
    save(make_collage(imgs, lbls,
        "A1 — Alpha sweep: DCT_low *= alpha  |  все значения alpha",
        thumb), "A1_alpha_sweep")

    # ── A2: Alpha + IR крупно ───────────────────────────────
    # (те же картинки, другое форматирование — IR выделен)
    lbls2 = [f"α={a:.2f}  IR={ir:.3f}"
             for a, (_, ir) in zip(alphas,
             [gen_image({"alpha": a}, "alpha") for a in []])]
    # используем уже сгенерированные
    lbls2 = [f"α={alphas[i]:.2f}  IR={float(scorer.score(PROMPT, imgs[i])):.3f}"
             if not skip_ir else f"α={alphas[i]:.2f}"
             for i in range(len(imgs))]
    save(make_collage(imgs, lbls2,
        "A2 — Alpha sweep + ImageReward  |  DCT_low *= alpha",
        thumb), "A2_alpha_with_ir")

    # ── A3: Alpha — две строки: low (<1) и high (>1) ────────
    low_a  = [a for a in alphas if a <= 1.0]
    high_a = [a for a in alphas if a >= 1.0]
    imgs_low  = [imgs[alphas.index(a)] for a in low_a]
    imgs_high = [imgs[alphas.index(a)] for a in high_a]
    lbls_low  = [f"α={a}" for a in low_a]
    lbls_high = [f"α={a}" for a in high_a]
    row1 = make_collage(imgs_low,  lbls_low,
        "A3 — Alpha LOW (ослабляем низкие частоты)", thumb)
    row2 = make_collage(imgs_high, lbls_high,
        "        HIGH (усиливаем низкие частоты)", thumb)
    combined = Image.new("RGB",
        (max(row1.width, row2.width), row1.height + row2.height + 8),
        (245, 245, 245))
    combined.paste(row1, (0, 0))
    combined.paste(row2, (0, row1.height + 8))
    save(combined, "A3_alpha_low_vs_high")

    # ── B1: Base — только DC компонента ─────────────────────
    print("\n[B1] Base — DC only sweep...")
    t_vals = [-3, -2, -1, 0, 1, 2, 3]
    imgs_b1, lbls_b1 = [], []
    c_zero = np.zeros(K)
    for t in t_vals:
        c = c_zero.copy(); c[0] = t
        img, ir = gen_image({"c": c}, "base")
        imgs_b1.append(img)
        lbls_b1.append(f"c[0]={t}\nIR={ir:.3f}")
    save(make_collage(imgs_b1, lbls_b1,
        "B1 — Base: сдвигаем только DC (коэф 0)  |  c=[t,0,...,0]",
        thumb), "B1_base_dc_only")

    # ── B2: Base — случайное направление ────────────────────
    print("\n[B2] Base — random direction sweep...")
    rng = np.random.RandomState(0)
    c_rand = rng.randn(K).astype(np.float32)
    c_rand /= np.linalg.norm(c_rand)
    t_vals2 = [-2, -1, 0, 1, 2]
    imgs_b2, lbls_b2 = [], []
    for t in t_vals2:
        c = (t * c_rand)
        img, ir = gen_image({"c": c}, "base")
        imgs_b2.append(img)
        lbls_b2.append(f"t={t}\nIR={ir:.3f}")
    save(make_collage(imgs_b2, lbls_b2,
        "B2 — Base: случайное направление  |  c = t · c_rand",
        thumb), "B2_base_random_dir")

    # ── B3: Base — каждый коэф отдельно ─────────────────────
    print("\n[B3] Base — per-coefficient (10 строк × 3 столбца)...")
    t3 = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    imgs_b3, lbls_b3 = [], []
    for ki in range(K):
        for t in t3:
            c = np.zeros(K); c[ki] = t
            img, ir = gen_image({"c": c}, "base")
            imgs_b3.append(img)
            lbls_b3.append(f"c[{ki}]={t}\nIR={ir:.3f}")
    save(make_collage(imgs_b3, lbls_b3,
        "B3 — Base: каждый коэф отдельно  |  строки=коэф, столбцы=t∈{-5..+5}",
        thumb, n_cols=3), "B3_base_per_coef")

    # ── V1a: Affine v1 — сетка alpha × c ────────────────────
    print("\n[V1a] Affine v1 — grid alpha × c...")
    alphas_v1 = [0.5, 1.0, 1.5, 2.0]
    t_c_v1    = [-1,  0,   1]
    imgs_v1a, lbls_v1a = [], []
    for a in alphas_v1:
        for t in t_c_v1:
            c = np.full(K, t, dtype=np.float32)
            img, ir = gen_image({"alpha": a, "c": c}, "affine_v1")
            imgs_v1a.append(img)
            lbls_v1a.append(f"α={a} c={t}\nIR={ir:.3f}")
    save(make_collage(imgs_v1a, lbls_v1a,
        "V1a — Affine v1: сетка  |  строки=alpha, столбцы=c  |  α·DCT_low+c",
        thumb, n_cols=len(t_c_v1)), "V1a_affine_v1_grid")

    # ── V1b: Affine v1 — две строки ─────────────────────────
    print("\n[V1b] Affine v1 — two rows...")
    imgs_v1b, lbls_v1b = [], []
    # строка 1: фикс c=0, варьируем alpha
    for a in [0.25, 0.5, 1.0, 1.5, 2.0]:
        c = np.zeros(K)
        img, ir = gen_image({"alpha": a, "c": c}, "affine_v1")
        imgs_v1b.append(img)
        lbls_v1b.append(f"α={a} c=0\nIR={ir:.3f}")
    # строка 2: фикс alpha=1, варьируем c
    for t in [-2, -1, 0, 1, 2]:
        c = np.full(K, t, dtype=np.float32)
        img, ir = gen_image({"alpha": 1.0, "c": c}, "affine_v1")
        imgs_v1b.append(img)
        lbls_v1b.append(f"α=1 c={t}\nIR={ir:.3f}")
    save(make_collage(imgs_v1b, lbls_v1b,
        "V1b — Affine v1: строка1=фикс c=0 варьируем α  |  строка2=фикс α=1 варьируем c",
        thumb, n_cols=5), "V1b_affine_v1_two_rows")

    # ── W1: Affine v2 — только a_0 ──────────────────────────
    print("\n[W1] Affine v2 — a_0 sweep...")
    a0_vals = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    imgs_w1, lbls_w1 = [], []
    for a0 in a0_vals:
        a = np.ones(K); a[0] = a0
        c = np.zeros(K)
        img, ir = gen_image({"a": a, "c": c}, "affine_v2")
        imgs_w1.append(img)
        lbls_w1.append(f"a[0]={a0}\nIR={ir:.3f}")
    save(make_collage(imgs_w1, lbls_w1,
        "W1 — Affine v2: варьируем только a[0] (DC)  |  остальные a_i=1, c=0",
        thumb), "W1_affine_v2_a0")

    # ── W2: Affine v2 — сетка a_0 × a_1 ────────────────────
    print("\n[W2] Affine v2 — grid a_0 × a_1...")
    a01_vals = [0.5, 1.0, 1.5, 2.0]
    imgs_w2, lbls_w2 = [], []
    for a0 in a01_vals:
        for a1 in a01_vals:
            a = np.ones(K); a[0] = a0; a[1] = a1
            c = np.zeros(K)
            img, ir = gen_image({"a": a, "c": c}, "affine_v2")
            imgs_w2.append(img)
            lbls_w2.append(f"a0={a0}\na1={a1} IR={ir:.3f}")
    save(make_collage(imgs_w2, lbls_w2,
        "W2 — Affine v2: сетка a[0]×a[1]  |  строки=a0, столбцы=a1",
        thumb, n_cols=len(a01_vals)), "W2_affine_v2_grid")

    # ── W3: Affine v2 — каждый a_i отдельно ─────────────────
    print("\n[W3] Affine v2 — per a_i (10 строк × 5 столбца)...")
    a_sweep = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    imgs_w3, lbls_w3 = [], []
    for ki in range(K):
        for av in a_sweep:
            a = np.ones(K); a[ki] = av
            c = np.zeros(K)
            img, ir = gen_image({"a": a, "c": c}, "affine_v2")
            imgs_w3.append(img)
            lbls_w3.append(f"a[{ki}]={av}\nIR={ir:.3f}")
    save(make_collage(imgs_w3, lbls_w3,
        "W3 — Affine v2: каждый a_i отдельно  |  строки=коэф, столбцы=a∈{0..3}}",
        thumb, n_cols=len(a_sweep)), "W3_affine_v2_per_ai")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Мотивационный эксперимент — 11 коллажей")
    print(f"PROMPT: {PROMPT[:50]}...")
    print(f"SEED={SEED}  K={K}  thumb={args.thumb}")
    print(f"skip_ir={args.skip_ir}")
    print("=" * 60)

    print("\n[1/3] Загрузка FLUX...")
    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL, torch_dtype=DTYPE).to(DEVICE)

    scorer = None
    if not args.skip_ir:
        print("[2/3] Загрузка ImageReward...")
        scorer = RM.load("ImageReward-v1.0", device=DEVICE)
    else:
        print("[2/3] ImageReward пропущен (--skip_ir)")

    print("[3/3] Подготовка базового шума...")
    z_base = prepare_latent(pipe, SEED)
    C, hh, ww = (pipe.transformer.config.in_channels // 4,
                 H // 8, W // 8)
    low_idx = get_low_idx(C, hh, ww, K)
    z_base_np = z_base[0].float().cpu().numpy()  # (C, hh, ww)

    print(f"\nLatent shape: ({C}, {hh}, {ww})")
    print(f"Low freq idx: первые {K} из {hh*ww} коэффициентов")
    print(f"\nСтарт генерации коллажей...\n")

    t0 = time.time()
    run_all(pipe, scorer, z_base_np, low_idx, C, hh, ww,
            save_dir, args.thumb, args.skip_ir)

    print(f"\n{'='*60}")
    print(f"Готово за {(time.time()-t0)/60:.1f} мин")
    print(f"Сохранено в {save_dir}/")
    print("  A1_alpha_sweep.png")
    print("  A2_alpha_with_ir.png")
    print("  A3_alpha_low_vs_high.png")
    print("  B1_base_dc_only.png")
    print("  B2_base_random_dir.png")
    print("  B3_base_per_coef.png")
    print("  V1a_affine_v1_grid.png")
    print("  V1b_affine_v1_two_rows.png")
    print("  W1_affine_v2_a0.png")
    print("  W2_affine_v2_grid.png")
    print("  W3_affine_v2_per_ai.png")


if __name__ == "__main__":
    main()