import base64
import io
import json
import math
import os
import random
from argparse import ArgumentParser

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()

PROMPT = (
    "You are watching a sequence of frames extracted from a short video. "
    "Images may contain multiple frames tiled side by side in chronological order (left to right, top to bottom). "
    "Look carefully for any physical violations: objects appearing or disappearing, "
    "teleporting, changing shape, or moving in physically impossible ways. "
    "It is ok for objects to come from outside the frame. "
    "About half of the videos are physically plausible, don't hesitate to give higher scores if you can't find any clear violation. "
    "Rate physical plausibility from 0 (impossible) to 1 (possible). "
    'Respond with JSON: {"score": <float between 0 and 1>}'
)
MAX_IMAGES = 4
THUMB_SIZE = 192
BORDER = 4


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dev"))
    parser.add_argument("--blocks", nargs="+", default=["O1", "O2", "O3"])
    parser.add_argument("--n-samples-per-block", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--model-id", default="qwen3-5-397b-a17b-fp8")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY"))
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--reasoning-effort",
        default="minimal",
        choices=["minimal", "low", "medium", "high"],
    )
    return parser.parse_args()


def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def make_grid(batch: list[Image.Image]) -> Image.Image:
    cols = math.ceil(math.sqrt(len(batch)))
    rows = math.ceil(len(batch) / cols)
    w = cols * THUMB_SIZE + BORDER * (cols + 1)
    h = rows * THUMB_SIZE + BORDER * (rows + 1)
    grid = Image.new("RGB", (w, h), color=(0, 0, 0))
    for i, img in enumerate(batch):
        r, c = divmod(i, cols)
        grid.paste(
            img.resize((THUMB_SIZE, THUMB_SIZE)),
            (BORDER + c * (THUMB_SIZE + BORDER), BORDER + r * (THUMB_SIZE + BORDER)),
        )
    return grid


def stitch(frames: list[Image.Image]) -> list[Image.Image]:
    if len(frames) <= MAX_IMAGES:
        return [img.resize((THUMB_SIZE, THUMB_SIZE)) for img in frames]
    per_tile = len(frames) // MAX_IMAGES
    return [make_grid(frames[i : i + per_tile]) for i in range(0, len(frames), per_tile)]


def sample_frames(scene_dir: Path, n: int) -> list[Image.Image]:
    frame_dir = scene_dir / "scene"
    paths = sorted(frame_dir.glob("*.png"))
    step = max(1, len(paths) // n)
    return [Image.open(p) for p in paths[::step][:n]]


def classify(
    client: OpenAI, model_id: str, frames: list[Image.Image], reasoning_effort: str
) -> tuple[float, str | None]:
    tiles = stitch(frames)
    for i, tile in enumerate(tiles):
        tile.save(f"tile_{i}.png")
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a physics reasoning assistant."},
            {"role": "user", "content": PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(t)}} for t in tiles
                ],
            },
        ],
        response_format={"type": "json_object"},
        reasoning_effort=reasoning_effort,
        extra_body={"chat_template_kwargs": {"enable_thinking": reasoning_effort != "minimal"}},
    )
    msg = response.choices[0].message
    score = max(0.0, min(1.0, float(json.loads(msg.content)["score"])))
    reasoning = getattr(msg, "reasoning_content", None)
    if not reasoning:
        raw = msg.model_extra or {}
        reasoning = raw.get("reasoning_content") or raw.get("reasoning")
    return score, reasoning


def evaluate(scores: list[float], labels: list[bool]) -> tuple[float, float]:
    abs_acc = sum((s >= 0.5) == gt for s, gt in zip(scores, labels)) / len(labels)
    ranked = sorted(zip(scores, labels), key=lambda x: (x[0], random.random()))
    rel_acc = sum(not gt for _, gt in ranked[:2]) + sum(gt for _, gt in ranked[2:])
    rel_acc /= len(labels)
    return abs_acc, rel_acc


def run_quadruplet(
    group_dir: Path,
    client: OpenAI,
    model_id: str,
    num_frames: int,
    verbose: bool,
    reasoning_effort: str,
) -> tuple[dict, float, float]:
    sid = group_dir.name
    print(sid)
    results, scores, labels = {}, [], []
    for vdir in sorted(d for d in group_dir.iterdir() if d.is_dir()):
        label = json.loads((vdir / "status.json").read_text())["header"]["is_possible"]
        frames = sample_frames(vdir, num_frames)
        score, reasoning = classify(client, model_id, frames, reasoning_effort)
        results[f"{sid}/{vdir.name}"] = {"score": score, "reasoning": reasoning}
        scores.append(score)
        labels.append(label)
        print(f"  {vdir.name} (possible: {label}) {score:.3f}")
        if verbose and reasoning:
            print(f"reasoning: {reasoning}")
    abs_acc, rel_acc = evaluate(scores, labels)
    print(f"abs={abs_acc:.0%}  rel={rel_acc:.0%}")
    return results, abs_acc, rel_acc


def run_block(
    block_dir: Path,
    client: OpenAI,
    model_id: str,
    num_frames: int,
    n_samples: int | None,
    verbose: bool,
    reasoning_effort: str,
):
    results = {}
    abs_accs, rel_accs = [], []
    groups = sorted(d for d in block_dir.iterdir() if d.is_dir())
    if n_samples is not None:
        groups = groups[:n_samples]
    for group_dir in groups:
        quad_results, abs_acc, rel_acc = run_quadruplet(
            group_dir,
            client,
            model_id,
            num_frames,
            verbose,
            reasoning_effort,
        )
        results.update(quad_results)
        abs_accs.append(abs_acc)
        rel_accs.append(rel_acc)
    return results, abs_accs, rel_accs


def main():
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    all_results = {}
    all_abs, all_rel = [], []
    for block in args.blocks:
        print(f"=== {block} ===")
        results, abs_accs, rel_accs = run_block(
            args.data_dir / block,
            client,
            args.model_id,
            args.num_frames,
            args.n_samples_per_block,
            args.verbose,
            args.reasoning_effort,
        )
        all_results.update({f"{block}/{k}": v for k, v in results.items()})
        all_abs.extend(abs_accs)
        all_rel.extend(rel_accs)
        print(
            f"{block}: abs={sum(abs_accs) / len(abs_accs):.1%}  rel={sum(rel_accs) / len(rel_accs):.1%}"
        )
    args.output.write_text(json.dumps(all_results, indent=2))
    print(f"Saved to {args.output}")
    print(f"Overall: abs={sum(all_abs) / len(all_abs):.1%}  rel={sum(all_rel) / len(all_rel):.1%}")


if __name__ == "__main__":
    main()
