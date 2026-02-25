import sys
import os
import math
from typing import Tuple

import numpy as np
from tdl import nn, image


def l2_normalize(feat: np.ndarray) -> np.ndarray:
    """L2 normalize a feature vector."""
    norm = np.linalg.norm(feat)
    if norm < 1e-12:
        return feat
    return feat / norm


def cosine_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    """Compute cosine similarity between two feature vectors."""
    f1 = l2_normalize(f1.astype(np.float32))
    f2 = l2_normalize(f2.astype(np.float32))
    return float(np.dot(f1, f2))


def infer_feature(
    model,
    img_path: str,
) -> np.ndarray:
    """Extract ReID feature for a single image."""
    if not os.path.exists(img_path):
        print(f"Failed to find image: {img_path}")
        sys.exit(1)

    img = image.read(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        sys.exit(1)

    out = model.inference(img)

    # C++ side uses ModelFeatureInfo, Python binding usually returns 1-D feature directly.
    if isinstance(out, np.ndarray):
        feat = out
    elif isinstance(out, list):
        # 兼容 list 形式，拼成一维向量
        feat = np.array(out, dtype=np.float32)
    else:
        print(f"Unsupported feature output type: {type(out)}")
        sys.exit(1)

    if feat.ndim > 1:
        feat = feat.reshape(-1)

    return feat.astype(np.float32)


def parse_args() -> Tuple[str, str, str]:
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <model_dir> <image1> <image2>")
        print(f"Example: {sys.argv[0]} ./models osnet_query.jpg osnet_gallery.jpg")
        print("ReID model: FEATURE_REID (osnet_cv181x_int8_sym.cvimodel)")
        sys.exit(1)

    model_dir = sys.argv[1]
    img1 = sys.argv[2]
    img2 = sys.argv[3]
    return model_dir, img1, img2


def main() -> int:
    model_dir, img1, img2 = parse_args()

    print("Running ReID python sample with:")
    print(f"  Model dir : {model_dir}")
    print(f"  Image 1   : {img1}")
    print(f"  Image 2   : {img2}")

    # Keep consistent with C++ / eval_reid: FEATURE_REID + osnet_cv181x_int8_sym.cvimodel
    try:
        model = nn.get_model_from_dir(nn.ModelType.FEATURE_REID, model_dir)
    except Exception as e:
        print(f"Failed to create ReID model from dir '{model_dir}': {e}")
        return -1

    if model is None:
        print("Failed to load FEATURE_REID model, please check model_dir and cvimodel file.")
        return -1

    feat1 = infer_feature(model, img1)
    feat2 = infer_feature(model, img2)

    if feat1.size == 0 or feat2.size == 0:
        print("Empty feature extracted.")
        return -1

    sim = cosine_similarity(feat1, feat2)
    print(f"ReID similarity: {sim:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

