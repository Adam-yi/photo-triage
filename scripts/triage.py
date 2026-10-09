#!/usr/bin/env python3
"""Conservative, offline photo triage. Source files are read only."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
except ImportError as exc:
    raise SystemExit(f"Missing required local package: {exc}. Install requirements.txt first.")

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

try:
    import rawpy
except ImportError:
    rawpy = None

DECODE_ERRORS = (OSError, ValueError, RuntimeError, UnidentifiedImageError)
if rawpy is not None:
    DECODE_ERRORS += (rawpy.LibRawError,)


CATEGORIES = ("推荐保留", "相似照片待挑选", "有明显问题", "无法判断")
RESULT_PREFIX = "_照片筛选结果_"
RAW_EXT = {
    ".3fr", ".arw", ".cr2", ".cr3", ".dcr", ".dng", ".erf", ".k25",
    ".kdc", ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf", ".pef",
    ".raf", ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".x3f",
}
IMAGE_EXT = RAW_EXT | {
    ".avif", ".bmp", ".gif", ".heic", ".heif", ".jfif", ".jpe",
    ".jpeg", ".jpg", ".jxl", ".png", ".psd", ".tif", ".tiff",
    ".webp",
} | set(Image.registered_extensions())
IMAGE_EXT -= {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
PREFERRED_PREVIEW = (".jpg", ".jpeg", ".heic", ".heif", ".png", ".tif", ".tiff")


@dataclass
class PhotoUnit:
    files: list[Path]
    relative: Path
    label: str = ""
    category: str = "无法判断"
    score: float = -1.0
    phash: int | None = None
    scene_hash: int | None = None
    color_hist: np.ndarray | None = None
    aspect: float = 0.0
    taken: float | None = None
    group: int = 0
    relative_best: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    pose_points: dict[int, tuple[float, float, float]] | None = None
    content_types: tuple[str, ...] = ()
    primary_content: str = "unknown"
    vision_vector: np.ndarray | None = None
    subject_vector: np.ndarray | None = None
    orientation: str = "unknown"
    content_scores: dict[str, float] = field(default_factory=dict)
    text_tokens: tuple[str, ...] = ()
    assessed: bool = False
    verified: bool = False
    initial_category: str = ""
    initial_label: str = ""
    problem_reasons: list[str] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)
    hint_reasons: list[str] = field(default_factory=list)
    face_observations: list[FaceInfo] = field(default_factory=list)
    subject_bounds: tuple[float, float, float, float] | None = None
    obstruction_edges: tuple[str, ...] = ()
    second_pass_done: bool = False
    second_pass_primary: str = "unknown"
    second_pass_metrics: dict[str, float] = field(default_factory=dict)
    second_pass_vector: np.ndarray | None = None


@dataclass
class FaceInfo:
    focus: float
    bright: float
    dark: float
    eye_min: float | None
    eye_max: float | None
    mouth: float | None
    cropped: bool


@dataclass
class BackPortrait:
    chroma: np.ndarray
    pose_family: str


@dataclass
class FullFrameReview:
    primary: str
    scores: dict[str, float]
    faces: list[FaceInfo]
    metrics: dict[str, float]
    subject_vector: np.ndarray | None = None
    subject_bounds: tuple[float, float, float, float] | None = None
    text_tokens: tuple[str, ...] = ()
    full_view_vector: np.ndarray | None = None


CONTENT_NAMES = {
    "person": "人物", "pet": "宠物", "animal": "动物", "landscape": "自然风景",
    "plants": "植物", "food": "食物饮品", "architecture": "城市建筑",
    "indoor": "室内", "vehicle": "交通工具", "object": "物品细节",
    "text": "文字信息", "art": "艺术展品", "sky": "天空", "event": "活动表演",
}

# Labels can overlap. The primary subject selects the comparison policy; a
# secondary background tag must never be enough to merge two photos.
CONTENT_FAMILIES = {
    "person": "person", "pet": "animal", "animal": "animal",
    "landscape": "scenery", "sky": "scenery", "plants": "plants",
    "food": "food", "architecture": "architecture", "indoor": "architecture",
    "vehicle": "vehicle", "object": "object", "text": "text",
    "art": "art", "event": "event",
}


class LocalVision:
    """Bundled CPU model: no photo or embedding leaves this computer."""

    def __init__(self) -> None:
        import onnxruntime as ort

        model_dir = Path(__file__).resolve().parents[1] / "models"
        metadata = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
        self.session = ort.InferenceSession(
            str(model_dir / metadata["model_file"]), providers=["CPUExecutionProvider"])
        with np.load(model_dir / "labels.npz") as labels:
            self.names = [str(name) for name in labels["names"]]
            self.text_vectors = labels["vectors"]

    @staticmethod
    def embed_input(image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        scale = 224 / min(image.size)
        image = image.resize(tuple(round(side * scale) for side in image.size),
                             Image.Resampling.BICUBIC)
        width, height = image.size
        image = image.crop(((width - 224) // 2, (height - 224) // 2,
                            (width + 224) // 2, (height + 224) // 2))
        values = np.asarray(image, dtype=np.float32) / 255.0
        values = (values - np.asarray([.48145466, .4578275, .40821073], dtype=np.float32))
        values /= np.asarray([.26862954, .26130258, .27577711], dtype=np.float32)
        return values.transpose(2, 0, 1)[None]

    def embed(self, image: Image.Image) -> np.ndarray:
        vector = self.session.run(None, {"pixel_values": self.embed_input(image)})[0][0]
        return vector / max(float(np.linalg.norm(vector)), 1e-8)

    def scores(self, vector: np.ndarray) -> dict[str, float]:
        values = vector @ self.text_vectors.T
        return dict(zip(self.names, map(float, values)))


@lru_cache(maxsize=1)
def local_vision() -> LocalVision | None:
    try:
        return LocalVision()
    except (ImportError, OSError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"Local content model unavailable: {exc}", file=sys.stderr)
        return None


def subject_crop(image: Image.Image) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    """Only accept a plausible upright person component; segmentation alone is noisy."""
    try:
        import cv2

        segmenter, _ = back_portrait_models()
        sample = image.copy()
        sample.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        result = segmenter.process(np.asarray(sample.convert("RGB")))
        if result.segmentation_mask is None:
            return None
        mask = np.asarray(result.segmentation_mask > .55, dtype=np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        candidates = [tuple(map(int, item)) for item in stats[1:]
                      if item[4] >= .01 * mask.size
                      and item[4] <= .4 * mask.size
                      and item[2] / sample.width <= .55
                      and item[3] / sample.height >= .25]
        if not candidates:
            return None
        x, y, width, height, _ = max(candidates, key=lambda item: item[4])
        pad = round(.2 * max(width, height))
        crop = sample.crop((max(0, x - pad), max(0, y - pad),
                            min(sample.width, x + width + pad),
                            min(sample.height, y + height + pad)))
        bounds = (x / sample.width, y / sample.height,
                  (x + width) / sample.width, (y + height) / sample.height)
        return crop, bounds
    except (ImportError, *DECODE_ERRORS):
        return None


def face_visible_in_crop(image: Image.Image) -> bool:
    try:
        detector, _ = expression_models()
        detections = detector.process(np.asarray(image.convert("RGB"))).detections or []
        return any(d.score[0] >= .65 for d in detections)
    except (ImportError, *DECODE_ERRORS):
        return False


def describe_content(unit: PhotoUnit, image: Image.Image) -> None:
    model = local_vision()
    if model is None:
        return
    unit.vision_vector = model.embed(image)
    scores = model.scores(unit.vision_vector)
    unit.content_scores = scores
    primary = max(CONTENT_NAMES, key=lambda name: scores[name])
    tags = [primary] if scores[primary] >= .22 else []
    unit.primary_content = primary if tags else "unknown"
    for name in CONTENT_NAMES:
        if (name != primary and scores[name] >= .22
                and scores[name] >= scores[primary] - .025):
            tags.append(name)
    plausible_mixed_person = (primary in {"pet", "animal"}
                              and scores["person"] >= scores[primary] - .04)
    if (primary not in {"pet", "animal", "food"} or plausible_mixed_person) and (
            scores["person"] >= max(scores[name] for name in CONTENT_NAMES) - .045):
        result = subject_crop(image)
        if result is not None:
            crop, bounds = result
            vector = model.embed(crop)
            crop_scores = model.scores(vector)
            person_crop = (crop_scores["person"] >=
                           max(crop_scores[name] for name in CONTENT_NAMES) - .02)
            if person_crop or plausible_mixed_person:
                unit.subject_bounds = bounds
                if person_crop:
                    unit.subject_vector = vector
                tags = ["person", *tags]
                unit.primary_content = "person"
                front = crop_scores["person_front"]
                back = crop_scores["person_back"]
                if front - back > .006:
                    unit.orientation = "front"
                elif back - front > .006:
                    unit.orientation = "back"
                if face_visible_in_crop(crop):
                    unit.orientation = "front"
                elif unit.orientation == "unknown" or front - back < .012:
                    path = next((p for p in unit.files
                                 if p.suffix.lower() in PREFERRED_PREVIEW), unit.files[0])
                    if back_portrait_profile(path) is not None:
                        unit.orientation = "back"
    unit.content_types = tuple(dict.fromkeys(tags[:4]))


def comparison_family(unit: PhotoUnit) -> str:
    primary = unit.primary_content
    if primary == "unknown" and unit.content_types:
        primary = unit.content_types[0]
    return CONTENT_FAMILIES.get(primary, "unknown")


@lru_cache(maxsize=1)
def local_ocr() -> object | None:
    try:
        from rapidocr import RapidOCR
        return RapidOCR()
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Local text recognition unavailable; text photos use exact copies only: {exc}",
              file=sys.stderr)
        return None


def extract_text(unit: PhotoUnit, image: Image.Image) -> None:
    engine = local_ocr()
    if engine is None:
        return
    try:
        result = engine(np.asarray(image.convert("RGB")))
        tokens: list[str] = []
        scores: list[float] = []
        for raw, score in zip(result.txts or (), result.scores or ()):
            normalized = unicodedata.normalize("NFKC", raw).casefold()
            token = "".join(char for index, char in enumerate(normalized)
                            if char.isalnum() or (char == "." and index > 0
                            and index + 1 < len(normalized)
                            and normalized[index - 1].isdigit()
                            and normalized[index + 1].isdigit()))
            if len(token) >= 2 and score >= .75:
                tokens.append(token)
                scores.append(float(score))
        unit.text_tokens = tuple(tokens)
        unit.metrics["text_chars"] = float(sum(map(len, tokens)))
        unit.metrics["text_confidence"] = min(scores) if scores else 0.0
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"Text recognition failed for {unit.relative}: {exc}", file=sys.stderr)


@lru_cache(maxsize=256)
def file_digest(path: Path) -> bytes | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.digest()
    except OSError:
        return None


def identical_preview_bytes(a: PhotoUnit, b: PhotoUnit) -> bool:
    first = next((p for p in a.files if p.suffix.lower() in PREFERRED_PREVIEW), a.files[0])
    second = next((p for p in b.files if p.suffix.lower() in PREFERRED_PREVIEW), b.files[0])
    left, right = file_digest(first), file_digest(second)
    return left is not None and left == right


def text_evidence_matches(a: PhotoUnit, b: PhotoUnit) -> bool:
    if not a.text_tokens or not b.text_tokens:
        return identical_preview_bytes(a, b)
    left, right = Counter(a.text_tokens), Counter(b.text_tokens)
    numbers_a = Counter(value for token in a.text_tokens
                        for value in re.findall(r"\d+(?:\.\d+)?", token))
    numbers_b = Counter(value for token in b.text_tokens
                        for value in re.findall(r"\d+(?:\.\d+)?", token))
    if numbers_a != numbers_b:
        return False
    overlap = sum((left & right).values())
    return (overlap >= 2 and overlap / sum((left | right).values()) >= .8
            and overlap / min(sum(left.values()), sum(right.values())) >= .9)


def same_comparison_family(a: PhotoUnit, b: PhotoUnit) -> bool:
    left, right = comparison_family(a), comparison_family(b)
    return left == right and left != "unknown"


def discover(root: Path) -> list[PhotoUnit]:
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(
            d for d in dirs
            if not d.startswith(RESULT_PREFIX) and not (Path(base) / d).is_symlink()
        )
        for name in sorted(names):
            path = Path(base) / name
            if path.is_symlink() or path.suffix.lower() not in IMAGE_EXT:
                continue
            rel = path.relative_to(root)
            groups[(str(rel.parent).casefold(), path.stem.casefold())].append(path)
    return [
        PhotoUnit(sorted(paths, key=lambda p: str(p).casefold()), paths[0].relative_to(root))
        for paths in groups.values()
    ]


def open_preview(path: Path) -> Image.Image:
    if path.suffix.lower() in RAW_EXT:
        if rawpy is None:
            raise RuntimeError("缺少本机 RAW 解码组件")
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True)
        image = Image.fromarray(rgb)
    else:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image


def capture_time(path: Path) -> float | None:
    try:
        with Image.open(path) as image:
            value = image.getexif().get(36867) or image.getexif().get(306)
        if value:
            return dt.datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S").timestamp()
    except (OSError, ValueError, TypeError):
        pass
    return None


def dhash(gray: Image.Image) -> int:
    pixels = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    result = 0
    for bit in bits.flat:
        result = (result << 1) | int(bit)
    return result


def color_histogram(image: Image.Image) -> np.ndarray:
    """A compact color signature that tolerates small pose and framing changes."""
    hsv = np.asarray(image.convert("HSV").resize((128, 128)), dtype=np.uint8)
    hist, _ = np.histogramdd(
        hsv.reshape(-1, 3), bins=(8, 4, 4),
        range=((0, 256), (0, 256), (0, 256)),
    )
    return hist.ravel() / hist.sum()


def perceptual_hash(gray: Image.Image) -> int:
    """Low-frequency DCT hash, more tolerant of portrait pose shifts than dHash."""
    size = 32
    samples = np.asarray(gray.resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32)
    positions = np.arange(size, dtype=np.float32)
    basis = np.cos(np.pi * (positions[:, None]) * (positions[None, :] + 0.5) / size)
    basis *= np.sqrt(2.0 / size)
    basis[0, :] = np.sqrt(1.0 / size)
    frequencies = (basis @ samples @ basis.T)[:8, :8]
    threshold = np.median(frequencies[1:, :])
    result = 0
    for bit in (frequencies > threshold).flat:
        result = (result << 1) | int(bit)
    return result


def local_focus(gray: np.ndarray) -> tuple[float, float, float, tuple[int, int]]:
    """Find a detailed region without letting a soft background dominate the photo."""
    height, width = gray.shape
    tiles: list[tuple[float, float, int, int]] = []
    for row in range(5):
        for col in range(5):
            tile = gray[
                int(row * height / 5):int((row + 1) * height / 5),
                int(col * width / 5):int((col + 1) * width / 5),
            ]
            edges = (
                -4 * tile[1:-1, 1:-1] + tile[:-2, 1:-1] + tile[2:, 1:-1]
                + tile[1:-1, :-2] + tile[1:-1, 2:]
            )
            tiles.append((float(edges.var()), float(np.mean(np.abs(edges) > 10)), row, col))
    tiles.sort(reverse=True)
    strongest, edge_fraction, row, col = tiles[0]
    return strongest, tiles[1][0], edge_fraction, (row, col)


def lap_variance(gray: np.ndarray) -> float:
    lap = (
        -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return float(lap.var())


@lru_cache(maxsize=1)
def expression_models() -> tuple[object, object]:
    import mediapipe as mp

    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.7,
    )
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.3,
    )
    return detector, mesh


def assess_faces(image: Image.Image, gray: np.ndarray) -> list[FaceInfo]:
    """Measure all prominent faces; an undetected face is never judged bad."""
    try:
        detector, mesh = expression_models()
        rgb = np.asarray(image.convert("RGB"))
        detections = [
            d for d in (detector.process(rgb).detections or [])
            if d.score[0] >= 0.8
            and d.location_data.relative_bounding_box.width * image.width >= 50
        ]
        if not detections:
            return []
        largest = max(
            d.location_data.relative_bounding_box.width
            * d.location_data.relative_bounding_box.height for d in detections
        )
        observations: list[FaceInfo] = []
        for detection in sorted(
            detections,
            key=lambda d: d.location_data.relative_bounding_box.width
                          * d.location_data.relative_bounding_box.height,
            reverse=True,
        )[:8]:
            box = detection.location_data.relative_bounding_box
            if box.width * box.height < largest * 0.4:
                continue
            fx0 = max(0, int(box.xmin * image.width))
            fy0 = max(0, int(box.ymin * image.height))
            fx1 = min(image.width, int((box.xmin + box.width) * image.width))
            fy1 = min(image.height, int((box.ymin + box.height) * image.height))
            face_gray = gray[fy0:fy1, fx0:fx1]
            if min(face_gray.shape) < 32:
                continue
            center_x = (box.xmin + box.width / 2) * image.width
            center_y = (box.ymin + box.height / 2) * image.height
            size = 2 * max(box.width * image.width, box.height * image.height)
            x0 = max(0, int(center_x - size / 2))
            x1 = min(image.width, int(center_x + size / 2))
            y0 = max(0, int(center_y - size / 2))
            y1 = min(image.height, int(center_y + size / 2))
            crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
            faces = mesh.process(crop).multi_face_landmarks or []
            eye_min = eye_max = mouth = None
            if faces:
                points = faces[0].landmark

                def distance(i: int, j: int) -> float:
                    return float(np.hypot(points[i].x - points[j].x,
                                          points[i].y - points[j].y))

                def eye_ratio(a: int, b: int, c: int, d: int, e: int, f: int) -> float:
                    return (distance(b, f) + distance(c, e)) / (2 * distance(a, d))

                eyes = (eye_ratio(33, 160, 158, 133, 153, 144),
                        eye_ratio(362, 385, 387, 263, 373, 380))
                eye_min, eye_max = min(eyes), max(eyes)
                mouth = distance(13, 14) / distance(61, 291)
            observations.append(FaceInfo(
                focus=lap_variance(face_gray),
                bright=float(np.mean(face_gray > 245)),
                dark=float(np.mean(face_gray < 18)),
                eye_min=eye_min, eye_max=eye_max, mouth=mouth,
                cropped=(fx0 <= 2 or fy0 <= 2 or fx1 >= image.width - 2
                         or fy1 >= image.height - 2),
            ))
        return observations
    except (ImportError, OSError, RuntimeError, ValueError, FileNotFoundError,
            ZeroDivisionError):
        return []


def noise_level(gray_image: Image.Image) -> float:
    median = np.asarray(gray_image.filter(ImageFilter.MedianFilter(size=5)), dtype=np.float32)
    source = np.asarray(gray_image, dtype=np.float32)
    residual = np.abs(source - median)
    samples = []
    for row in range(0, source.shape[0] - 31, 32):
        for col in range(0, source.shape[1] - 31, 32):
            smooth = median[row:row+32, col:col+32]
            if float(smooth.std()) < 10:
                samples.append(float(np.percentile(residual[row:row+32, col:col+32], 75)))
    return float(np.percentile(samples, 75)) if len(samples) >= 4 else 0.0


def jpeg_compression(path: Path) -> float:
    if path.suffix.lower() not in {".jpg", ".jpeg", ".jpe", ".jfif"}:
        return 0.0
    try:
        with Image.open(path) as image:
            tables = image.quantization or {}
            return float(np.mean(tables.get(0, []))) if tables.get(0) else 0.0
    except DECODE_ERRORS:
        return 0.0


def obstruction_edges(image: Image.Image, gray: np.ndarray) -> tuple[str, ...]:
    height, width = gray.shape
    if lap_variance(gray[int(height*.3):int(height*.7),
                         int(width*.3):int(width*.7)]) < 45:
        return ()
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    bands = {
        "left": rgb[:, :int(width*.28)], "right": rgb[:, int(width*.72):],
        "top": rgb[:int(height*.28), :], "bottom": rgb[int(height*.72):, :],
    }
    found: list[str] = []
    for edge, band in bands.items():
        pixels = band.reshape(-1, 3)
        typical = np.median(pixels, axis=0)
        # A large featureless, warm-coloured edge region can be a finger.
        # A dark edge alone can equally be a shadow or intentional background.
        warm = typical[0] > typical[1] * 1.13 and typical[1] > typical[2] * 1.06
        uniform = float(np.mean(np.linalg.norm(pixels - typical, axis=1) < 25)) > 0.82
        if warm and uniform:
            found.append(edge)
    return tuple(found)


def possible_obstruction(image: Image.Image, gray: np.ndarray) -> bool:
    return bool(obstruction_edges(image, gray))


def face_findings(faces: list[FaceInfo], sharpness: float) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    review: list[str] = []
    for index, face in enumerate(faces, 1):
        who = f"第{index}位人物" if len(faces) > 1 else "人物"
        if face.focus < 15 and sharpness > 70:
            problems.append(f"{who}面部明显失焦")
        elif face.focus < 30 and sharpness > 70:
            review.append(f"{who}面部可能失焦")
        if face.bright > 0.45:
            problems.append(f"{who}面部明显过曝")
        elif face.dark > 0.65:
            problems.append(f"{who}面部明显欠曝")
        if face.eye_min is not None and face.eye_max is not None:
            if face.eye_max < 0.12:
                # A small eye aspect ratio cannot distinguish closed eyes from
                # looking downward, a posed shot, or a blink. Do not claim a
                # definite eye state from this measurement alone.
                if face.mouth is not None and face.mouth >= 0.17:
                    review.append(f"{who}眼部开合度较低且张嘴，需确认是否为有意表情")
            elif (face.mouth is not None and face.mouth >= 0.17
                  and face.eye_min <= 0.19
                  and face.eye_max <= 0.22
                  and face.eye_max - face.eye_min >= 0.025):
                problems.append(f"{who}疑似抓拍瞬间：张嘴且一只眼未充分睁开")
        if face.cropped:
            review.append(f"{who}面部贴近画面边缘，需确认裁切")
    return problems, review


def selection_score(sharpness: float, dark: float, bright: float,
                    faces: list[FaceInfo], noise: float) -> float:
    score = min(sharpness, 250) / 250 - dark * 0.7 - bright * 0.7
    if faces:
        score += 0.8 * min(min(f.focus for f in faces), 250) / 250
        eyes = [f.eye_min for f in faces if f.eye_min is not None]
        if eyes:
            score += 0.15 * min(min(eyes), 0.25) / 0.25
        score -= 0.5 * max(f.bright + f.dark for f in faces)
    return score - min(noise, 30) / 150


def type_selection_score(unit: PhotoUnit) -> float:
    """Prefer the detail that matters for the photo's primary purpose."""
    metrics = unit.metrics
    base = unit.score
    focus = min(metrics.get("local_focus", 0.0), 250.0) / 250.0
    sharpness = min(metrics.get("sharpness", 0.0), 250.0) / 250.0
    clipping = metrics.get("dark", 0.0) + metrics.get("bright", 0.0)
    family = comparison_family(unit)
    if family == "person":
        return base
    if family == "text":
        legibility = min(metrics.get("text_chars", 0.0), 80.0) / 80.0
        return 2.0 * focus + 0.2 * sharpness + 0.2 * legibility - clipping
    if family in {"animal", "food", "object", "art"}:
        return 1.5 * focus + 0.45 * sharpness - clipping
    if family in {"scenery", "plants", "architecture", "vehicle", "event"}:
        return 0.8 * focus + 0.7 * sharpness - 0.8 * clipping
    return base


def assess(unit: PhotoUnit) -> None:
    reasons: list[str] = []
    for path in sorted(
        unit.files,
        key=lambda p: (p.suffix.lower() not in PREFERRED_PREVIEW,
                       PREFERRED_PREVIEW.index(p.suffix.lower()) if p.suffix.lower() in PREFERRED_PREVIEW else 99),
    ):
        try:
            image = open_preview(path)
            break
        except DECODE_ERRORS as exc:
            reasons.append(f"{path.suffix.lower()}: {type(exc).__name__}")
    else:
        unit.label = "无法读取预览（" + ", ".join(reasons[:3]) + "）"
        return

    unit.aspect = image.width / image.height
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    gray_image = ImageOps.grayscale(image)
    gray = np.asarray(gray_image, dtype=np.float32)
    if min(gray.shape) < 64:
        unit.label = "图片尺寸过小，无法可靠判断"
        return

    sharpness = lap_variance(gray)
    dark = float(np.mean(gray < 18))
    bright = float(np.mean(gray > 245))
    mean = float(gray.mean())
    focus, next_focus, focus_edges, _ = local_focus(gray)
    gradient_95 = float(np.percentile(np.abs(gray[:, 1:] - gray[:, :-1]), 95))
    unit.metrics = {
        "sharpness": sharpness, "local_focus": focus, "next_focus": next_focus,
        "focus_edges": focus_edges, "gradient_95": gradient_95,
        "dark": dark, "bright": bright, "mean": mean,
    }
    unit.phash = dhash(gray_image)
    unit.scene_hash = perceptual_hash(gray_image)
    unit.color_hist = color_histogram(image)
    unit.taken = capture_time(path)
    describe_content(unit, image)

    faces = assess_faces(image, gray)
    if (faces and unit.primary_content not in {"pet", "animal", "food"}
            and unit.content_scores.get("person", 0.0) >=
            max((unit.content_scores.get(name, 0.0) for name in CONTENT_NAMES), default=0.0) - .045):
        unit.primary_content = "person"
        unit.content_types = tuple(dict.fromkeys(("person", *unit.content_types)))
        unit.orientation = "front"
    person_faces = faces if comparison_family(unit) == "person" else []
    unit.face_observations = person_faces
    noise = noise_level(gray_image)
    if comparison_family(unit) == "text":
        extract_text(unit, image)
    compression = jpeg_compression(path)
    unit.metrics.update({"face_count": float(len(faces)), "noise": noise,
                         "jpeg_quantization": compression})
    if faces:
        unit.metrics.update({
            "face_focus_min": min(f.focus for f in faces),
            "face_bright_max": max(f.bright for f in faces),
            "face_dark_max": max(f.dark for f in faces),
        })

    problems: list[str] = []
    review: list[str] = []
    contrast = float(gray.std())
    blur_candidate = sharpness < 12 and contrast > 15
    focused_region = focus >= 40 and focus_edges >= 0.03
    strong_blur = (
        blur_candidate and not focused_region and sharpness < 8
        and focus < 18 and gradient_95 >= 5 and contrast >= 35
    )
    # A verified person-shaped subject plus a uniformly featureless frame is
    # stronger evidence than the gradient of sky/skin alone.
    if (comparison_family(unit) == "person" and unit.subject_bounds is not None
            and sharpness < 8 and focus < 22 and next_focus < 20
            and contrast > 25 and gradient_95 >= 3):
        strong_blur = True
    if strong_blur:
        problems.append("严重模糊")
    dark_excess = dark > 0.78 or (mean < 35 and dark > 0.55)
    bright_excess = bright > 0.78 or (mean > 230 and bright > 0.55)
    if dark > .98 and mean < 10:
        problems.append("严重欠曝")
    elif comparison_family(unit) in {"scenery", "event"}:
        if dark_excess or bright_excess:
            review.append("极端明暗可能是有意拍摄，需人工确认")
    else:
        if dark_excess:
            problems.append("严重欠曝")
        if bright_excess:
            problems.append("严重过曝")
    # Broad clipping, weak edge detail, and an animal as primary subject are
    # independent clues that a bright animal is losing usable fur detail.
    if (comparison_family(unit) == "animal" and bright > .35
            and mean > 180 and focus_edges < .10 and gradient_95 < 6):
        problems.append("动物主体大面积高光细节丢失")

    if person_faces:
        face_problems, face_review = face_findings(person_faces, sharpness)
        problems.extend(face_problems)
        review.extend(face_review)
        face_focus = min(face.focus for face in person_faces)
        if focus > 1500:
            relative_focus = face_focus / focus
            if relative_focus < .05 and face_focus < 60:
                problems.append("人物面部相对背景明显失焦")
            elif ((relative_focus < .05 and face_focus < 80)
                  or .05 <= relative_focus < .20):
                review.append("背景清晰但人物面部较柔和，需确认主体对焦")
    else:
        middle = gray[int(gray.shape[0]*.3):int(gray.shape[0]*.7),
                      int(gray.shape[1]*.3):int(gray.shape[1]*.7)]
        scene_with_depth = comparison_family(unit) in {"scenery", "plants", "architecture"}
        lower_left = gray[gray.shape[0]//2:, :gray.shape[1]//2]
        lower_right = gray[gray.shape[0]//2:, gray.shape[1]//2:]
        other_planes_clear = (lap_variance(lower_left) > 80
                              and lap_variance(lower_right) > 80)
        if (lap_variance(middle) < 18 and focus > 80 and contrast > 25
                and not (scene_with_depth and other_planes_clear)):
            review.append("画面中心较柔和、其他区域清晰，需确认主体对焦")

    if noise >= 24 and contrast < 65:
        problems.append("严重噪点")
    elif noise >= 18 and contrast < 75:
        review.append("噪点较多，建议复核")
    if compression >= 180:
        problems.append("严重 JPEG 压缩")
    elif compression >= 95:
        review.append("JPEG 压缩较重，建议复核")
    unit.obstruction_edges = obstruction_edges(image, gray)
    if unit.obstruction_edges:
        review.append("边缘疑似大面积遮挡，建议复核")
    if blur_candidate and not focused_region:
        review.append("画面较柔和，无法区分失焦与低纹理场景")

    unit.score = selection_score(sharpness, dark, bright, person_faces, noise)
    unit.score = type_selection_score(unit)
    if problems:
        unit.category = "有明显问题"
        unit.label = "、".join(problems)
    elif review:
        unit.category = "无法判断"
        unit.label = "、".join(review)
    else:
        unit.category = "推荐保留"
        low_eye_opening = any(
            face.eye_max is not None and face.eye_max < 0.12
            and face.mouth is not None and face.mouth < 0.17
            for face in person_faces
        )
        unit.label = ("眼部开合度较低；可能是低头注视或有意闭眼，建议结合相邻照片复选"
                      if low_eye_opening else
                      "局部清晰、背景虚化" if blur_candidate else "无明显技术问题")
    unit.assessed = True
    unit.initial_category = unit.category
    unit.initial_label = unit.label
    unit.problem_reasons = problems
    unit.review_reasons = review


def review_full_image(unit: PhotoUnit) -> FullFrameReview | None:
    """Read the source again and inspect a full-frame view at a second scale.

    The first content pass uses CLIP's centered square crop. This pass fits the
    entire frame into a square, so edge subjects and the surrounding scene are
    visible. It also measures local regions from newly decoded pixels rather
    than reusing the first pass's quality scores.
    """
    model = local_vision()
    if model is None:
        return None
    for path in sorted(
        unit.files,
        key=lambda p: (p.suffix.lower() not in PREFERRED_PREVIEW,
                       PREFERRED_PREVIEW.index(p.suffix.lower())
                       if p.suffix.lower() in PREFERRED_PREVIEW else 99),
    ):
        try:
            image = open_preview(path)
            break
        except DECODE_ERRORS:
            continue
    else:
        return None

    image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    if min(image.size) < 64:
        return None
    rgb = image.convert("RGB")
    fitted = ImageOps.contain(rgb, (224, 224), Image.Resampling.LANCZOS)
    full_view = Image.new("RGB", (224, 224), (127, 127, 127))
    full_view.paste(fitted, ((224 - fitted.width) // 2, (224 - fitted.height) // 2))
    full_view_vector = model.embed(full_view)
    scores = model.scores(full_view_vector)
    primary = max(CONTENT_NAMES, key=lambda name: scores[name])
    if scores[primary] < .22:
        primary = "unknown"

    gray = np.asarray(ImageOps.grayscale(rgb), dtype=np.float32)
    height, width = gray.shape
    zones = [gray[int(row * height / 3):int((row + 1) * height / 3),
                  int(col * width / 3):int((col + 1) * width / 3)]
             for row in range(3) for col in range(3)]
    zone_focus = [lap_variance(zone) for zone in zones]
    zone_bright = [float(np.mean(zone > 245)) for zone in zones]
    zone_dark = [float(np.mean(zone < 18)) for zone in zones]
    focus, second_focus, focus_edges, _ = local_focus(gray)
    middle = gray[int(height * .2):int(height * .8),
                  int(width * .2):int(width * .8)]
    metrics = {
        "sharpness": lap_variance(gray), "local_focus": focus,
        "next_focus": second_focus, "focus_edges": focus_edges,
        "contrast": float(gray.std()),
        "bright": float(np.mean(gray > 245)),
        "dark": float(np.mean(gray < 18)),
        "center_focus": lap_variance(middle),
        "center_bright": float(np.mean(middle > 245)),
        "center_dark": float(np.mean(middle < 18)),
        "zones_clear": float(sum(value >= 80 for value in zone_focus)),
        "zones_soft": float(sum(value < 18 for value in zone_focus)),
        "zones_bright": float(sum(value > .55 for value in zone_bright)),
        "zones_dark": float(sum(value > .65 for value in zone_dark)),
    }

    # Re-detect the actual human region at a different scale. Face or person
    # evidence can override a background-driven full-frame content label.
    faces = assess_faces(rgb, gray)
    person_plausible = (bool(faces) or
                        scores["person"] >= max(scores[name] for name in CONTENT_NAMES) - .06)
    subject_vector = None
    subject_bounds = None
    if person_plausible:
        subject = subject_crop(rgb)
        if subject is not None:
            crop, subject_bounds = subject
            subject_vector = model.embed(crop)
            crop_scores = model.scores(subject_vector)
            if crop_scores["person"] < max(crop_scores[name] for name in CONTENT_NAMES) - .02:
                subject_vector = None
                subject_bounds = None
            else:
                subject_gray = np.asarray(ImageOps.grayscale(crop), dtype=np.float32)
                metrics["subject_focus"] = lap_variance(subject_gray)
                metrics["subject_bright"] = float(np.mean(subject_gray > 245))
                metrics["subject_dark"] = float(np.mean(subject_gray < 18))
    if not person_plausible:
        target = unit.primary_content if unit.primary_content in {
            "pet", "animal", "food", "object", "art", "text", "vehicle"
        } else primary
        if target in {"pet", "animal", "food", "object", "art", "text", "vehicle"}:
            candidates = []
            for top in (0.0, .35):
                for left in (0.0, .35):
                    x0, y0 = int(left * width), int(top * height)
                    x1, y1 = int((left + .65) * width), int((top + .65) * height)
                    patch = rgb.crop((x0, y0, x1, y1))
                    patch_scores = model.scores(model.embed(patch))
                    candidates.append((patch_scores[target], x0, y0, x1, y1))
            match, x0, y0, x1, y1 = max(candidates)
            subject_gray = gray[y0:y1, x0:x1]
            metrics["subject_match"] = float(match)
            metrics["subject_focus"] = lap_variance(subject_gray)
            metrics["subject_bright"] = float(np.mean(subject_gray > 245))
            metrics["subject_dark"] = float(np.mean(subject_gray < 18))
    second_text: tuple[str, ...] = ()
    if primary == "text" or unit.primary_content == "text":
        text_probe = PhotoUnit([], Path("second-pass"))
        extract_text(text_probe, rgb)
        second_text = text_probe.text_tokens
    return FullFrameReview(primary, scores, faces, metrics,
                           subject_vector, subject_bounds, second_text,
                           full_view_vector)


def independent_review_all(units: list[PhotoUnit]) -> bool:
    """Verify every readable photo from fresh pixels before final grouping.

    Only corroborated content changes alter the comparison policy. Conflicting
    weak signals go to human review instead of silently changing the subject.
    """
    content_changed = False
    for unit in units:
        if not unit.assessed or unit.second_pass_done:
            continue
        observation = review_full_image(unit)
        unit.second_pass_done = True
        if observation is None:
            unit.review_reasons.append("第二遍无法重新读取完整画面，需人工复核")
            continue
        unit.second_pass_primary = observation.primary
        unit.second_pass_metrics = observation.metrics
        unit.second_pass_vector = observation.full_view_vector
        scores = observation.scores
        ordered = sorted((scores[name] for name in CONTENT_NAMES), reverse=True)
        gap = ordered[0] - ordered[1]
        first_person_evidence = bool(unit.face_observations or unit.subject_bounds)
        second_person_evidence = bool(observation.faces or observation.subject_vector is not None)
        candidate = observation.primary
        if (second_person_evidence and scores["person"] >= ordered[0] - .05):
            candidate = "person"
        confident_change = (
            candidate != "unknown" and candidate != unit.primary_content
            and ((candidate == "person" and second_person_evidence and not first_person_evidence)
                 or (candidate != "person" and gap >= .035
                     and not first_person_evidence and not second_person_evidence))
        )
        if confident_change:
            unit.primary_content = candidate
            unit.content_types = tuple(dict.fromkeys((candidate, *unit.content_types)))[:4]
            if candidate == "person":
                unit.face_observations = observation.faces
                unit.subject_bounds = observation.subject_bounds
                unit.subject_vector = observation.subject_vector
                unit.orientation = "front" if observation.faces else unit.orientation
            else:
                unit.face_observations = []
                unit.subject_bounds = None
                unit.subject_vector = None
                unit.orientation = "unknown"
            content_changed = True
        elif (candidate != "unknown" and candidate != unit.primary_content
              and gap >= .025 and not (first_person_evidence and second_person_evidence)):
            unit.review_reasons.append("两次完整画面识别的主要对象不一致，需人工确认")

        metrics = observation.metrics
        family = comparison_family(unit)
        if (metrics["sharpness"] < 8 and metrics["local_focus"] < 22
                and metrics["next_focus"] < 20 and metrics["contrast"] > 25
                and metrics["zones_soft"] >= 7):
            unit.problem_reasons.append("第二遍复核确认画面大范围严重模糊")
        if (family == "animal" and metrics["bright"] > .30
                and metrics.get("subject_match", 0) >= .25
                and metrics.get("subject_bright", 0) > .40
                and metrics.get("subject_focus", float("inf")) < 100):
            unit.problem_reasons.append("第二遍复核确认动物主体高光细节大面积丢失")
        if family == "person" and observation.faces:
            face_focus = min(face.focus for face in observation.faces)
            if metrics["local_focus"] > 1500:
                relative_focus = face_focus / metrics["local_focus"]
                if (metrics["local_focus"] > 2000
                        and relative_focus < .04 and face_focus < 60):
                    unit.problem_reasons.append("第二遍复核确认人物面部相对背景明显失焦")
                elif ((relative_focus < .04 and face_focus < 80)
                      or .04 <= relative_focus < .16):
                    unit.review_reasons.append("第二遍复核发现背景清晰但人物面部较柔和")
        if (family in {"scenery", "plants", "architecture"}
                and metrics["zones_clear"] >= 3):
            center_warning = "画面中心较柔和、其他区域清晰，需确认主体对焦"
            if center_warning in unit.review_reasons:
                unit.review_reasons.remove(center_warning)
                unit.hint_reasons.append("多个景深区域保有细节，中景柔和不单独视为失焦")
        if family == "text" and observation.text_tokens:
            if not unit.text_tokens:
                unit.text_tokens = observation.text_tokens
                content_changed = True
            else:
                old_numbers = set(re.findall(r"\d+(?:\.\d+)?", " ".join(unit.text_tokens)))
                new_numbers = set(re.findall(r"\d+(?:\.\d+)?", " ".join(observation.text_tokens)))
                if old_numbers != new_numbers:
                    unit.review_reasons.append("第二遍文字识别与初判数字不一致，需人工核对")
        if confident_change:
            base = selection_score(unit.metrics.get("sharpness", 0),
                                   unit.metrics.get("dark", 0),
                                   unit.metrics.get("bright", 0),
                                   unit.face_observations,
                                   unit.metrics.get("noise", 0))
            unit.score = base
            unit.score = type_selection_score(unit)
    return content_changed


def verify_assessments(units: list[PhotoUnit]) -> None:
    """Reconcile preliminary labels with subject evidence and capture context.

    This is a second decision pass, not a repeat of the triggering threshold.
    A weak region-level signal cannot overrule a clearly visible main subject.
    """
    groups: dict[int, list[PhotoUnit]] = defaultdict(list)
    for unit in units:
        if unit.group:
            groups[unit.group].append(unit)

    for unit in units:
        if not unit.assessed:
            continue  # unreadable files and too-small previews retain their reason
        problems = list(unit.problem_reasons)
        review = list(unit.review_reasons)
        hints: list[str] = list(unit.hint_reasons)
        person = comparison_family(unit) == "person"
        clear_faces = bool(unit.face_observations) and all(
            face.focus >= 40 and face.bright < .25 and face.dark < .35
            and not face.cropped for face in unit.face_observations
        )
        clear_person = person and clear_faces

        # Expression and pose timing are relative to the burst. A single low
        # eye-opening measurement is ambiguous; a clear neighboring frame of
        # the same view supplies the missing counterfactual. Repeated low-eye
        # frames instead support an intentional expression.
        close_peers = [peer for peer in units if peer is not unit and person
                       and comparison_family(peer) == "person"
                       and unit.taken is not None and peer.taken is not None
                       and abs(unit.taken - peer.taken) <= 12
                       and unit.relative.parent == peer.relative.parent
                       and unit.orientation == peer.orientation
                       and unit.vision_vector is not None
                       and peer.vision_vector is not None
                       and float(unit.vision_vector @ peer.vision_vector) >= .90
                       and unit.scene_hash is not None
                       and peer.scene_hash is not None
                       and (unit.scene_hash ^ peer.scene_hash).bit_count() <= 23]
        eyes = [face.eye_max for face in unit.face_observations
                if face.eye_max is not None]
        if eyes and min(eyes) < .22:
            nearby_low = any(
                peer is not unit and peer.taken is not None
                and unit.relative.parent == peer.relative.parent
                and abs(unit.taken - peer.taken) <= 20
                and unit.vision_vector is not None and peer.vision_vector is not None
                and float(unit.vision_vector @ peer.vision_vector) >= .84
                and any(face.eye_max is not None and face.eye_max < .16
                        for face in peer.face_observations)
                for peer in units
            )
            repeated_low = any(
                peer is not unit and peer.taken is not None
                and unit.relative.parent == peer.relative.parent
                and abs(unit.taken - peer.taken) <= 5
                and unit.vision_vector is not None and peer.vision_vector is not None
                and float(unit.vision_vector @ peer.vision_vector) >= .84
                and any(face.eye_max is not None and face.eye_max < .16
                        for face in peer.face_observations)
                for peer in units
            )
            clear_neighbors = [peer for peer in close_peers
                               if any(face.eye_min is not None and face.eye_min >= .22
                                      and face.eye_max is not None and face.eye_max >= .27
                                      for face in peer.face_observations)]
            if clear_neighbors and not nearby_low:
                problems.append("同一连拍中眼睛开合明显异常，疑似眨眼瞬间")
            elif clear_neighbors and nearby_low and not repeated_low:
                review.append("同场景既有低眼位又有睁眼照片，需确认低头注视或抓拍瞬间")
            elif clear_neighbors and unit.pose_points is not None:
                hand_motion = max((float(np.hypot(
                    unit.pose_points[wrist][0] - peer.pose_points[wrist][0],
                    unit.pose_points[wrist][1] - peer.pose_points[wrist][1]))
                    for peer in clear_neighbors if peer.pose_points is not None
                    for wrist in (15, 16)
                    if wrist in unit.pose_points and wrist in peer.pose_points
                    and min(unit.pose_points[wrist][2], peer.pose_points[wrist][2]) >= .55),
                    default=0.0)
                if hand_motion > 1.2 and min(eyes) < .16:
                    problems.append("连拍中眼部开合较低且手部大幅移动，疑似动作未完成")
                elif hand_motion > .9:
                        review.append("眼部开合较低且手部动作变化快，需确认是否抓拍中途")

        # A lone hand position between a pair of stable neighboring poses is
        # a timing warning, not a claim that the pose is aesthetically bad.
        if person and unit.pose_points is not None and not problems:
            pose_peers = [peer for peer in units if peer is not unit
                          and comparison_family(peer) == "person"
                          and peer.pose_points is not None
                          and unit.taken is not None and peer.taken is not None
                          and 0 < peer.taken - unit.taken <= 30
                          and unit.relative.parent == peer.relative.parent
                          and unit.orientation == peer.orientation
                          and unit.vision_vector is not None
                          and peer.vision_vector is not None
                          and float(unit.vision_vector @ peer.vision_vector) >= .88]
            for index, left in enumerate(pose_peers):
                for right in pose_peers[index + 1:]:
                    if not same_action(left, right):
                        continue
                    def hand_shift(peer: PhotoUnit) -> list[float]:
                        return [float(np.hypot(unit.pose_points[w][0] - peer.pose_points[w][0],
                                               unit.pose_points[w][1] - peer.pose_points[w][1]))
                                for w in (15, 16)
                                if w in unit.pose_points and w in peer.pose_points
                                and min(unit.pose_points[w][2], peer.pose_points[w][2]) >= .55]
                    left_shift, right_shift = hand_shift(left), hand_shift(right)
                    if (len(left_shift) == len(right_shift) == 2
                            and min(left_shift) > .45 and min(right_shift) > .45):
                        review.append("双手姿态与相邻稳定照片不同，需确认动作是否完成")
                        break
                if "双手姿态与相邻稳定照片不同，需确认动作是否完成" in review:
                    break
        if (person and not unit.face_observations and unit.pose_points is None
                and unit.taken is not None and not problems):
            rapid_turn = any(
                peer is not unit and peer.taken is not None
                and unit.relative.parent == peer.relative.parent
                and abs(unit.taken - peer.taken) <= 2
                and {unit.orientation, peer.orientation} == {"front", "back"}
                and unit.vision_vector is not None and peer.vision_vector is not None
                and float(unit.vision_vector @ peer.vision_vector) >= .95
                and unit.scene_hash is not None and peer.scene_hash is not None
                and (unit.scene_hash ^ peer.scene_hash).bit_count() <= 8
                for peer in units
            )
            if rapid_turn:
                review.append("相邻照片间人物快速转身且面部不可见，需确认动作是否完成")

        # Whole-frame exposure/softness can be caused by dark surroundings or
        # shallow depth of field. Check the actual portrait subject separately.
        if clear_person:
            for reason in ("严重欠曝", "严重过曝"):
                if reason in problems:
                    problems.remove(reason)
                    review.append("背景明暗极端但人脸可辨，需确认身体和场景是否受影响")
            for reason in ("严重模糊",):
                if reason in problems:
                    problems.remove(reason)
                    hints.append("人物面部清晰，背景柔和未影响关键主体")
            center_soft = "画面中心较柔和，无法区分失焦与低纹理场景"
            if center_soft in review:
                review.remove(center_soft)
                hints.append("人物面部清晰，画面中心柔和未影响关键主体")

            bounds = unit.subject_bounds
            edge_clear = bounds is not None and all({
                "left": bounds[0] >= .28,
                "right": bounds[2] <= .72,
                "top": bounds[1] >= .28,
                "bottom": bounds[3] <= .72,
            }[edge] for edge in unit.obstruction_edges)
            edge_warning = "边缘疑似大面积遮挡，建议复核"
            if edge_clear and edge_warning in review:
                review.remove(edge_warning)
                hints.append("可疑边缘未覆盖人物主体")

        # Moderate artifact scores are observations, not evidence that the
        # photograph cannot be judged. Keep a note for the user's comparison.
        if unit.metrics.get("local_focus", 0.0) >= 40:
            for reason in ("噪点较多，建议复核", "JPEG 压缩较重，建议复核"):
                if reason in review:
                    review.remove(reason)
                    hints.append(reason)

        # Repetition within a trusted same-action group is counterevidence to
        # a one-off bad expression. It does not prove the expression is good.
        transient = "人物疑似抓拍瞬间：张嘴且一只眼未充分睁开"
        if transient in problems and unit.group:
            similar_expression = sum(
                any(face.mouth is not None and face.mouth >= .17
                    and face.eye_min is not None and face.eye_max is not None
                    and face.eye_min <= .19
                    and face.eye_max - face.eye_min >= .025
                    for face in peer.face_observations)
                for peer in groups[unit.group] if peer is not unit
            )
            if similar_expression >= 2:
                problems.remove(transient)
                review.append("同一表情在相似组中反复出现，需确认是否为有意动作")

        problems = list(dict.fromkeys(problems))
        review = list(dict.fromkeys(review))
        unit.problem_reasons = problems
        unit.review_reasons = review
        unit.hint_reasons = list(dict.fromkeys(hints))
        unit.verified = True
        if problems:
            unit.category = "有明显问题"
            unit.label = "、".join(problems)
        elif review:
            unit.category = "无法判断"
            unit.label = "、".join(review)
        else:
            unit.category = "推荐保留"
            if hints:
                unit.label = "已复核：" + "；".join(unit.hint_reasons)
            elif unit.initial_category != "推荐保留":
                unit.label = "已复核：未见足以改变分类的明显问题"


def adjacent_capture(a: PhotoUnit, b: PhotoUnit) -> bool:
    if a.relative.parent != b.relative.parent:
        return False
    if a.taken is not None and b.taken is not None and abs(a.taken - b.taken) > 300:
        return False
    first = re.fullmatch(r"(.*?)(\d+)", a.relative.stem, re.IGNORECASE)
    second = re.fullmatch(r"(.*?)(\d+)", b.relative.stem, re.IGNORECASE)
    if first and second and first.group(1).casefold() == second.group(1).casefold():
        if len(first.group(2)) == len(second.group(2)):
            if abs(int(first.group(2)) - int(second.group(2))) <= 2:
                return True
    return bool(a.taken is not None and b.taken is not None and abs(a.taken - b.taken) <= 45)


@lru_cache(maxsize=256)
def local_features(path: Path) -> tuple[object, object] | None:
    """Local detail points survive reframing better than a whole-image hash."""
    try:
        import cv2

        image = open_preview(path)
        image.thumbnail((800, 800), Image.Resampling.LANCZOS)
        gray = np.asarray(image.convert("L"))
        keypoints, descriptors = cv2.ORB_create(nfeatures=1000).detectAndCompute(gray, None)
        if descriptors is None:
            return None
        return keypoints, descriptors
    except (ImportError, *DECODE_ERRORS):
        return None


def local_correspondence(a: PhotoUnit, b: PhotoUnit) -> bool:
    """Require many geometrically consistent details, not just a shared background color."""
    try:
        import cv2

        path_a = next((p for p in a.files if p.suffix.lower() in PREFERRED_PREVIEW), a.files[0])
        path_b = next((p for p in b.files if p.suffix.lower() in PREFERRED_PREVIEW), b.files[0])
        features_a, features_b = local_features(path_a), local_features(path_b)
        if features_a is None or features_b is None:
            return False
        key_a, desc_a = features_a
        key_b, desc_b = features_b
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
        good = [pair[0] for pair in pairs
                if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        if len(good) < 20:
            return False
        points_a = np.float32([key_a[match.queryIdx].pt for match in good])
        points_b = np.float32([key_b[match.trainIdx].pt for match in good])
        _, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, 5.0)
        return mask is not None and int(mask.sum()) >= 20 and float(mask.mean()) >= 0.5
    except ImportError:
        return False
    except (cv2.error, *DECODE_ERRORS):
        return False


@lru_cache(maxsize=1)
def back_portrait_models() -> tuple[object, object]:
    import mediapipe as mp

    return (
        mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1),
        mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.2,
        ),
    )


@lru_cache(maxsize=256)
def back_portrait_profile(path: Path) -> BackPortrait | None:
    """Describe an upright back-view subject when arm landmarks cannot be trusted."""
    try:
        import cv2

        segmenter, face_mesh = back_portrait_models()
        image = open_preview(path)
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        rgb = np.asarray(image.convert("RGB"))
        result = segmenter.process(rgb)
        if result.segmentation_mask is None:
            return None
        mask = np.asarray(result.segmentation_mask > 0.55, dtype=np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count < 2:
            return None
        component_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, width, height, area = map(int, stats[component_id])
        if (area < 0.015 * mask.size or area < 0.65 * int(mask.sum())
                or height < 0.25 * image.height or height / max(width, 1) < 1.5):
            return None
        top = max(0, y - 30)
        bottom = min(image.height, y + max(1, round(height * 0.45)))
        left = max(0, x - 30)
        right = min(image.width, x + width + 30)
        head = Image.fromarray(rgb[top:bottom, left:right]).resize(
            (512, 512), Image.Resampling.LANCZOS)
        if face_mesh.process(np.asarray(head)).multi_face_landmarks:
            return None
        subject_mask = np.asarray(labels == component_id, dtype=np.uint8)

        def band_width(start: float, stop: float) -> float:
            band = subject_mask[y + round(height * start):y + round(height * stop), x:x + width]
            _, occupied_x = np.nonzero(band)
            return ((int(occupied_x.max()) - int(occupied_x.min()) + 1) / height
                    if len(occupied_x) else 0.0)

        upper_width = band_width(0.0, 0.25)
        shoulder_width = band_width(0.25, 0.5)
        if upper_width >= 0.56:
            pose_family = "both_arms_raised"
        elif shoulder_width >= 0.60:
            pose_family = "hands_on_hips"
        else:
            pose_family = "standing"
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        chroma = cv2.calcHist([hsv], [0, 1], subject_mask,
                              [12, 4], [0, 180, 0, 256]).flatten()
        return BackPortrait(chroma / max(float(chroma.sum()), 1.0), pose_family)
    except ImportError:
        return None
    except (cv2.error, *DECODE_ERRORS):
        return None


def back_portrait_match(a: PhotoUnit, b: PhotoUnit) -> bool:
    """Match a short series of upright back-view portraits across camera moves."""
    if (a.relative.parent != b.relative.parent
            or a.taken is None or b.taken is None
            or abs(a.taken - b.taken) > 120
            or a.metrics.get("face_count", 0) > 0
            or b.metrics.get("face_count", 0) > 0
            or a.scene_hash is None or b.scene_hash is None
            or a.color_hist is None or b.color_hist is None):
        return False
    first = re.fullmatch(r"(.*?)(\d+)", a.relative.stem, re.IGNORECASE)
    second = re.fullmatch(r"(.*?)(\d+)", b.relative.stem, re.IGNORECASE)
    if (not first or not second or first.group(1).casefold() != second.group(1).casefold()
            or len(first.group(2)) != len(second.group(2))
            or abs(int(first.group(2)) - int(second.group(2))) > 3
            or (a.scene_hash ^ b.scene_hash).bit_count() > 35
            or float(np.minimum(a.color_hist, b.color_hist).sum()) < 0.25):
        return False
    path_a = next((p for p in a.files if p.suffix.lower() in PREFERRED_PREVIEW), a.files[0])
    path_b = next((p for p in b.files if p.suffix.lower() in PREFERRED_PREVIEW), b.files[0])
    profile_a, profile_b = back_portrait_profile(path_a), back_portrait_profile(path_b)
    return (profile_a is not None and profile_b is not None
            and profile_a.pose_family == profile_b.pose_family
            and float(np.minimum(profile_a.chroma, profile_b.chroma).sum()) >= 0.44)


def similar(a: PhotoUnit, b: PhotoUnit) -> bool:
    """Loose scene gate; a matching scene alone is never a duplicate action."""
    if a.phash is None or b.phash is None:
        return False
    aspect_close = abs(a.aspect - b.aspect) / max(a.aspect, b.aspect) <= 0.04
    distance = (a.phash ^ b.phash).bit_count()
    if aspect_close and distance <= 4:
        return True
    # A short portrait burst may change crop enough to defeat hashes and
    # geometric matching. Pose must still pass the later action gate.
    if (same_burst(a, b) and a.taken is not None and b.taken is not None
            and abs(a.taken - b.taken) <= 60
            and comparison_family(a) == comparison_family(b) == "person"
            and a.orientation == b.orientation
            and a.vision_vector is not None and b.vision_vector is not None
            and float(a.vision_vector @ b.vision_vector) >= .82
            and a.scene_hash is not None and b.scene_hash is not None
            and (a.scene_hash ^ b.scene_hash).bit_count() <= 35):
        return True
    if (not adjacent_capture(a, b) or a.color_hist is None or b.color_hist is None
            or a.scene_hash is None or b.scene_hash is None):
        return back_portrait_match(a, b)
    color_overlap = float(np.minimum(a.color_hist, b.color_hist).sum())
    scene_distance = (a.scene_hash ^ b.scene_hash).bit_count()
    if aspect_close and scene_distance <= 26 and color_overlap >= 0.72:
        return True
    return local_correspondence(a, b) or back_portrait_match(a, b)


def extract_pose(unit: PhotoUnit, estimator: object) -> None:
    try:
        path = next((p for p in unit.files if p.suffix.lower() in PREFERRED_PREVIEW), unit.files[0])
        image = open_preview(path)
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        result = estimator.process(np.asarray(image.convert("RGB")))
        if result.pose_landmarks is None:
            return
        marks = result.pose_landmarks.landmark
        left, right = marks[11], marks[12]
        width = float(np.hypot(left.x - right.x, left.y - right.y))
        if width < 0.08 or min(left.visibility, right.visibility) < 0.6:
            return
        center_x, center_y = (left.x + right.x) / 2, (left.y + right.y) / 2
        unit.pose_points = {
            i: ((marks[i].x - center_x) / width,
                (marks[i].y - center_y) / width,
                marks[i].visibility)
            for i in (0, 11, 12, 13, 14, 15, 16)
        }
    except DECODE_ERRORS:
        return


def same_action(a: PhotoUnit, b: PhotoUnit) -> bool:
    if "text" in {comparison_family(a), comparison_family(b)}:
        return (comparison_family(a) == comparison_family(b) == "text"
                and text_evidence_matches(a, b))
    # Near-identical pixels remain duplicates even when a person is not detected.
    if a.phash is not None and b.phash is not None and (a.phash ^ b.phash).bit_count() <= 4:
        return True
    # Content labels can flip between an object and an animal, or spuriously
    # call a figurine a person. Require same physical details, short capture
    # interval, high visual agreement, and no detected human face.
    if (not same_comparison_family(a, b)
            and "text" not in {comparison_family(a), comparison_family(b)}
            and not a.face_observations and not b.face_observations
            and same_burst(a, b) and a.vision_vector is not None
            and b.vision_vector is not None
            and float(a.vision_vector @ b.vision_vector) >= .92
            and local_correspondence(a, b)):
        return True
    if (comparison_family(a) != "unknown" and comparison_family(b) != "unknown"
            and not same_comparison_family(a, b)):
        return False
    if {a.orientation, b.orientation} == {"front", "back"}:
        return False
    if (comparison_family(a) in {"person", "unknown"}
            and comparison_family(b) in {"person", "unknown"}
            and back_portrait_match(a, b)):
        return True
    if a.pose_points is None or b.pose_points is None:
        return False
    ap, bp = a.pose_points, b.pose_points
    if min(ap[0][2], bp[0][2]) < 0.7:
        return False
    head_shift = float(np.hypot(ap[0][0] - bp[0][0], ap[0][1] - bp[0][1]))
    scene_distance = ((a.scene_hash ^ b.scene_hash).bit_count()
                      if a.scene_hash is not None and b.scene_hash is not None else 64)
    visual_agreement = (float(a.vision_vector @ b.vision_vector)
                        if a.vision_vector is not None and b.vision_vector is not None else 0.0)
    quick_stable_scene = (a.taken is not None and b.taken is not None
                          and abs(a.taken - b.taken) <= 12
                          and scene_distance <= 10 and visual_agreement >= .88)
    # A small head turn or downward glance should not split an otherwise stable action.
    if head_shift > 0.5:
        return False
    wrist_shifts = [float(np.hypot(ap[w][0] - bp[w][0], ap[w][1] - bp[w][1]))
                    for w in (15, 16) if w in ap and w in bp]
    if quick_stable_scene and all(
            np.hypot(ap[e][0] - bp[e][0], ap[e][1] - bp[e][1]) <= .5
            for e in (13, 14) if e in ap and e in bp):
        return True
    if (wrist_shifts and max(wrist_shifts) > .9 and scene_distance > 25):
        return False
    if (len(wrist_shifts) == 2 and min(wrist_shifts) > .48
            and a.taken is not None and b.taken is not None
            and abs(a.taken - b.taken) > 20 and visual_agreement < .90):
        return False
    reliable_elbows = 0
    for elbow in (13, 14):
        if elbow in ap and elbow in bp and min(ap[elbow][2], bp[elbow][2]) >= 0.55:
            reliable_elbows += 1
            if np.hypot(ap[elbow][0] - bp[elbow][0],
                        ap[elbow][1] - bp[elbow][1]) > 0.7:
                return False
    reliable_hands = 0
    for wrist in (15, 16):
        if wrist in ap and wrist in bp and min(ap[wrist][2], bp[wrist][2]) >= 0.55:
            reliable_hands += 1
            if np.hypot(ap[wrist][0] - bp[wrist][0],
                        ap[wrist][1] - bp[wrist][1]) > 0.52:
                return False
    if reliable_hands:
        return True
    # When wrists are occluded, require a much tighter visual burst as well as
    # consistent elbows. A matching background alone is insufficient.
    return (reliable_elbows >= 1 and a.scene_hash is not None and b.scene_hash is not None
            and (a.scene_hash ^ b.scene_hash).bit_count() <= 10)


def same_burst(a: PhotoUnit, b: PhotoUnit) -> bool:
    if a.relative.parent != b.relative.parent:
        return False
    if a.taken is not None and b.taken is not None:
        if abs(a.taken - b.taken) <= 90:
            return True
        if abs(a.taken - b.taken) > 300:
            return False
    first = re.fullmatch(r"(.*?)(\d+)", a.relative.stem, re.IGNORECASE)
    second = re.fullmatch(r"(.*?)(\d+)", b.relative.stem, re.IGNORECASE)
    return bool(first and second and first.group(1).casefold() == second.group(1).casefold()
                and abs(int(first.group(2)) - int(second.group(2))) <= 2)


def semantic_match(a: PhotoUnit, b: PhotoUnit, *, scene_matched: bool = False) -> bool:
    """Use content and subject evidence only inside a matched capture scene."""
    if (not same_burst(a, b) or a.vision_vector is None or b.vision_vector is None
            or (not scene_matched and not similar(a, b))):
        return False
    left_kind, right_kind = comparison_family(a), comparison_family(b)
    if left_kind != right_kind or left_kind == "unknown":
        return False
    if a.pose_points is not None and b.pose_points is not None:
        if not (a.subject_vector is not None and b.subject_vector is not None
                and "back" in (a.orientation, b.orientation)):
            for wrist in (15, 16):
                left, right = a.pose_points.get(wrist), b.pose_points.get(wrist)
                if (left is not None and right is not None
                        and min(left[2], right[2]) >= .55
                        and np.hypot(left[0] - right[0], left[1] - right[1]) > .52):
                    return False
    image_similarity = float(a.vision_vector @ b.vision_vector)
    people = left_kind == "person"
    if people and a.subject_vector is not None and b.subject_vector is not None:
        if (a.orientation != "unknown" and b.orientation != "unknown"
                and a.orientation != b.orientation):
            return False
        if image_similarity < .75:
            return False
        subject_similarity = float(a.subject_vector @ b.subject_vector)
        threshold = .65 if a.orientation == b.orientation == "front" else .75
        if subject_similarity < threshold:
            return False
        # A clearly different arm silhouette is a different action, even in one scene.
        if a.orientation != "front" or b.orientation != "front":
            path_a = next((p for p in a.files if p.suffix.lower() in PREFERRED_PREVIEW), a.files[0])
            path_b = next((p for p in b.files if p.suffix.lower() in PREFERRED_PREVIEW), b.files[0])
            profile_a, profile_b = back_portrait_profile(path_a), back_portrait_profile(path_b)
            if (profile_a is not None and profile_b is not None
                    and profile_a.pose_family != profile_b.pose_family):
                return False
        return True
    if people:
        return False
    scene_distance = ((a.scene_hash ^ b.scene_hash).bit_count()
                      if a.scene_hash is not None and b.scene_hash is not None else 64)
    pixel_distance = ((a.phash ^ b.phash).bit_count()
                      if a.phash is not None and b.phash is not None else 64)
    if left_kind == "text":
        return (image_similarity >= .82 and scene_distance <= 26
                and text_evidence_matches(a, b))
    if left_kind == "food":
        return (image_similarity >= .91 and scene_distance <= 24
                and (pixel_distance <= 12 or local_correspondence(a, b)))
    if left_kind == "animal":
        return (image_similarity >= .91 and scene_distance <= 24
                and (pixel_distance <= 12 or local_correspondence(a, b)))
    if left_kind in {"scenery", "plants", "architecture"}:
        threshold = .78 if left_kind == "scenery" else .84
        if image_similarity >= .91 and scene_distance <= 18:
            return True
        return (image_similarity >= threshold and scene_distance <= 26
                and local_correspondence(a, b))
    if left_kind in {"vehicle", "object", "art", "event"}:
        return (image_similarity >= .90 and scene_distance <= 20
                and (pixel_distance <= 10 or local_correspondence(a, b)))
    return False


def reviewed_subject_match(a: PhotoUnit, b: PhotoUnit) -> bool:
    """Catch a reframed subject missed by the first pass after both full-frame reviews.

    This needs independent local geometry as well as visual agreement; capture
    proximity or a shared background by itself cannot establish a substitute.
    """
    if (not a.second_pass_metrics or not b.second_pass_metrics
            or not same_burst(a, b) or not adjacent_capture(a, b)
            or a.vision_vector is None or b.vision_vector is None
            or a.second_pass_vector is None or b.second_pass_vector is None
            or a.scene_hash is None or b.scene_hash is None
            or (a.scene_hash ^ b.scene_hash).bit_count() > 32):
        return False
    left, right = comparison_family(a), comparison_family(b)
    if "text" in {left, right} or a.face_observations or b.face_observations:
        return False
    if left == right == "person":
        return False  # People still need the action and pose checks.
    agreement = float(a.vision_vector @ b.vision_vector)
    threshold = .95 if left == right == "food" else .92
    full_view_agreement = float(a.second_pass_vector @ b.second_pass_vector)
    return (agreement >= threshold and full_view_agreement >= .80
            and local_correspondence(a, b))


def cluster_compatible(a: PhotoUnit, b: PhotoUnit) -> bool:
    if a.vision_vector is None or b.vision_vector is None:
        return True
    if not same_comparison_family(a, b):
        return False
    family = comparison_family(a)
    people = family == "person"
    if people:
        if (a.orientation != "unknown" and b.orientation != "unknown"
                and a.orientation != b.orientation):
            return False
        if float(a.vision_vector @ b.vision_vector) < .74:
            return False
        if (a.subject_vector is not None and b.subject_vector is not None
                and float(a.subject_vector @ b.subject_vector) < .6):
            return False
        return True
    similarity = float(a.vision_vector @ b.vision_vector)
    if family == "text":
        return similarity >= .82 and text_evidence_matches(a, b)
    if family in {"animal", "food", "object", "art", "vehicle", "event"}:
        return similarity >= (.90 if family in {"animal", "food"} else .88)
    if family in {"scenery", "plants", "architecture"}:
        if similarity >= .86:
            return True
        return similarity >= (.78 if family == "scenery" else .84) and local_correspondence(a, b)
    return False


def categorize(units: list[PhotoUnit], *, _second_pass_complete: bool = False,
               _preliminary_groups: dict[int, int] | None = None) -> None:
    for unit in units:
        unit.group = 0
        unit.relative_best = False
    scene_pairs = [(i, j) for i, a in enumerate(units)
                   for j in range(i + 1, len(units)) if similar(a, units[j])]
    pose_candidates = {i for pair in scene_pairs for i in pair}
    if pose_candidates and any(units[i].pose_points is None for i in pose_candidates):
        try:
            import mediapipe as mp
            with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1,
                                        min_detection_confidence=0.5) as estimator:
                for i in sorted(pose_candidates):
                    if units[i].pose_points is None:
                        extract_pose(units[i], estimator)
        except (ImportError, OSError, RuntimeError, FileNotFoundError) as exc:
            print(f"Pose recognition unavailable; only exact visual repeats are grouped: {exc}",
                  file=sys.stderr)

    old_matching = {(i, j) for i, j in scene_pairs if same_action(units[i], units[j])}
    semantic_matching = {(i, j) for i, j in scene_pairs
                         if (i, j) not in old_matching
                         and semantic_match(units[i], units[j], scene_matched=True)}
    reviewed_matching = ({(i, j) for i, j in scene_pairs
                          if reviewed_subject_match(units[i], units[j])}
                         if _second_pass_complete else set())
    # Seed from the most convincing pair before considering weaker links.
    # Input order used to let an early, merely compatible frame consume one
    # member of a much stronger pair and block that pair by complete linkage.
    clusters: list[list[int]] = [[i] for i in range(len(units))]
    owner = list(range(len(units)))

    def pair_strength(pair: tuple[int, int]) -> float:
        left, right = (units[index] for index in pair)
        if left.vision_vector is not None and right.vision_vector is not None:
            visual = float(left.vision_vector @ right.vision_vector)
        elif left.phash is not None and right.phash is not None:
            visual = 1 - (left.phash ^ right.phash).bit_count() / 64
        else:
            visual = 0.0
        distance = ((left.scene_hash ^ right.scene_hash).bit_count()
                    if left.scene_hash is not None and right.scene_hash is not None else 64)
        return visual + (.01 if pair in old_matching else 0) - distance / 3200

    matching = old_matching | semantic_matching | reviewed_matching
    # A newly confirmed pair may need to replace a provisional group whose
    # other member is incompatible. Re-evaluate every group from scratch.
    newly_reviewed = {pair for pair in reviewed_matching
                      if _preliminary_groups is not None
                      and (_preliminary_groups[pair[0]] == 0
                           or _preliminary_groups[pair[0]]
                           != _preliminary_groups[pair[1]])}
    candidates = sorted(matching,
                        key=lambda pair: (pair in newly_reviewed,
                                          pair_strength(pair), -pair[0], -pair[1]),
                        reverse=True)
    for i, j in candidates:
        left, right = owner[i], owner[j]
        if left == right:
            continue
        if not all((min(a, b), max(a, b)) in matching
                   and ((min(a, b), max(a, b)) in old_matching | reviewed_matching
                        or cluster_compatible(units[a], units[b]))
                   for a in clusters[left] for b in clusters[right]):
            continue
        clusters[left].extend(clusters[right])
        for member in clusters[right]:
            owner[member] = left
        clusters[right] = []

    group_number = 0
    for indices in clusters:
        cluster = [units[i] for i in indices]
        if len(cluster) < 2:
            continue
        group_number += 1
        for unit in cluster:
            unit.group = group_number

    # Reopen every readable source image after the preliminary grouping. If
    # strong new subject evidence changes a comparison family, build the
    # groups once more before using neighboring frames or recommending one.
    if not _second_pass_complete:
        preliminary_groups = {i: unit.group for i, unit in enumerate(units)}
        independent_review_all(units)
        return categorize(units, _second_pass_complete=True,
                          _preliminary_groups=preliminary_groups)

    verify_assessments(units)

    for indices in clusters:
        cluster = [units[i] for i in indices]
        if len(cluster) < 2:
            continue
        group_number = cluster[0].group
        eligible = [u for u in cluster if u.category == "推荐保留"]
        if eligible:
            best = max(eligible, key=lambda u: (u.score, str(u.relative)))
            best.relative_best = True
            basis = {
                "person": "主体、眼睛和曝光", "animal": "主体局部清晰度和曝光",
                "food": "菜品局部清晰度和曝光", "text": "文字区域清晰度",
                "scenery": "景物清晰度和明暗层次", "plants": "花草局部清晰度",
                "architecture": "建筑细节和曝光", "vehicle": "主体轮廓和清晰度",
                "object": "物品细节和曝光", "art": "作品细节和曝光",
                "event": "关键主体清晰度和曝光",
            }.get(comparison_family(best), "画质")
            near_tie = any(u is not best and best.score - u.score < .05 for u in eligible)
            tie_note = "；差异较小，请人工复选" if near_tie else ""
            best.label = f"相似组 {group_number}：推荐此张；组内{basis}综合较好{tie_note}"
            for unit in eligible:
                if unit is not best:
                    unit.category = "相似照片待挑选"
                    unit.label = f"相似组 {group_number}：另有推荐照片{tie_note}"
        else:
            judged = [u for u in cluster if u.category == "有明显问题"]
            if judged:
                best = max(judged, key=lambda u: (u.score, str(u.relative)))
                best.relative_best = True
                best.label += f"；相似组 {group_number} 中相对最好"


def unique_destination(root: Path) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"{RESULT_PREFIX}{stamp}"
    n = 2
    while candidate.exists():
        candidate = root / f"{RESULT_PREFIX}{stamp}_{n}"
        n += 1
    return candidate


def write_reports(destination: Path, root: Path, units: list[PhotoUnit], errors: list[str]) -> None:
    counts = Counter(u.category for u in units for _ in u.files)
    with (destination / "判断标注.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("原始相对路径", "分类", "简短判断", "相似组", "组内相对最好",
                         "内容标签", "主比较对象", "人物朝向"))
        for unit in units:
            for path in unit.files:
                writer.writerow((str(path.relative_to(root)), unit.category, unit.label,
                                 unit.group or "", "是" if unit.relative_best else "",
                                 "、".join(CONTENT_NAMES[tag] for tag in unit.content_types)
                                 or "无法识别", CONTENT_NAMES.get(unit.primary_content, "无法识别"),
                                 {"front": "正面", "back": "背面"}.get(
                                     unit.orientation, "无法判断")))
    lines = ["照片筛选摘要", "", f"扫描照片：{sum(len(u.files) for u in units)} 张"]
    lines.extend(f"{category}：{counts[category]} 张" for category in CATEGORIES)
    types = Counter(tag for unit in units for tag in unit.content_types)
    if types:
        lines.extend(("", "内容概况：" + "、".join(
            f"{CONTENT_NAMES[tag]} {count}" for tag, count in types.most_common())))
    lines.extend(("", "所有原始文件均保留在原处；本次结果为复制件。", "自动判断仅供人工复选。"))
    if errors:
        lines.extend(("", f"复制失败：{len(errors)} 个文件", *errors[:20]))
    (destination / "摘要.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline photo triage without deletion")
    parser.add_argument("source", type=Path, help="Source folder")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="Analyze without copying")
    mode.add_argument("--apply", action="store_true", help="Copy to category folders")
    args = parser.parse_args()
    root = args.source.expanduser().resolve(strict=True)
    if not root.is_dir():
        parser.error("Source must be a folder")
    if local_vision() is None:
        print("Offline content model is required; no photos were sorted.", file=sys.stderr)
        return 2
    units = discover(root)
    for index, unit in enumerate(units, 1):
        assess(unit)
        if index % 50 == 0 or index == len(units):
            print(f"Analyzed {index}/{len(units)} photo units", file=sys.stderr, flush=True)
    print("Grouping similar photos...", file=sys.stderr, flush=True)
    categorize(units)
    counts = Counter(u.category for u in units for _ in u.files)
    needed = sum(path.stat().st_size for unit in units for path in unit.files)
    print(f"Photos: {sum(counts.values())}; units: {len(units)}; copy size: {needed / 1024**3:.2f} GiB")
    for category in CATEGORIES:
        print(f"{category}: {counts[category]}")
    types = Counter(tag for unit in units for tag in unit.content_types)
    print("Content: " + ", ".join(f"{CONTENT_NAMES[tag]} {count}"
                                  for tag, count in types.most_common()))
    if args.plan:
        return 0

    free = shutil.disk_usage(root).free
    if free < needed * 1.05 + 10 * 1024 * 1024:
        print("Insufficient free space for safe copying; source unchanged.", file=sys.stderr)
        return 2
    destination = unique_destination(root)
    destination.mkdir()
    errors: list[str] = []
    for category in CATEGORIES:
        (destination / category).mkdir()
    for unit in units:
        for path in unit.files:
            relative = path.relative_to(root)
            target = destination / unit.category / relative
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            except OSError as exc:
                errors.append(f"{relative}: {exc}")
    write_reports(destination, root, units, errors)
    print(f"Result: {destination}")
    if errors:
        print(f"Copy failures: {len(errors)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
