"""Behavior checks for the source-preserving photo workflow."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "triage.py"


class PhotoTriageTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_reported_portrait_false_positive_defects(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        for numbers, expected in (
            ((7512,), {7512: {"推荐保留", "相似照片待挑选"}}),
            ((7411,), {7411: {"推荐保留", "相似照片待挑选"}}),
            ((7330, 7331, 7332, 7333, 7334, 7335, 7337, 7338),
             {7331: {"无法判断"}}),
        ):
            with self.subTest(numbers=numbers):
                units = []
                for number in numbers:
                    path = root / f"IMG_{number}.JPG"
                    unit = module["PhotoUnit"]([path], Path(path.name))
                    module["assess"](unit)
                    units.append(unit)
                module["categorize"](units)
                for number, allowed in expected.items():
                    unit = next(unit for unit in units
                                if unit.relative.stem == f"IMG_{number}")
                    self.assertIn(unit.category, allowed,
                                  (number, unit.category, unit.label,
                                   unit.metrics, unit.face_observations))

    def test_reviewed_pair_needs_fresh_full_frame_and_local_evidence(self) -> None:
        module = runpy.run_path(str(SCRIPT))
        first = module["PhotoUnit"]([Path("IMG_1.JPG")], Path("IMG_1.JPG"))
        second = module["PhotoUnit"]([Path("IMG_2.JPG")], Path("IMG_2.JPG"))
        for item in (first, second):
            item.taken = 1.0
            item.primary_content = "object"
            item.scene_hash = 0
            item.vision_vector = np.array([1.0, 0.0], dtype=np.float32)
            item.second_pass_vector = np.array([1.0, 0.0], dtype=np.float32)
            item.second_pass_metrics = {"zones_clear": 3}
        globals_ = module["reviewed_subject_match"].__globals__
        with patch.dict(globals_, {"local_correspondence": lambda a, b: True}):
            self.assertTrue(module["reviewed_subject_match"](first, second))
            second.second_pass_vector = None
            self.assertFalse(module["reviewed_subject_match"](first, second))
            second.second_pass_vector = np.array([1.0, 0.0], dtype=np.float32)
            second.primary_content = "text"
            self.assertFalse(module["reviewed_subject_match"](first, second))
            second.primary_content = "object"
        self.assertFalse(module["reviewed_subject_match"](first, second))

    def test_second_pass_can_replace_an_earlier_conflicting_group(self) -> None:
        module = runpy.run_path(str(SCRIPT))
        units = [module["PhotoUnit"]([Path(f"IMG_{i}.JPG")],
                                     Path(f"IMG_{i}.JPG")) for i in range(1, 4)]
        for item in units:
            item.category = "推荐保留"
            item.score = 1.0
            item.assessed = True
            item.phash = 0
            item.scene_hash = 0
        ids = {id(item): i for i, item in enumerate(units)}
        pair = lambda a, b: frozenset((ids[id(a)], ids[id(b)]))

        def review_all(items):
            for item in items:
                item.second_pass_done = True
                item.second_pass_metrics = {"zones_clear": 1}
            return False

        rules = {
            "similar": lambda a, b: pair(a, b) in ({0, 1}, {1, 2}),
            "same_action": lambda a, b: pair(a, b) == {0, 1},
            "semantic_match": lambda a, b, **kwargs: False,
            "reviewed_subject_match": lambda a, b: pair(a, b) == {1, 2},
            "independent_review_all": review_all,
            "verify_assessments": lambda items: None,
        }
        with patch.dict(module["categorize"].__globals__, rules):
            module["categorize"](units)
        self.assertEqual(units[1].group, units[2].group)
        self.assertNotEqual(units[1].group, 0)
        self.assertNotEqual(units[0].group, units[1].group)
        self.assertEqual(sum(item.category == "推荐保留" for item in units[1:]), 1)

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_second_pass_regroups_reported_reframed_pairs_with_neighbors(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        for numbers, expected in (
            ((7657, 7658, 7659), (7658, 7659)),
            ((7665, 7666, 7667), (7666, 7667)),
            ((7763, 7764, 7765), (7764, 7765)),
        ):
            with self.subTest(numbers=numbers):
                units = []
                for number in numbers:
                    path = root / f"IMG_{number}.JPG"
                    unit = module["PhotoUnit"]([path], Path(path.name))
                    module["assess"](unit)
                    units.append(unit)
                module["categorize"](units)
                by_number = {int(unit.relative.stem[-4:]): unit for unit in units}
                left, right = (by_number[number] for number in expected)
                self.assertTrue(all(unit.second_pass_done for unit in units))
                self.assertNotEqual(left.group, 0)
                self.assertEqual(left.group, right.group)
                self.assertEqual(sum(unit.category == "推荐保留"
                                     for unit in units if unit.group == left.group), 1)

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_second_pass_reopens_full_images_and_can_correct_first_pass(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        units = []
        for number in (7271, 7272, 7418, 7490):
            path = root / f"IMG_{number}.JPG"
            unit = module["PhotoUnit"]([path], Path(path.name))
            module["assess"](unit)
            units.append(unit)
        # Simulate a mistaken primary label and a missed first-pass blur.
        units[0].primary_content = "object"
        units[0].content_types = ("object",)
        units[-1].problem_reasons = []
        units[-1].review_reasons = []
        units[-1].category = "推荐保留"
        module["categorize"](units)
        self.assertTrue(all(unit.second_pass_done for unit in units))
        self.assertTrue(all("zones_clear" in unit.second_pass_metrics for unit in units))
        self.assertEqual(units[0].primary_content, "pet")
        self.assertEqual(units[0].group, units[1].group)
        self.assertNotEqual(units[0].group, 0)
        self.assertEqual(units[2].category, "推荐保留")
        self.assertEqual(units[3].category, "有明显问题")
        self.assertIn("第二遍复核", units[3].label)

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_reported_subject_quality_and_scene_focus(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        expected = {
            7418: {"推荐保留"},
            7490: {"有明显问题"},
            7347: {"有明显问题"},
            7368: {"无法判断", "有明显问题"},
            7695: {"有明显问题"},
        }
        for number, allowed in expected.items():
            with self.subTest(number=number):
                path = root / f"IMG_{number}.JPG"
                unit = module["PhotoUnit"]([path], Path(path.name))
                module["assess"](unit)
                self.assertIn(unit.category, allowed, (unit.label, unit.metrics))

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_reported_same_subject_pairs_are_grouped(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        for first, second in ((7318, 7319), (7330, 7337),
                              (7499, 7500), (7666, 7667)):
            with self.subTest(pair=(first, second)):
                units = []
                for number in (first, second):
                    path = root / f"IMG_{number}.JPG"
                    unit = module["PhotoUnit"]([path], Path(path.name))
                    module["assess"](unit)
                    units.append(unit)
                module["categorize"](units)
                self.assertNotEqual(units[0].group, 0)
                self.assertEqual(units[0].group, units[1].group)

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_reported_expression_and_pose_use_neighboring_frames(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        for numbers, reported in (
            ((7290, 7291, 7292, 7293), {7290: "无法判断"}),
            ((7574, 7576, 7577), {7577: "有明显问题"}),
            ((7582, 7583), {7582: "无法判断"}),
            ((7593, 7594, 7595, 7596), {7594: "有明显问题"}),
            ((7608, 7609, 7610), {7609: "有明显问题"}),
            ((7690, 7691, 7692), {7690: "有明显问题"}),
        ):
            with self.subTest(numbers=numbers):
                units = []
                for number in numbers:
                    path = root / f"IMG_{number}.JPG"
                    unit = module["PhotoUnit"]([path], Path(path.name))
                    module["assess"](unit)
                    units.append(unit)
                module["categorize"](units)
                by_number = {int(unit.relative.stem[-4:]): unit for unit in units}
                for number, category in reported.items():
                    self.assertEqual(by_number[number].category, category,
                                     (number, by_number[number].label))

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_reported_portrait_groups_survive_full_scene(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        units = []
        for number in range(7316, 7338):
            path = root / f"IMG_{number}.JPG"
            if path.exists():
                unit = module["PhotoUnit"]([path], Path(path.name))
                module["assess"](unit)
                units.append(unit)
        module["categorize"](units)
        groups = {int(unit.relative.stem[-4:]): unit.group for unit in units}
        self.assertNotEqual(groups[7318], 0, groups)
        self.assertEqual(groups[7318], groups[7319], groups)
        self.assertNotEqual(groups[7330], 0, groups)
        self.assertEqual(groups[7330], groups[7337], groups)
        self.assertNotEqual(groups[7326], groups[7330], groups)

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_mixed_person_and_pet_uses_visible_person_as_primary_subject(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        for number in (7301, 7302, 7303):
            path = root / f"IMG_{number}.JPG"
            unit = module["PhotoUnit"]([path], Path(path.name))
            module["assess"](unit)
            self.assertEqual(unit.primary_content, "person", number)

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_back_view_does_not_block_front_view_group_in_full_sequence(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        units = []
        for number in range(7378, 7389):
            path = root / f"IMG_{number}.JPG"
            unit = module["PhotoUnit"]([path], Path(path.name))
            module["assess"](unit)
            units.append(unit)
        module["categorize"](units)
        by_number = {int(unit.relative.stem[-4:]): unit for unit in units}
        self.assertEqual(by_number[7381].orientation, "back")
        front_groups = {by_number[number].group for number in range(7382, 7389)}
        self.assertEqual(len(front_groups), 1, front_groups)
        self.assertNotIn(0, front_groups)
        self.assertNotEqual(by_number[7381].group, by_number[7387].group)

    def test_mixed_content_uses_the_primary_subject_before_grouping(self) -> None:
        module = runpy.run_path(str(SCRIPT))
        unit = module["PhotoUnit"]
        a = unit([Path("IMG_1.JPG")], Path("IMG_1.JPG"))
        b = unit([Path("IMG_2.JPG")], Path("IMG_2.JPG"))
        for item in (a, b):
            item.taken = 1.0
            item.vision_vector = np.array([1.0, 0.0], dtype=np.float32)
            item.scene_hash = 0
            item.content_types = ("food", "indoor")
        a.primary_content = "food"
        b.primary_content = "indoor"
        self.assertFalse(module["semantic_match"](a, b, scene_matched=True))

    def test_text_content_needs_document_evidence_beyond_global_similarity(self) -> None:
        module = runpy.run_path(str(SCRIPT))
        unit = module["PhotoUnit"]
        a = unit([Path("IMG_1.JPG")], Path("IMG_1.JPG"))
        b = unit([Path("IMG_2.JPG")], Path("IMG_2.JPG"))
        for item in (a, b):
            item.taken = 1.0
            item.vision_vector = np.array([1.0, 0.0], dtype=np.float32)
            item.content_types = ("text",)
            item.primary_content = "text"
        a.phash, b.phash = 0, 255
        a.scene_hash, b.scene_hash = 0, 255
        self.assertFalse(module["semantic_match"](a, b, scene_matched=True))

    def test_text_pages_with_different_numbers_do_not_merge(self) -> None:
        module = runpy.run_path(str(SCRIPT))
        unit = module["PhotoUnit"]
        a = unit([Path("IMG_1.JPG")], Path("IMG_1.JPG"))
        b = unit([Path("IMG_2.JPG")], Path("IMG_2.JPG"))
        for item in (a, b):
            item.taken = 1.0
            item.primary_content = "text"
            item.content_types = ("text",)
            item.vision_vector = np.array([1.0, 0.0], dtype=np.float32)
            item.phash = 0
            item.scene_hash = 0
        a.text_tokens = ("gate", "42", "open")
        b.text_tokens = ("gate", "43", "open")
        self.assertFalse(module["same_action"](a, b))
        self.assertFalse(module["semantic_match"](a, b, scene_matched=True))

    def test_same_readable_text_can_match_after_reframing(self) -> None:
        module = runpy.run_path(str(SCRIPT))
        unit = module["PhotoUnit"]
        a = unit([Path("IMG_1.JPG")], Path("IMG_1.JPG"))
        b = unit([Path("IMG_2.JPG")], Path("IMG_2.JPG"))
        for item in (a, b):
            item.taken = 1.0
            item.primary_content = "text"
            item.content_types = ("text",)
            item.vision_vector = np.array([1.0, 0.0], dtype=np.float32)
            item.scene_hash = 0
            item.text_tokens = ("gate", "42", "open")
        a.phash, b.phash = 0, 255
        self.assertTrue(module["semantic_match"](a, b, scene_matched=True))

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_sign_photo_has_local_text_evidence(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        path = root / "IMG_7366.JPG"
        unit = module["PhotoUnit"]([path], Path(path.name))
        module["assess"](unit)
        self.assertEqual(unit.primary_content, "text")
        self.assertGreaterEqual(len(unit.text_tokens), 3)

    def test_subject_priority_changes_with_photo_type(self) -> None:
        module = runpy.run_path(str(SCRIPT))
        unit = module["PhotoUnit"]
        crisp_text = unit([Path("a.jpg")], Path("a.jpg"))
        broad_texture = unit([Path("b.jpg")], Path("b.jpg"))
        crisp_text.primary_content = broad_texture.primary_content = "text"
        crisp_text.metrics = {"sharpness": 60.0, "local_focus": 180.0,
                              "dark": 0.01, "bright": 0.01, "noise": 0.0}
        broad_texture.metrics = {"sharpness": 180.0, "local_focus": 30.0,
                                 "dark": 0.01, "bright": 0.01, "noise": 0.0}
        self.assertGreater(module["type_selection_score"](crisp_text),
                           module["type_selection_score"](broad_texture))

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_independent_content_examples_keep_distinct_subjects(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        numbers = (7347, 7355, 7356, 7366, 7370, 7371,
                   7682, 7683, 7684, 7687)
        units = []
        for number in numbers:
            path = root / f"IMG_{number}.JPG"
            unit = module["PhotoUnit"]([path], Path(path.name))
            module["assess"](unit)
            units.append(unit)
        module["categorize"](units)
        by_number = {int(unit.relative.stem[-4:]): unit for unit in units}
        for left, right in ((7355, 7356), (7370, 7371), (7682, 7683)):
            self.assertNotEqual(by_number[left].group, 0)
            self.assertEqual(by_number[left].group, by_number[right].group)
        self.assertNotEqual(by_number[7683].group, by_number[7684].group)
        self.assertNotEqual(by_number[7683].group, by_number[7687].group)
        self.assertIn("pet", by_number[7347].content_types)
        self.assertIn("text", by_number[7366].content_types)

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_content_aware_groups_cover_people_pets_scenery_and_food(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        numbers = (7271, 7272, 7274, 7275, 7350, 7351, 7378, 7380,
                   7382, 7383, 7384, 7386, 7387, 7388, 7684, 7685, 7686)
        units = []
        for number in numbers:
            path = root / f"IMG_{number}.JPG"
            unit = module["PhotoUnit"]([path], Path(path.name))
            module["assess"](unit)
            units.append(unit)
        module["categorize"](units)
        by_number = {int(unit.relative.stem[-4:]): unit for unit in units}
        for group in ((7271, 7272), (7274, 7275), (7350, 7351),
                      (7378, 7380), (7382, 7383, 7384, 7386, 7387, 7388)):
            ids = {by_number[number].group for number in group}
            self.assertEqual(len(ids), 1, [(number, by_number[number].group) for number in group])
            self.assertNotIn(0, ids, group)
        for left, right in ((7272, 7274), (7378, 7383),
                            (7684, 7685), (7685, 7686)):
            self.assertFalse(by_number[left].group and by_number[left].group == by_number[right].group,
                             (left, right, by_number[left].group, by_number[right].group))

    def test_local_scene_match_survives_a_different_crop(self) -> None:
        with tempfile.TemporaryDirectory(prefix="photo-triage-crop-") as folder:
            root = Path(folder)
            rng = np.random.default_rng(19)
            original = Image.fromarray(
                rng.integers(0, 256, size=(640, 960, 3), dtype=np.uint8)
            )
            original.save(root / "IMG_1.JPG", quality=98)
            original.crop((80, 35, 840, 565)).save(root / "IMG_2.JPG", quality=93)
            module = runpy.run_path(str(SCRIPT))
            units = []
            for number in (1, 2):
                path = root / f"IMG_{number}.JPG"
                unit = module["PhotoUnit"]([path], Path(path.name))
                module["assess"](unit)
                unit.taken = float(number)
                units.append(unit)
            self.assertTrue(module["similar"](*units))

    def test_focused_subject_with_blurred_background_is_not_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="photo-triage-focus-") as folder:
            root = Path(folder)
            rng = np.random.default_rng(42)
            background = Image.fromarray(
                rng.integers(20, 225, size=(1024, 683, 3), dtype=np.uint8)
            ).filter(ImageFilter.GaussianBlur(35))
            backdrop = ImageDraw.Draw(background)
            backdrop.rectangle((0, 0, 682, 300), fill=(65, 75, 90))
            backdrop.rectangle((0, 700, 682, 1023), fill=(165, 150, 135))
            background = background.filter(ImageFilter.GaussianBlur(35))
            background.save(root / "all-blurred.jpg", quality=95)
            draw = ImageDraw.Draw(background)
            draw.ellipse((325, 475, 365, 515), fill=(210, 75, 135), outline=(155, 65, 110), width=1)
            for angle in range(0, 360, 45):
                cx = 345 + int(16 * math.cos(math.radians(angle)))
                cy = 495 + int(16 * math.sin(math.radians(angle)))
                draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6),
                             fill=(220, 95, 150), outline=(150, 65, 105), width=1)
            draw.ellipse((340, 490, 350, 500), fill=(215, 150, 70))
            background = background.filter(ImageFilter.GaussianBlur(0.5))
            background.save(root / "flower.jpg", quality=95)
            module = runpy.run_path(str(SCRIPT))
            unit = module["PhotoUnit"]([root / "flower.jpg"], Path("flower.jpg"))
            module["assess"](unit)
            self.assertLess(unit.metrics["sharpness"], 12, unit.metrics)
            self.assertNotEqual(unit.category, "有明显问题", unit.metrics)
            blurry = module["PhotoUnit"]([root / "all-blurred.jpg"], Path("all-blurred.jpg"))
            module["assess"](blurry)
            self.assertEqual(blurry.category, "无法判断", blurry.metrics)

            blocks = np.indices((1024, 683)).sum(axis=0)
            pattern = (((blocks // 32) % 2) * 180 + 35).astype("uint8")
            pattern_rgb = np.stack((pattern, pattern, pattern), axis=2)
            Image.fromarray(pattern_rgb).filter(ImageFilter.GaussianBlur(4)).save(
                root / "blurred-pattern.jpg", quality=95
            )
            severely_blurred = module["PhotoUnit"](
                [root / "blurred-pattern.jpg"], Path("blurred-pattern.jpg")
            )
            module["assess"](severely_blurred)
            self.assertEqual(severely_blurred.category, "有明显问题", severely_blurred.metrics)

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_reported_flower_photo_is_not_rejected(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        unit = module["PhotoUnit"]([root / "IMG_7426.JPG"], Path("IMG_7426.JPG"))
        module["assess"](unit)
        self.assertEqual(unit.category, "推荐保留", unit.metrics)

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_reported_unflattering_expression_is_not_recommended(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        unit = module["PhotoUnit"]([root / "IMG_7818.JPG"], Path("IMG_7818.JPG"))
        module["assess"](unit)
        self.assertEqual(unit.category, "有明显问题", unit.label)
        self.assertIn("张嘴", unit.label)
        neighbor = module["PhotoUnit"]([root / "IMG_7819.JPG"], Path("IMG_7819.JPG"))
        module["assess"](neighbor)
        self.assertNotIn("张嘴且一只眼", neighbor.label)

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_relaxed_closed_eye_portraits_are_not_obvious_problems(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        units = []
        for number in (7343, 7344, 7345):
            path = root / f"IMG_{number}.JPG"
            unit = module["PhotoUnit"]([path], Path(path.name))
            module["assess"](unit)
            units.append(unit)
        self.assertTrue(all(unit.category != "有明显问题" for unit in units[1:]),
                        [(unit.relative.name, unit.label) for unit in units])
        module["categorize"](units)
        self.assertEqual(len({unit.group for unit in units}), 1)
        self.assertNotEqual(units[0].group, 0)
        self.assertEqual(sum(unit.category == "推荐保留" for unit in units), 1)
        self.assertEqual(sum(unit.category == "相似照片待挑选" for unit in units), 2)

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_downward_gaze_is_not_claimed_to_be_closed_eyes(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        path = root / "IMG_7325.JPG"
        unit = module["PhotoUnit"]([path], Path(path.name))
        module["assess"](unit)
        self.assertEqual(unit.category, "推荐保留", unit.label)
        self.assertNotIn("双眼闭合", unit.label)
        self.assertIn("低头注视", unit.label)

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_dark_background_is_not_edge_obstruction(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        path = root / "IMG_7791.JPG"
        unit = module["PhotoUnit"]([path], Path(path.name))
        module["assess"](unit)
        self.assertEqual(unit.category, "推荐保留", unit.label)
        self.assertNotIn("遮挡", unit.label)

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_second_pass_checks_edge_against_the_whole_subject(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        with tempfile.TemporaryDirectory(prefix="photo-triage-verify-") as folder:
            for number, edge, expected in (
                (7387, "left", "推荐保留"),
                (7325, "right", "无法判断"),
            ):
                with Image.open(root / f"IMG_{number}.JPG") as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    image.thumbnail((1024, 1024))
                draw = ImageDraw.Draw(image)
                if edge == "left":
                    rectangle = (0, 0, int(image.width * .28), image.height)
                else:
                    rectangle = (int(image.width * .72), 0, image.width, image.height)
                draw.rectangle(rectangle, fill=(210, 135, 105))
                path = Path(folder) / f"IMG_{number}.jpg"
                image.save(path, quality=95)
                unit = module["PhotoUnit"]([path], Path(path.name))
                module["assess"](unit)
                self.assertEqual(unit.primary_content, "person")
                self.assertEqual(unit.category, "无法判断", unit.label)
                module["categorize"]([unit])
                self.assertTrue(unit.verified)
                self.assertEqual(unit.category, expected, (number, unit.label))

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_same_pose_with_small_head_change_forms_one_group(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        units = []
        for number in (7301, 7302, 7303):
            jpg, raw = (root / f"IMG_{number}.{suffix}" for suffix in ("JPG", "CR3"))
            unit = module["PhotoUnit"]([jpg, raw], Path(jpg.name))
            module["assess"](unit)
            units.append(unit)
        module["categorize"](units)
        self.assertEqual(len({u.group for u in units}), 1, [(u.relative, u.group) for u in units])
        self.assertNotEqual(units[0].group, 0)
        self.assertEqual(sum(u.category == "推荐保留" for u in units), 1)
        self.assertEqual(sum(u.category == "相似照片待挑选" for u in units), 2)

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_same_flower_pose_with_tighter_crop_forms_one_group(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        units = []
        for number in (7324, 7326, 7327, 7328, 7329, 7330):
            jpg, raw = (root / f"IMG_{number}.{suffix}" for suffix in ("JPG", "CR3"))
            unit = module["PhotoUnit"]([jpg, raw], Path(jpg.name))
            module["assess"](unit)
            units.append(unit)
        module["categorize"](units)
        by_number = {int(u.relative.stem[-4:]): u for u in units}
        flower_group = [by_number[n] for n in (7326, 7327, 7328, 7329)]
        self.assertEqual(len({u.group for u in flower_group}), 1,
                         [(u.relative, u.group) for u in units])
        self.assertNotEqual(flower_group[0].group, 0)
        self.assertEqual(sum(u.category == "推荐保留" for u in flower_group), 1)
        self.assertEqual(sum(u.category == "相似照片待挑选" for u in flower_group), 3)
        self.assertNotEqual(by_number[7324].group, flower_group[0].group)
        self.assertNotEqual(by_number[7330].group, flower_group[0].group)

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_back_view_portraits_group_when_pose_is_occluded(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        numbers = (7394, 7395, 7396, 7397, 7398, 7399, 7400, 7401, 7402,
                   7403, 7404, 7405, 7407)
        units = []
        for number in numbers:
            jpg, raw = (root / f"IMG_{number}.{suffix}" for suffix in ("JPG", "CR3"))
            unit = module["PhotoUnit"]([jpg, raw], Path(jpg.name))
            module["assess"](unit)
            units.append(unit)
        module["categorize"](units)
        groups = {int(u.relative.stem[-4:]): u.group for u in units}
        first = {groups[n] for n in (7394, 7395, 7396, 7397)}
        second = {groups[n] for n in (7400, 7401, 7402)}
        self.assertEqual(len(first), 1, groups)
        self.assertEqual(len(second), 1, groups)
        self.assertNotEqual(next(iter(first)), 0, groups)
        self.assertNotEqual(next(iter(second)), 0, groups)
        self.assertNotEqual(first, second, groups)
        for number in (7398, 7399, 7403, 7405):
            self.assertNotIn(groups[number], first | second, groups)
        for left, right in ((7404, 7405), (7404, 7407), (7405, 7407)):
            self.assertFalse(groups[left] and groups[left] == groups[right], groups)

    @unittest.skipUnless(os.environ.get("PHOTO_TRIAGE_REAL_DIR"), "real photos unavailable")
    def test_different_actions_in_same_scene_stay_separate(self) -> None:
        root = Path(os.environ["PHOTO_TRIAGE_REAL_DIR"])
        module = runpy.run_path(str(SCRIPT))
        units = []
        for number in range(7674, 7681):
            path = root / f"IMG_{number}.JPG"
            unit = module["PhotoUnit"]([path], Path(path.name))
            module["assess"](unit)
            units.append(unit)
        module["categorize"](units)
        groups = {int(unit.relative.stem[-4:]): unit.group for unit in units}
        self.assertEqual(groups[7674], groups[7676], groups)
        self.assertNotEqual(groups[7674], 0, groups)
        self.assertEqual(groups[7679], groups[7680], groups)
        self.assertNotEqual(groups[7679], 0, groups)
        self.assertNotEqual(groups[7677], groups[7679], groups)
        self.assertNotEqual(groups[7678], groups[7679], groups)

    def test_adjacent_portraits_with_different_actions_stay_separate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="photo-triage-burst-") as folder:
            root = Path(folder)
            for number, offset in ((7674, -22), (7675, 0), (7676, 20)):
                image = Image.new("RGB", (320, 240), (70, 64, 59))
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 100, 115), fill=(29, 28, 32))
                draw.rectangle((218, 0, 319, 95), fill=(105, 80, 58))
                draw.rectangle((0, 188, 319, 239), fill=(132, 88, 63))
                draw.ellipse((132 + offset, 32, 207 + offset, 111), fill=(200, 151, 118))
                draw.polygon(((115 + offset, 111), (218 + offset, 111),
                              (246 + offset, 239), (88 + offset, 239)), fill=(224, 217, 204))
                draw.line((145 + offset, 150, 195 + offset, 122 + abs(offset)),
                          fill=(179, 125, 92), width=12)
                image.save(root / f"IMG_{number}.JPG")

            module = runpy.run_path(str(SCRIPT))
            units = []
            for number in (7674, 7675, 7676):
                path = root / f"IMG_{number}.JPG"
                unit = module["PhotoUnit"]([path], Path(path.name))
                module["assess"](unit)
                units.append(unit)
            for unit, wrist, signature in zip(units, (0.1, 1.0, 0.15),
                                              (0, 0x3FF, 0x3FF000)):
                unit.pose_points = {0: (0.0, 0.0, 1.0),
                                    15: (wrist, 0.0, 1.0), 16: (0.0, 0.2, 1.0)}
                unit.phash = signature
                unit.scene_hash = 0
                unit.color_hist = np.array([1.0])
            module["categorize"](units)
            self.assertEqual(units[0].group, units[2].group)
            self.assertNotEqual(units[0].group, 0)
            self.assertEqual(units[1].group, 0)
            self.assertEqual(sum(u.category == "相似照片待挑选" for u in units), 1)

    def test_group_faces_and_subject_quality_affect_selection(self) -> None:
        module = runpy.run_path(str(SCRIPT))
        face = module["FaceInfo"]
        good = face(150, 0.0, 0.0, 0.22, 0.24, 0.0, False)
        blinking = face(140, 0.5, 0.0, 0.08, 0.1, 0.0, False)
        problems, review = module["face_findings"]([good, blinking], 120)
        self.assertTrue(any("第2位人物" in item and "过曝" in item for item in problems))
        self.assertEqual(review, [])

        posed = face(140, 0.0, 0.0, 0.04, 0.06, 0.01, False)
        problems, review = module["face_findings"]([posed], 120)
        self.assertEqual((problems, review), ([], []))
        ambiguous = face(140, 0.0, 0.0, 0.04, 0.06, 0.22, False)
        problems, review = module["face_findings"]([ambiguous], 120)
        self.assertEqual(problems, [])
        self.assertTrue(any("眼部开合度较低且张嘴" in item for item in review))

        soft_face = face(10, 0.0, 0.0, 0.22, 0.24, 0.0, False)
        problems, _ = module["face_findings"]([soft_face], 140)
        self.assertIn("人物面部明显失焦", problems)
        bright_background = module["selection_score"](160, 0, 0, [soft_face], 0)
        clear_face = module["selection_score"](100, 0, 0, [good], 0)
        self.assertGreater(clear_face, bright_background)

        clipped = face(100, 0.0, 0.0, 0.22, 0.24, 0.0, True)
        _, review = module["face_findings"]([clipped], 100)
        self.assertTrue(any("裁切" in item for item in review))

    def test_multiple_prominent_faces_are_measured(self) -> None:
        module = runpy.run_path(str(SCRIPT))
        detections = [
            SimpleNamespace(score=[0.95], location_data=SimpleNamespace(
                relative_bounding_box=SimpleNamespace(xmin=x, ymin=0.3, width=0.23, height=0.25)
            )) for x in (0.12, 0.62)
        ]
        detector = SimpleNamespace(process=lambda image: SimpleNamespace(detections=detections))
        mesh = SimpleNamespace(process=lambda image: SimpleNamespace(multi_face_landmarks=None))
        module["assess_faces"].__globals__["expression_models"] = lambda: (detector, mesh)
        image = Image.new("RGB", (512, 512), "gray")
        faces = module["assess_faces"](image, np.full((512, 512), 128, dtype=np.float32))
        self.assertEqual(len(faces), 2)

    def test_noise_compression_and_edge_obstruction_are_labeled(self) -> None:
        module = runpy.run_path(str(SCRIPT))
        with tempfile.TemporaryDirectory(prefix="photo-triage-quality-") as folder:
            root = Path(folder)
            rng = np.random.default_rng(17)
            noisy = np.clip(80 + rng.normal(0, 30, (512, 512)), 0, 255).astype("uint8")
            Image.fromarray(noisy).save(root / "noisy.png")
            patterned = (np.indices((512, 512)).sum(axis=0) // 5 % 2 * 180 + 30).astype("uint8")
            Image.fromarray(patterned).save(root / "compressed.jpg", quality=10)
            rgb = np.stack((patterned, patterned, patterned), axis=2)
            rgb[:, :140] = (210, 135, 105)
            Image.fromarray(rgb).save(root / "covered.png")
            shadowed = np.stack((patterned, patterned, patterned), axis=2)
            shadowed[:, 370:] = (5, 6, 5)
            Image.fromarray(shadowed).save(root / "shadowed.png")

            with Image.open(root / "shadowed.png") as shadow_image:
                shadow_gray = np.asarray(ImageOps.grayscale(shadow_image), dtype=np.float32)
                self.assertFalse(module["possible_obstruction"](shadow_image, shadow_gray))

            for name, reason, category in (
                ("noisy.png", "噪点", "有明显问题"),
                ("compressed.jpg", "JPEG 压缩", "有明显问题"),
                ("covered.png", "遮挡", "无法判断"),
            ):
                unit = module["PhotoUnit"]([root / name], Path(name))
                module["assess"](unit)
                self.assertEqual(unit.category, category, (name, unit.label, unit.metrics))
                self.assertIn(reason, unit.label)

    def test_copy_categories_and_repeat_run_preserve_sources(self) -> None:
        with tempfile.TemporaryDirectory(prefix="photo-triage-test-") as folder:
            root = Path(folder)
            album = root / "trip"
            album.mkdir()
            checker = np.indices((256, 256)).sum(axis=0) % 2 * 255
            rgb = np.stack((checker, checker, checker), axis=2).astype("uint8")
            Image.fromarray(rgb).save(album / "good.jpg")
            Image.fromarray(rgb).save(album / "good-copy.jpg")
            Image.new("RGB", (256, 256), "black").save(album / "dark.png")
            (album / "camera.nef").write_bytes(b"not a decoded raw photo")
            (album / "clip.mp4").write_bytes(b"video")
            originals = {
                p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in album.iterdir()
            }

            plan = subprocess.run(
                [sys.executable, str(SCRIPT), str(root), "--plan"],
                text=True, capture_output=True, check=True,
            )
            self.assertEqual(list(root.glob("_照片筛选结果_*")), [])
            self.assertIn("Photos: 4", plan.stdout)

            first = subprocess.run(
                [sys.executable, str(SCRIPT), str(root), "--apply"],
                text=True, capture_output=True, check=True,
            )
            self.assertIn("Result:", first.stdout)
            result = next(root.glob("_照片筛选结果_*"))
            with (result / "判断标注.csv").open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            by_name = {Path(row["原始相对路径"]).name: row for row in rows}
            self.assertEqual(len(rows), 4)
            self.assertEqual(by_name["dark.png"]["分类"], "有明显问题")
            self.assertEqual(by_name["camera.nef"]["分类"], "无法判断")
            self.assertEqual(
                {by_name["good.jpg"]["分类"], by_name["good-copy.jpg"]["分类"]},
                {"推荐保留", "相似照片待挑选"},
            )
            self.assertFalse(any(result.rglob("clip.mp4")))
            for relative, checksum in originals.items():
                self.assertEqual(hashlib.sha256((root / relative).read_bytes()).hexdigest(), checksum)

            second = subprocess.run(
                [sys.executable, str(SCRIPT), str(root), "--apply"],
                text=True, capture_output=True, check=True,
            )
            self.assertIn("Photos: 4", second.stdout)
            self.assertEqual(len(list(root.glob("_照片筛选结果_*"))), 2)


if __name__ == "__main__":
    unittest.main()
