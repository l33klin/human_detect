"""Draw only boxes and small cached label patches; avoid full-frame Pillow copies."""

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Annotator:
    def __init__(self, font_path, font_size=22):
        # Fail explicitly instead of silently replacing Chinese names with missing glyphs.
        self.font = ImageFont.truetype(font_path, font_size)
        self.label_height = font_size + 12

    @lru_cache(maxsize=256)
    def label(self, text):
        bounds = self.font.getbbox(text)
        width = max(1, bounds[2] - min(0, bounds[0])) + 16
        image = Image.new("RGB", (width, self.label_height), (18, 40, 35))
        ImageDraw.Draw(image).text((8, 6 - bounds[1]), text, font=self.font, fill=(240, 255, 247))
        return np.asarray(image)

    def _draw_label(self, rgb, text, x, y):
        height, width = rgb.shape[:2]
        label = self.label(text)
        x = max(0, min(width - min(width, label.shape[1]), x))
        y = max(0, min(height - 1, y))
        h, w = min(label.shape[0], height - y), min(label.shape[1], width - x)
        rgb[y : y + h, x : x + w] = label[:h, :w]

    def draw(self, rgb, persons):
        height, width = rgb.shape[:2]
        for person in persons:
            box = person["box"]
            cv2.rectangle(
                rgb, (box["x1"], box["y1"]), (box["x2"] - 1, box["y2"] - 1), (55, 220, 155), 2
            )
            labels = []
            for face in person["faces"]:
                fb = face["box"]
                cv2.rectangle(
                    rgb, (fb["x1"], fb["y1"]), (fb["x2"] - 1, fb["y2"] - 1), (255, 205, 90), 2
                )
                self._draw_label(
                    rgb,
                    f"人脸 {face['confidence']:.3f}",
                    fb["x1"],
                    min(height - self.label_height, fb["y2"] + 2),
                )
                identity = face["identity"]
                labels.append(
                    f"{identity['name']} [{identity['person_id']}]"
                    if identity["status"] == "matched"
                    else "未知"
                )
            identity_text = " / ".join(labels) if labels else "未检测到人脸"
            self._draw_label(
                rgb,
                f"人体 {person['confidence']:.3f} | {identity_text}",
                box["x1"],
                box["y1"] - self.label_height,
            )
        return rgb
