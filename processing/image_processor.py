import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance

TARGET_WIDTH = 1600
TARGET_HEIGHT = 900
TARGET_RATIO = TARGET_WIDTH / TARGET_HEIGHT


def expand_to_ratio(img: Image.Image, target_ratio: float) -> Image.Image:
    width, height = img.size
    current_ratio = width / height
    if abs(current_ratio - target_ratio) < 0.01:
        return img
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if current_ratio > target_ratio:
        new_height = int(width / target_ratio)
        pad = (new_height - height) // 2
        background = cv2.GaussianBlur(cv_img, (151, 151), 50)
        expanded = cv2.copyMakeBorder(cv_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        background_resized = cv2.resize(background, (expanded.shape[1], expanded.shape[0]))
        expanded[0:pad, :] = background_resized[0:pad, :]
        expanded[-pad:, :] = background_resized[-pad:, :]
    else:
        new_width = int(height * target_ratio)
        pad = (new_width - width) // 2
        background = cv2.GaussianBlur(cv_img, (151, 151), 50)
        expanded = cv2.copyMakeBorder(cv_img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        background_resized = cv2.resize(background, (expanded.shape[1], expanded.shape[0]))
        expanded[:, 0:pad] = background_resized[:, 0:pad]
        expanded[:, -pad:] = background_resized[:, -pad:]
    expanded_rgb = cv2.cvtColor(expanded, cv2.COLOR_BGR2RGB)
    return Image.fromarray(expanded_rgb)


def enhance_image(img: Image.Image) -> Image.Image:
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]
    if w < TARGET_WIDTH or h < TARGET_HEIGHT:
        scale = max(TARGET_WIDTH / w, TARGET_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)
        cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    cv_img = cv2.fastNlMeansDenoisingColored(cv_img, None, 5, 5, 7, 21)
    img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    img = ImageEnhance.Contrast(img).enhance(1.1)
    img = ImageEnhance.Brightness(img).enhance(1.05)
    return img


def add_all_padding_blur(img: Image.Image) -> Image.Image:
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(cv_img, (151, 151), 50)
    expanded = cv2.copyMakeBorder(cv_img, 100, 100, 150, 150, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    background_resized = cv2.resize(blurred, (expanded.shape[1], expanded.shape[0]))
    expanded[:100, :] = background_resized[:100, :]
    expanded[-100:, :] = background_resized[-100:, :]
    expanded[:, :150] = background_resized[:, :150]
    expanded[:, -150:] = background_resized[:, -150:]
    expanded_rgb = cv2.cvtColor(expanded, cv2.COLOR_BGR2RGB)
    return Image.fromarray(expanded_rgb)


def process_image(file_stream: io.BytesIO) -> io.BytesIO:
    img = Image.open(file_stream).convert("RGB")
    img = enhance_image(img)
    img = expand_to_ratio(img, TARGET_RATIO)
    img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
    img = add_all_padding_blur(img)

    output_stream = io.BytesIO()
    img.save(output_stream, format="JPEG", quality=95)
    output_stream.seek(0)
    return output_stream
