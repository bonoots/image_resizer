import io
from typing import BinaryIO
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from PIL import UnidentifiedImageError

# Target output size and ratio
TARGET_WIDTH = 1600
TARGET_HEIGHT = 900
TARGET_RATIO = TARGET_WIDTH / TARGET_HEIGHT

def autocrop_black_borders(img: Image.Image, threshold=255):
    """
    Automatically crops black (or near-black) borders from the image.
    threshold: pixel values less than this (0-255) are considered black.
    """
    img_np = np.array(img)
    if img_np.ndim == 3:
        gray = np.mean(img_np, axis=2)
    else:
        gray = img_np

    # Find non-black rows and columns
    mask = gray > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img  # Nothing to crop

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # Slices are exclusive at the top
    cropped = img.crop((x0, y0, x1, y1))
    return cropped

def expand_to_ratio(img: Image.Image, target_ratio: float) -> Image.Image:
    """Expands the image canvas to the target aspect ratio using blurred background."""
    width, height = img.size
    current_ratio = width / height

    if abs(current_ratio - target_ratio) < 0.01:
        return img  # Already correct ratio

    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if current_ratio > target_ratio:
        # Image too wide → pad top/bottom
        new_height = int(width / target_ratio)
        pad = (new_height - height) // 2

        background = cv2.blur(cv_img, (201, 201))
        background = cv2.GaussianBlur(background, (151, 151), 50)

        expanded = cv2.copyMakeBorder(cv_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        background_resized = cv2.resize(background, (expanded.shape[1], expanded.shape[0]), interpolation=cv2.INTER_LINEAR)

        expanded[0:pad, :] = background_resized[0:pad, :]
        expanded[-pad:, :] = background_resized[-pad:, :]

        cv_img = expanded
    else:
        # Image too tall → pad left/right
        new_width = int(height * target_ratio)
        pad = (new_width - width) // 2

        background = cv2.blur(cv_img, (201, 201))
        background = cv2.GaussianBlur(background, (151, 151), 50)

        expanded = cv2.copyMakeBorder(cv_img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        background_resized = cv2.resize(background, (expanded.shape[1], expanded.shape[0]), interpolation=cv2.INTER_LINEAR)

        expanded[:, 0:pad] = background_resized[:, 0:pad]
        expanded[:, -pad:] = background_resized[:, -pad:]

        cv_img = expanded

    expanded_rgb = cv2.cvtColor(expanded, cv2.COLOR_BGR2RGB)
    return Image.fromarray(expanded_rgb)

def enhance_image(img: Image.Image) -> Image.Image:
    """Improves image quality: upscale, sharpen, contrast, brightness."""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    # Upscale smaller images before processing
    if w < TARGET_WIDTH or h < TARGET_HEIGHT:
        scale_x = TARGET_WIDTH / w
        scale_y = TARGET_HEIGHT / h
        scale = max(scale_x, scale_y)
        new_w, new_h = int(w * scale), int(h * scale)
        cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Light denoise (good for PNG logos or compressed images)
    cv_img = cv2.fastNlMeansDenoisingColored(cv_img, None, 5, 5, 7, 21)

    img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    # Sharpness
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    # Contrast
    img = ImageEnhance.Contrast(img).enhance(1.1)
    # Brightness
    img = ImageEnhance.Brightness(img).enhance(1.05)

    return img

def create_radial_fade_mask(h, w, pad_height, pad_width, blend_px):
    """Create a radial fade mask for seamless corner blending."""
    y0, y1 = pad_height, h - pad_height
    x0, x1 = pad_width, w - pad_width

    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    dy = np.maximum(np.maximum(y0 - y, y - y1 + 1), 0)
    dx = np.maximum(np.maximum(x0 - x, x - x1 + 1), 0)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    mask = np.clip(dist / blend_px, 0, 1)
    return mask.astype(np.float32)

def create_content_alpha_mask(h, w, pad_height, pad_width, blend_px):
    """Create a mask: 1 in center, 0 in padding, smooth transition in between."""
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    y0, y1 = pad_height, h - pad_height
    x0, x1 = pad_width, w - pad_width
    # Distance to the closest edge of the content rectangle
    dy = np.minimum(y - y0, y1 - y - 1)
    dx = np.minimum(x - x0, x1 - x - 1)
    dist = np.minimum(dy, dx)
    mask = np.clip((dist / blend_px), 0, 1)
    return mask.astype(np.float32)

def add_all_padding_blur(img: Image.Image, side_padding_ratio=0.10, top_bottom_padding_ratio=0.20, blend_px=40) -> Image.Image:
    """
    Adds padding on all sides with blurred background and seamless corner blending using a radial mask,
    and smooths the edge of the original image using a soft content mask.
    """
    width, height = img.size
    pad_width = int(width * side_padding_ratio)
    pad_height = int(height * top_bottom_padding_ratio)
    new_width = width + 2 * pad_width
    new_height = height + 2 * pad_height

    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Create heavily blurred background
    blurred = cv2.blur(cv_img, (201, 201))
    blurred = cv2.GaussianBlur(blurred, (151, 151), 50)

    # Add padding on all sides at once
    expanded = cv2.copyMakeBorder(cv_img, pad_height, pad_height, pad_width, pad_width,
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Resize blurred background to match expanded dimensions
    background_resized = cv2.resize(blurred, (expanded.shape[1], expanded.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Radial fade for seamless corner blending
    mask = create_radial_fade_mask(expanded.shape[0], expanded.shape[1], pad_height, pad_width, blend_px)
    mask_3c = np.stack([mask]*3, axis=2)
    expanded = (expanded * (1 - mask_3c) + background_resized * mask_3c).astype(np.uint8)

    # Soft blend for the center region
    y_start, y_end = pad_height, pad_height + height
    x_start, x_end = pad_width, pad_width + width

    content_mask = create_content_alpha_mask(expanded.shape[0], expanded.shape[1], pad_height, pad_width, blend_px)
    content_mask_3c = np.stack([content_mask]*3, axis=2)
    cv_img_padded = np.zeros_like(expanded)
    cv_img_padded[y_start:y_end, x_start:x_end] = cv_img

    expanded = (expanded * (1 - content_mask_3c) + cv_img_padded * content_mask_3c).astype(np.uint8)

    expanded_rgb = cv2.cvtColor(expanded, cv2.COLOR_BGR2RGB)
    return Image.fromarray(expanded_rgb)

def process_image(file_path: str, output_path: str):
    try:
        img = Image.open(file_path).convert("RGB")
        img = autocrop_black_borders(img)  # <--- NEW: remove black borders before processing
        img = enhance_image(img)

        # Step 1: Expand to 16:9 ratio
        img = expand_to_ratio(img, TARGET_RATIO)

        # Step 2: Resize to target dimensions
        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)

        # Step 3: Add all padding at once with seamless corner blending and smooth content transition
        img = add_all_padding_blur(img, side_padding_ratio=0.10, top_bottom_padding_ratio=0.20, blend_px=40)

        output_stream = io.BytesIO()
        img.save(output_stream, format="JPEG", quality=95)
        output_stream.seek(0)
        return output_stream

    except UnidentifiedImageError:
        raise ValueError(f"Cannot identify image file: {file_path}")
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")
