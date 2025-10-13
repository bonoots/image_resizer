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


def add_all_padding_blur(img: Image.Image, side_padding_ratio=0.10, top_bottom_padding_ratio=0.20) -> Image.Image:
    width, height = img.size
    pad_width = int(width * side_padding_ratio)
    pad_height = int(height * top_bottom_padding_ratio)
    new_width, new_height = width + 2 * pad_width, height + 2 * pad_height

    # Create blurred background
    background = img.filter(ImageFilter.GaussianBlur(radius=50)).resize((new_width, new_height), Image.LANCZOS)
    result = background.copy()
    result.paste(img, (pad_width, pad_height))

    # Optional: blend edges with a PIL mask for smooth transition (not as fancy as OpenCV, but no artifacts)
    mask = Image.new("L", (new_width, new_height), 0)
    blend = 40  # pixel width for fade
    # Top blend
    for y in range(blend):
        alpha = int(255 * (y / blend))
        mask.paste(alpha, box=(pad_width, y, pad_width + width, y + 1))
    # Bottom blend
    for y in range(blend):
        alpha = int(255 * (1 - y / blend))
        mask.paste(alpha, box=(pad_width, new_height - blend + y, pad_width + width, new_height - blend + y + 1))
    # Left/right blend
    for x in range(blend):
        alpha = int(255 * (x / blend))
        mask.paste(alpha, box=(x, pad_height, x + 1, pad_height + height))
        alpha = int(255 * (1 - x / blend))
        mask.paste(alpha, box=(new_width - blend + x, pad_height, new_width - blend + x + 1, pad_height + height))

    # Composite the sharp image onto the blurry background using the mask
    sharp = Image.new("RGB", (new_width, new_height))
    sharp.paste(img, (pad_width, pad_height))
    result = Image.composite(sharp, result, mask)

    return result


#def process_image(file_path: str, output_path: str):
#    try:
#        img = Image.open(file_path).convert("RGB")
#        img = expand_to_ratio(img, TARGET_RATIO)
        
        # Resize to target dimensions first
#        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
        
        # Calculate padding sizes
#        width, height = img.size
#        pad_width = int(width * 0.20)  # 10% side padding
#        pad_height = int(height * 0.20)  # 20% top/bottom padding
        
        # Create new canvas with padding
#        new_width = width + (pad_width * 2)
#        new_height = height + (pad_height * 2)
        
        # Create blurred background
#        blurred_bg = img.filter(ImageFilter.GaussianBlur(radius=50))
#        blurred_bg = blurred_bg.resize((new_width, new_height), Image.LANCZOS)
        
        # Paste original image on top of blurred background
#        blurred_bg.paste(img, (pad_width, pad_height))
        
        # Save to BytesIO
#        output_stream = io.BytesIO()
#        blurred_bg.save(output_stream, format="JPEG", quality=100)
#        output_stream.seek(0)
#        return output_stream
    
#    except Exception as e:
#        print(f"Error processing {file_path}: {e}")
#        return None




def process_image(file_path: str, output_path: str):
    try:
        img = Image.open(file_path).convert("RGB")
        img = enhance_image(img)
        
        # Step 1: Expand to 16:9 ratio
        img = expand_to_ratio(img, TARGET_RATIO)
        
        # Step 2: Resize to target dimensions
        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
        
        # Step 3: Add all padding at once with seamless corner blending
        img = add_all_padding_blur(img, side_padding_ratio=0.10, top_bottom_padding_ratio=0.20)   

        output_stream = io.BytesIO()
        img.save(output_stream, format="JPEG", quality=95)
        output_stream.seek(0)
        return output_stream

#def process_image(file_path: str, output_path: str):
#    try:
#        # Simple version to test
#        img = Image.open(file_path).convert("RGB")
#        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
        
#        output_stream = io.BytesIO()
#        img.save(output_stream, format="JPEG", quality=95)
#        output_stream.seek(0)
#        return output_stream
    
    except UnidentifiedImageError:
        raise ValueError(f"Cannot identify image file: {file_path}")
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")
    
