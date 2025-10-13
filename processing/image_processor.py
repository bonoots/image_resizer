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
    """
    Adds padding on all sides with blurred background and seamless corner blending.
    """
    width, height = img.size
    pad_width = int(width * side_padding_ratio)
    pad_height = int(height * top_bottom_padding_ratio)
    
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Create heavily blurred background
    blurred = cv2.blur(cv_img, (201, 201))
    blurred = cv2.GaussianBlur(blurred, (151, 151), 50)
    
    # Add padding on all sides at once
    expanded = cv2.copyMakeBorder(cv_img, pad_height, pad_height, pad_width, pad_width, 
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # Resize blurred background to match expanded dimensions
    background_resized = cv2.resize(blurred, (expanded.shape[1], expanded.shape[0]), 
                                     interpolation=cv2.INTER_LINEAR)
    
    # Apply blurred background to all padded areas
    expanded[0:pad_height, :] = background_resized[0:pad_height, :]
    expanded[-pad_height:, :] = background_resized[-pad_height:, :]
    expanded[pad_height:-pad_height, 0:pad_width] = background_resized[pad_height:-pad_height, 0:pad_width]
    expanded[pad_height:-pad_height, -pad_width:] = background_resized[pad_height:-pad_height, -pad_width:]
    
    # Create smooth gradient masks for edge transitions
    h, w = expanded.shape[:2]
    
    # Create distance transform masks for smoother transitions
    # Vertical gradient (for top/bottom edges)
    vert_mask = np.ones((h, w), dtype=np.float32)
    fade_zone = max(pad_height, pad_width)
    
    for y in range(h):
        if y < fade_zone:
            vert_mask[y, :] = y / fade_zone
        elif y > h - fade_zone:
            vert_mask[y, :] = (h - y) / fade_zone
    
    # Horizontal gradient (for left/right edges)
    horiz_mask = np.ones((h, w), dtype=np.float32)
    
    for x in range(w):
        if x < fade_zone:
            horiz_mask[:, x] = x / fade_zone
        elif x > w - fade_zone:
            horiz_mask[:, x] = (w - x) / fade_zone
    
    # Combine masks using minimum (creates smooth corners)
    combined_mask = np.minimum(vert_mask, horiz_mask)
    
    # Apply multiple blur passes to the entire padding area for ultra-smooth transitions
    padding_area = expanded.copy()
    padding_area = cv2.GaussianBlur(padding_area, (151, 151), 75)
    padding_area = cv2.GaussianBlur(padding_area, (99, 99), 50)
    
    # Blend using the smooth mask
    mask_3channel = np.stack([combined_mask, combined_mask, combined_mask], axis=2)
    #expanded = (expanded * mask_3channel + padding_area * (1 - mask_3channel)).astype(np.uint8)
    expanded = cv2.convertScaleAbs(expanded * mask_3channel + padding_area * (1 - mask_3channel))

    # Final smoothing pass on the entire image to eliminate any remaining seams
    expanded = cv2.GaussianBlur(expanded, (51, 51), 20)
    
    # Restore the center (original image area) to maintain sharpness
    center_y_start = pad_height
    center_y_end = h - pad_height
    center_x_start = pad_width
    center_x_end = w - pad_width
    
    # Create a gradient mask for the center restoration
    center_mask = np.zeros((h, w), dtype=np.float32)
    blend_distance = 30  # pixels to blend at the edge
    
    for y in range(h):
        for x in range(w):
            # Distance from the content edge
            dist_to_content = min(
                abs(y - center_y_start),
                abs(y - center_y_end),
                abs(x - center_x_start),
                abs(x - center_x_end)
            )
            
            if y >= center_y_start and y < center_y_end and x >= center_x_start and x < center_x_end:
                # Inside content area
                center_mask[y, x] = min(1.0, dist_to_content / blend_distance)
            else:
                center_mask[y, x] = 0
    
    # Restore center with gradient blend
    center_mask_3channel = np.stack([center_mask, center_mask, center_mask], axis=2)
    cv_img_padded = cv2.copyMakeBorder(cv_img, pad_height, pad_height, pad_width, pad_width, 
                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])
#    expanded = (cv_img_padded * center_mask_3channel + expanded * (1 - center_mask_3channel)).astype(np.uint8)
#    expanded = cv2.convertScaleAbs(expanded * mask_3channel + padding_area * (1 - mask_3channel))
#    expanded = cv2.convertScaleAbs(expanded * mask_3channel + padding_area * (1 - mask_3channel))

#    expanded = cv2.addWeighted(expanded, 1.0, padding_area, 0.0, 0)
#    blended = np.zeros_like(expanded)
#    for i in range(3):  # For each color channel
#        blended[:,:,i] = (expanded[:,:,i] * mask_3channel[:,:,i] + 
#                        padding_area[:,:,i] * (1 - mask_3channel[:,:,i]))
#    expanded = blended.astype(np.uint8)


    expanded_rgb = cv2.cvtColor(expanded, cv2.COLOR_BGR2RGB)
    return Image.fromarray(expanded_rgb)
    #return Image.fromarray(expanded)


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
    
