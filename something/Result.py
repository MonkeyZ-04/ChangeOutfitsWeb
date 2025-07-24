# -*- coding: utf-8 -*-
"""
This script processes images in the 'OUTFITS' folder using the Segment Anything Model (SAM)
and the 'rembg' library to generate masks and remove backgrounds.

The script will:
1. Download the SAM model checkpoint if not already present.
2. Iterate through all .jpg images in the 'OUTFITS' folder.
3. For each image, generate segmentation masks using SAM.
4. Save the first two masks (mask 0 and mask 1) as PNG files in a 'MASKS' subfolder within 'OUTFITS'.
5. Use 'rembg' to remove the background from the entire image.
6. Save the background-removed image as a PNG file in a 'RESULT_CUT_BG' subfolder within 'OUTFITS'.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import requests
import tqdm
import ssl
import certifi
from PIL import Image
from io import BytesIO
import os
from rembg import remove
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# ==================================================
# 0) Configuration and Model Loading
# ==================================================

# Set the device for PyTorch (use CUDA if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# SAM model checkpoint details
CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"
CHECKPOINT_URL = f"https://dl.fbaipublicfiles.com/segment_anything/{CHECKPOINT_NAME}"
CHECKPOINT_PATH = pathlib.Path(CHECKPOINT_NAME)

# Download the checkpoint if it doesn't exist
if not CHECKPOINT_PATH.exists():
    print(f"Downloading {CHECKPOINT_NAME}...")
    try:
        # Add SSL context for verification to handle potential certificate issues
        ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())
        response = requests.get(CHECKPOINT_URL, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes
        
        # Get the total file size for the progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        # Use tqdm to show a progress bar during the download
        with open(CHECKPOINT_PATH, 'wb') as f:
            for chunk in tqdm.tqdm(response.iter_content(chunk_size=block_size), total=total_size // block_size, unit='KB'):
                f.write(chunk)
        print("Download finished.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the checkpoint: {e}")
        print("Please check your internet connection or URL.")
        exit()
else:
    print("Checkpoint already exists.")

# Initialize the SAM model
print("Initializing SAM model...")
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT_PATH).to(DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)
print("SAM model initialized.")

# ==================================================
# 1) Setup output directories inside OUTFITS folder
# ==================================================
outfits_dir = pathlib.Path("OUTFITS")
mask_folder = "MASKS"
result_folder = "RESULT_CUT_BG"

# Create directories if they don't exist
mask_folder.mkdir(parents=True, exist_ok=True)
result_folder.mkdir(parents=True, exist_ok=True)

print(f"✔️  Created directories: {mask_folder.resolve()} and {result_folder.resolve()} inside 'OUTFITS'")

# ==================================================
# 2) Process all images in the 'OUTFITS' folder
# ==================================================

# Check if the OUTFITS directory exists
if not outfits_dir.is_dir():
    print(f"Error: Directory '{outfits_dir}' not found. Please create it and add your images.")
else:
    # Iterate through all JPEG images in the folder
    # You can change "*.jpg" to "*.png" or other formats if needed
    image_files = list(outfits_dir.glob("*.jpg"))
    if not image_files:
        print(f"Warning: No .jpg images found in '{outfits_dir}'.")
    
    for img_path in image_files:
        print(f"\n==========================================")
        print(f"Processing image: {img_path.name}")
        print(f"==========================================")

        # A) Read the image using OpenCV
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"Warning: Could not read image {img_path.name}. Skipping.")
            continue
        
        # Convert the image from BGR (OpenCV) to RGB (PIL/Matplotlib)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # B) Generate segmentation masks using SAM
        print("Generating masks with SAM...")
        masks = mask_generator.generate(image_rgb)
        print(f"Generated {len(masks)} masks.")
        
        # C) Save the first two masks (mask 0 and mask 1)
        if len(masks) > 0:
            for i in range(min(2, len(masks))): # Loop for mask 0 and 1
                mask_data = masks[i]['segmentation']
                
                # Convert the mask (boolean array) to a grayscale image (0-255)
                mask_image = Image.fromarray(mask_data * 255).convert("L")
                
                # Create a filename for the mask
                mask_filename = f"{img_path.stem}_mask_{i}.png"
                mask_output_path = mask_folder / mask_filename
                
                # Save the mask image
                mask_image.save(mask_output_path)
                print(f"✔️  Saved mask {i} → {mask_output_path.resolve()}")
        else:
            print("No masks were generated by SAM.")

        # D) Use 'rembg' to remove the background from the entire image
        print("Removing background with rembg...")
        try:
            # Open the image using PIL and convert to RGBA for transparency
            pil_image = Image.open(img_path).convert("RGBA")
            
            # Remove the background
            bg_removed = remove(pil_image)
            
            # Ensure the output is a PIL Image object
            if isinstance(bg_removed, bytes):
                bg_removed = Image.open(BytesIO(bg_removed)).convert("RGBA")
            
            # Create a filename for the result
            result_filename = f"{img_path.stem}_cut_background.png"
            result_output_path = result_folder / result_filename
            
            # Save the background-removed image
            bg_removed.save(result_output_path)
            print(f"✔️  Saved result → {result_output_path.resolve()}")
            
        except Exception as e:
            print(f"Error processing image with rembg: {img_path.name} - {e}")

    print("\n✅ All images in 'OUTFITS' folder processed successfully.")