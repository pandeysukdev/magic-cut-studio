from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
from rembg import remove, new_session
from PIL import Image
import io
import numpy as np
import cv2
import time
import uvicorn
import gc  # For memory management

app = FastAPI(title="Magic Cut Studio")

# Allow everyone to use this tool
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create downloads folder
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Load AI model once (cached for speed)
# Using u2net - smaller and works on 512MB RAM
session_cache = new_session("u2net")

# Cache system - makes it super fast!
GLOBAL_CACHE = {
    "filename": None,
    "alpha_channel": None,
    "ai_clean_img": None,
    "source_pil": None
}

@app.get("/")
async def serve_frontend():
    """Serve the main website"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Website files not found</h1>", status_code=404)

@app.post("/api/remove-bg")
async def remove_background(
    file: UploadFile = File(...),
    full_bg_only: str = Form("false"),
    click_x: int = Form(-1),
    click_y: int = Form(-1)
):
    """Remove background from image"""
    try:
        start_time = time.time()
        img_bytes = await file.read()
        filename = file.filename
        
        # Check if we already processed this image (CACHE)
        if GLOBAL_CACHE["filename"] == filename and GLOBAL_CACHE["alpha_channel"] is not None:
            source_pil = GLOBAL_CACHE["source_pil"]
            ai_clean_img = GLOBAL_CACHE["ai_clean_img"]
            alpha_channel = GLOBAL_CACHE["alpha_channel"]
        else:
            # First time - process the image
            source_pil = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
            
            # Remove background with professional quality
            cleaned_bytes = remove(
                img_bytes, 
                session=session_cache, 
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=1
            )
            ai_clean_img = Image.open(io.BytesIO(cleaned_bytes)).convert("RGBA")
            
            ai_np = np.array(ai_clean_img)
            GLOBAL_CACHE["filename"] = filename
            GLOBAL_CACHE["source_pil"] = source_pil
            GLOBAL_CACHE["ai_clean_img"] = ai_clean_img
            GLOBAL_CACHE["alpha_channel"] = ai_np[:, :, 3]
            
            alpha_channel = GLOBAL_CACHE["alpha_channel"]

        W, H = source_pil.size

        # MODE 1: Remove whole background
        if full_bg_only.lower() == "true":
            output_img = ai_clean_img
            
        # MODE 2: Click to select specific object
        else:
            target_x = max(0, min(int(click_x), W - 1))
            target_y = max(0, min(int(click_y), H - 1))
            
            _, binary_mask = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            clicked_label = labels[target_y, target_x]
            
            # If clicked outside, find nearest object
            if clicked_label == 0:
                min_dist = float("inf")
                best_label = 0
                for label_idx in range(1, num_labels):
                    x_stat = stats[label_idx, cv2.CC_STAT_LEFT]
                    y_stat = stats[label_idx, cv2.CC_STAT_TOP]
                    w_stat = stats[label_idx, cv2.CC_STAT_WIDTH]
                    h_stat = stats[label_idx, cv2.CC_STAT_HEIGHT]
                    
                    dx = max(0, x_stat - target_x, target_x - (x_stat + w_stat))
                    dy = max(0, y_stat - target_y, target_y - (y_stat + h_stat))
                    dist = (dx**2 + dy**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_label = label_idx
                clicked_label = best_label

            if clicked_label > 0:
                component_mask = np.where(labels == clicked_label, 255, 0).astype(np.uint8)
                final_smooth_alpha = cv2.bitwise_and(alpha_channel, component_mask)
            else:
                final_smooth_alpha = alpha_channel

            # Create final image
            source_np = np.array(source_pil)
            isolated_np = source_np.copy()
            isolated_np[:, :, 3] = final_smooth_alpha
            output_img = Image.fromarray(isolated_np)
            
            # Crop to selected object
            bbox = output_img.getbbox()
            if bbox:
                output_img = output_img.crop(bbox)

        # Save result
        output_path = os.path.join(DOWNLOAD_DIR, "cutout_result.png")
        output_img.save(output_path, "PNG")
        
        processing_time = round((time.time() - start_time) * 1000)
        
        # Free up memory
        gc.collect()
        
        # Return the image with processing time
        response = FileResponse(output_path, media_type="image/png")
        response.headers["X-Processing-Time"] = str(processing_time)
        return response
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
