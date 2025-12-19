from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import shutil
import os
import cv2
import numpy as np

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Model
MODEL_PATH = "yolov8n.pt" 
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

IGNORED_CLASSES = [0, 1, 2, 3, 5, 7, 15, 16] 

def get_adaptive_road_mask(image):
    """
    Adaptive Road Segmentation:
    1. Samples color statistics from the bottom-center (safe road area).
    2. Thresholds the entire image based on those variance statistics.
    3. Keeps only the largest connected component touching the bottom (The Road).
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 1. Define ROI: Bottom Center (assumed to be road)
    roi_w = int(w * 0.2)
    roi_h = int(h * 0.1)
    roi_x = (w - roi_w) // 2
    roi_y = h - roi_h
    
    roi = hsv[roi_y:h, roi_x:roi_x+roi_w]
    
    # 2. Calculate Stats
    # We care mostly about Hue (color) and Saturation (vibrancy). 
    # Value (brightness) varies a lot on roads (shadows vs sun), so be loose there.
    mean = np.mean(roi, axis=(0,1))
    std = np.std(roi, axis=(0,1))
    
    # Tunable Tolerances
    # Hue: Road is usually neutral, but can be brownish/yellowish.
    # Sat: Road is usually low sat.
    # Val: High variance allowed.
    
    # Heuristic Tweaks:
    # If std dev is very small (homogeneous), expand it a bit to be robust.
    std[0] = max(std[0], 10) # Min std dev for Hue
    std[1] = max(std[1], 15) # Min std dev for Sat
    std[2] = max(std[2], 50) # Min std dev for Val (allow shadows)
    
    lower_bound = mean - (std * 3.5) # 3.5 sigma
    upper_bound = mean + (std * 3.5)
    
    # Clamp
    lower_bound = np.clip(lower_bound, 0, 255)
    upper_bound = np.clip(upper_bound, 0, 255)
    
    # Modify Value range specifically to allow shadows/highlights
    lower_bound[2] = max(0, mean[2] - 100) 
    upper_bound[2] = 255
    
    # 3. Create Color Mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # 4. Filter for Connectivity (Identify the Road Blob)
    # Morph open to disconnect trees from road if potential overlap
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros((h,w), dtype=np.uint8) # No road found?
    
    cleaned_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Keep only contours that intersect with the bottom ROI or are essentially large
    # Strategy: Find largest contour. 
    max_area = 0
    best_cnt = None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            # Check if it touches bottom half?
            x,y,w_c,h_c = cv2.boundingRect(cnt)
            if (y + h_c) > (h * 0.8): # Must be near bottom
                max_area = area
                best_cnt = cnt
                
    if best_cnt is not None:
        cv2.drawContours(cleaned_mask, [best_cnt], -1, 255, -1)
        
    # Also explicitly add Yellow/White lines back in case they were filtered out by stats
    # (Lines might be brighter/yellow-er than the asphalt sample)
    # Define generic line colors in HSV
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    lines_mask = cv2.bitwise_or(yellow_mask, white_mask)
    
    # Combine: (Adaptive Road) OR (Lines), but restricted to basically the road area
    # Dilation of road mask to catch lines on edge?
    cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=2)
    final_mask = cv2.bitwise_or(cleaned_mask, lines_mask)
    
    # Final Geometric Clip (Trapezoid) to kill sky artifacts if any remain
    geo_mask = np.zeros((h, w), dtype=np.uint8)
    points = np.array([
        [0, h],
        [w, h],
        [w, int(h * 0.35)], # Allow horizon
        [0, int(h * 0.35)]
    ])
    cv2.fillPoly(geo_mask, [points], 255)
    final_mask = cv2.bitwise_and(final_mask, geo_mask)
    
    return final_mask

def detect_damage_texture(image, grid_rows=5, grid_cols=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = get_adaptive_road_mask(image)
    
    # Use Bilateral Filter instead of Gaussian
    # This keeps edges (cracks) sharp but removes surface texture (grain)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Stricter Canny Thresholds to ignore fine grain
    edges = cv2.Canny(blurred, 60, 180)
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
    
    h, w = edges_masked.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    
    grid_status = np.zeros((grid_rows, grid_cols), dtype=int)
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            x_start = c * cell_w
            y_start = r * cell_h
            x_end = (c + 1) * cell_w
            y_end = (r + 1) * cell_h
            
            # Check mask coverage
            cell_mask = mask[y_start:y_end, x_start:x_end]
            if cv2.countNonZero(cell_mask) < (cell_mask.size * 0.25): 
                grid_status[r, c] = 0 
                continue

            cell_roi = edges_masked[y_start:y_end, x_start:x_end]
            edge_score = np.mean(cell_roi)
            
            # Higher Threshold: Requires significant edge density (actual cracks)
            # Normal road grain gives ~2-5. Cracks give > 15.
            if edge_score > 20: 
                grid_status[r, c] = 1 
                
    return grid_status, mask

def draw_hybrid_analysis(image, boxes, grid_rows=5, grid_cols=5):
    h, w, _ = image.shape
    overlay = image.copy()
    
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    
    texture_grid, road_mask = detect_damage_texture(image, grid_rows, grid_cols)
    final_grid = texture_grid.copy()
    
    if boxes is not None:
        for box in boxes:
            cls = int(box.cls[0])
            if cls in IGNORED_CLASSES:
                continue 
                
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            if cy < h and cx < w and road_mask[cy, cx] == 0:
                continue

            col = int(cx // cell_w)
            row = int(cy // cell_h)
            col = max(0, min(col, grid_cols - 1))
            row = max(0, min(row, grid_rows - 1))
            final_grid[row, col] = 1

    # Draw Grid
    alpha = 0.45
    for r in range(grid_rows):
        for c in range(grid_cols):
            x_start = c * cell_w
            y_start = r * cell_h
            x_end = (c + 1) * cell_w
            y_end = (r + 1) * cell_h
            
            cell_mask = road_mask[y_start:y_end, x_start:x_end]
            if cv2.countNonZero(cell_mask) < (cell_mask.size * 0.25):
                continue
            
            color = (0, 255, 0) # Green
            if final_grid[r, c] == 1:
                color = (0, 0, 255) # Red
            
            cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), color, -1)
            cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (255, 255, 255), 1)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image, np.sum(final_grid) > 0

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if not file or file.filename == "":
        return templates.TemplateResponse("index.html", {"request": request, "error": "No file selected"})

    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if model is None:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Model not loaded"})

    try:
        results = model(file_path)
        original_img = cv2.imread(file_path)
        boxes = results[0].boxes
        
        processed_img, has_damage = draw_hybrid_analysis(original_img, boxes)
        
        if has_damage:
            status_text = "⚠️ Damage Detected! Maintenance Recommended."
            status_color = "red"
        else:
            status_text = "No Damage Detected - Road Surface Integrity: 100%"
            status_color = "green"
        
        output_filename = f"result_{file.filename}"
        output_path = os.path.join(upload_dir, output_filename)
        cv2.imwrite(output_path, processed_img)
        
        original_url = f"/static/uploads/{file.filename}"
        result_url = f"/static/uploads/{output_filename}"
        
        return templates.TemplateResponse("result.html", {
            "request": request, 
            "original_image": original_url,
            "result_image": result_url,
            "status_text": status_text,
            "status_color": status_color
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Error: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
