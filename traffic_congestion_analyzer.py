import cv2
import numpy as np
import argparse
import json
from ultralytics import YOLO
from collections import defaultdict, deque
from tqdm import tqdm

# --- CONFIGURATION ---
TARGET_H = 720
TARGET_W = 1280
DASHBOARD_W = 400
VIDEO_W = TARGET_W - DASHBOARD_W  # 880px

# Target Classes (Vehicles)
TARGET_CLASSES = [1, 2, 3, 4, 5, 7]

CLASS_NAME_MAP = {
    'bicycle': 'bicycle',
    'car': 'car',
    'motorcycle': '2wheeler',
    'bus': 'bus',
    'truck': 'truck',
    '3wheeler': '3wheeler'
}

# UPDATED COLORS (BGR) - Truck is Deep Purple
CLASS_COLORS = {
    'bicycle': (255, 191, 0),    # Deep Sky Blue
    'car': (0, 255, 255),        # Yellow
    '2wheeler': (0, 140, 255),   # Orange
    'bus': (255, 0, 255),        # Magenta
    'truck': (128, 0, 128),      # Deep Purple
    '3wheeler': (0, 255, 127)    # Spring Green
}

# SMOOTHING SETTINGS
BUFFER_SIZE = 10  # Number of frames to average (Higher = Smoother but slower reaction)

# DEFAULT REGIONS (Fallback if no config file is provided)
DEFAULT_REGIONS = [
    # Region 1
    [
        [412, 409],
        [649, 404],
        [846, 704],
        [340, 711],
        [413, 411]
    ],
    # Region 2
    [
        [677, 407],
        [879, 392],
        [874, 684],
        [678, 406]
    ],
    # Region 3
    [
        [127, 416],
        [264, 431],
        [73, 712],
        [4, 713],
        [0, 495],
        [123, 417]
    ]
]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Traffic Congestion Renderer (Smoothed)")
    parser.add_argument("--source", type=str, required=True, help="Input video path")
    parser.add_argument("--config", type=str, default=None, help="Path to regions_config.json (optional)")
    parser.add_argument("--draw", action="store_true", help="Draw regions interactively on the first frame of the video")
    parser.add_argument("--save-config", type=str, default=None, help="Save drawn regions to a JSON config file")
    parser.add_argument("--output", type=str, default="final_output.avi", help="Output video path")
    parser.add_argument("--weights", type=str, default="yolov8x.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    
    # Congestion Thresholds
    parser.add_argument("--min-count", type=int, default=5, help="Min vehicles to check congestion")
    parser.add_argument("--congestion-ratio", type=float, default=0.70, help="Occupancy threshold (0.0 - 1.0)")
    
    return parser.parse_args()

class ServerRenderer:
    def __init__(self, args):
        print(f"Loading model: {args.weights}...")
        self.model = YOLO(args.weights)
        self.args = args
        self.regions = []
        
        # Load regions based on arguments (Draw mode defers region initialization)
        if not args.draw:
            if args.config:
                self.load_config(args.config)
            else:
                print("No config file provided. Using default regions.")
                self.load_default_regions()
            self.init_buffers()

    def init_buffers(self):
        """Initializes deques and dictionaries used for temporal smoothing."""
        # 1. Buffers for Congestion Logic (Total Count & Pixel Ratio)
        self.occupancy_buffers = [deque(maxlen=BUFFER_SIZE) for _ in self.regions]
        self.total_count_buffers = [deque(maxlen=BUFFER_SIZE) for _ in self.regions]

        # 2. Buffers for Dashboard Numbers (To prevent graph flickering)
        self.region_class_buffers = []
        for _ in self.regions:
            # Create a buffer for every possible class
            class_buffer = defaultdict(lambda: deque(maxlen=BUFFER_SIZE))
            self.region_class_buffers.append(class_buffer)

    def load_config(self, path):
        print(f"Loading configuration from {path}...")
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            for r_data in data['regions']:
                points = np.array(r_data['points'], np.int32)
                self.regions.append(points)
            print(f"Successfully loaded {len(self.regions)} regions.")
        except Exception as e:
            print(f"Error loading config: {e}")
            exit(1)

    def load_default_regions(self):
        for points_list in DEFAULT_REGIONS:
            points = np.array(points_list, np.int32)
            self.regions.append(points)
        print(f"Loaded {len(self.regions)} default regions.")

    def save_regions_to_config(self, path):
        """Saves current regions to a JSON configuration file."""
        config_data = {
            "video_dim": {
                "width": VIDEO_W,
                "height": TARGET_H
            },
            "regions": []
        }
        for idx, poly in enumerate(self.regions):
            points_list = poly.tolist()
            config_data["regions"].append({
                "id": idx + 1,
                "name": f"Region {idx + 1}",
                "points": points_list
            })
        try:
            with open(path, 'w') as f:
                json.dump(config_data, f, indent=4)
            print(f"Successfully saved regions configuration to {path}")
        except Exception as e:
            print(f"Error saving config file: {e}")

    def draw_regions_interactively(self, frame):
        """Allows the user to draw region polygons interactively using mouse clicks."""
        print("\n" + "="*50)
        print(" INTERACTIVE REGION DRAWER MODE")
        print("="*50)
        print("Instructions:")
        print("  1. Left-Click: Add a vertex to the current region polygon.")
        print("  2. 'n' Key: Save the current region and start a new one.")
        print("  3. 'c' Key: Clear the current region's vertices.")
        print("  4. 's' Key: Finish drawing, save regions, and start processing.")
        print("  5. 'q' Key: Quit without processing.")
        print("="*50 + "\n")

        window_name = "Draw Regions - Click to add points"
        cv2.namedWindow(window_name)

        current_points = []
        finalized_regions = []
        mouse_pos = [0, 0]

        def mouse_callback(event, x, y, flags, param):
            nonlocal current_points
            if event == cv2.EVENT_LBUTTONDOWN:
                current_points.append([x, y])
                print(f"Added point: ({x}, {y})")
            elif event == cv2.EVENT_MOUSEMOVE:
                mouse_pos[0] = x
                mouse_pos[1] = y

        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            # Draw on a copy of the frame
            display_frame = frame.copy()

            # Draw finalized regions
            for idx, poly in enumerate(finalized_regions):
                cv2.polylines(display_frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
                # Draw centroid text label
                M = cv2.moments(poly)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    cv2.putText(display_frame, f"Region {idx+1}", (cx - 30, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw current active region points and lines
            if len(current_points) > 0:
                pts = np.array(current_points, np.int32)
                cv2.polylines(display_frame, [pts], isClosed=False, color=(0, 165, 255), thickness=2)
                cv2.line(display_frame, tuple(current_points[-1]), tuple(mouse_pos), (0, 165, 255), 1, cv2.LINE_AA)
                for pt in current_points:
                    cv2.circle(display_frame, tuple(pt), 4, (0, 0, 255), -1)

            # Show HUD / instructions overlay
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (330, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
            cv2.putText(display_frame, "Left-Click: Add Point", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, "'n': Finish Region & Start New", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, "'c': Clear Active Points", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, "'s': Save & Start Processing", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('n') or key == ord('N'):
                if len(current_points) >= 3:
                    finalized_regions.append(np.array(current_points, np.int32))
                    print(f"Saved Region {len(finalized_regions)} with {len(current_points)} points.")
                    current_points = []
                else:
                    print("Error: A region polygon must have at least 3 points!")

            elif key == ord('c') or key == ord('C'):
                current_points = []
                print("Cleared active region points.")

            elif key == ord('s') or key == ord('S'):
                if len(current_points) >= 3:
                    finalized_regions.append(np.array(current_points, np.int32))
                    print(f"Saved Region {len(finalized_regions)} with {len(current_points)} points.")
                    current_points = []
                
                if len(finalized_regions) > 0:
                    self.regions = finalized_regions
                    print(f"Saved {len(self.regions)} regions total. Starting processing...")
                    break
                else:
                    print("Error: Draw at least one region polygon before saving!")

            elif key == ord('q') or key == ord('Q'):
                print("Exited drawer mode. Quitting program.")
                cv2.destroyAllWindows()
                exit(0)

        cv2.destroyAllWindows()

    def calculate_occupancy(self, frame_shape, region_poly, box_list):
        """Calculates pixel occupancy ratio."""
        mask_region = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_region, [region_poly], 255)
        area_region = cv2.countNonZero(mask_region)
        if area_region == 0: return 0.0

        mask_vehicles = np.zeros(frame_shape[:2], dtype=np.uint8)
        for box in box_list:
            x1, y1, x2, y2 = box
            cv2.rectangle(mask_vehicles, (x1, y1), (x2, y2), 255, -1)
        
        intersection = cv2.bitwise_and(mask_vehicles, mask_region)
        return cv2.countNonZero(intersection) / area_region

    def draw_text_centered(self, img, text, center_x, y, font_scale=0.6, color=(255, 255, 255), thickness=1):
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        x = center_x - (size[0] // 2)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    def draw_pie_chart(self, img, counts, center, radius):
        total = sum(counts.values())
        if total == 0:
            cv2.circle(img, center, radius, (60, 60, 60), 1, cv2.LINE_AA)
            return
        start_angle = 0
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for cls, count in sorted_counts:
            if count == 0: continue
            angle = (count / total) * 360
            color = CLASS_COLORS.get(cls, (200, 200, 200))
            cv2.ellipse(img, center, (radius, radius), 0, start_angle, start_angle + angle, color, -1, cv2.LINE_AA)
            start_angle += angle
        cv2.circle(img, center, int(radius * 0.5), (40, 40, 40), -1, cv2.LINE_AA)

    def draw_status_legend(self, frame):
        """Top Right Legend for Color Meanings."""
        h, w = frame.shape[:2]
        box_w, box_h = 240, 100
        x1, y1 = w - box_w - 20, 20
        x2, y2 = w - 20, 20 + box_h
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        cv2.circle(frame, (x1 + 20, y1 + 25), 6, (0, 255, 0), -1)
        cv2.putText(frame, f"Free: < {self.args.min_count} Veh", (x1 + 40, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        pct = int(self.args.congestion_ratio * 100)
        cv2.circle(frame, (x1 + 20, y1 + 50), 6, (0, 165, 255), -1)
        cv2.putText(frame, f"Busy: < {pct}% Occ", (x1 + 40, y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.circle(frame, (x1 + 20, y1 + 75), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"Jam: > {pct}% Occ", (x1 + 40, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def create_dashboard_frame(self, counts_per_region):
        dash = np.zeros((TARGET_H, DASHBOARD_W, 3), dtype=np.uint8)
        cv2.rectangle(dash, (0, 0), (DASHBOARD_W, TARGET_H), (40, 40, 40), -1)
        
        # Header
        cv2.rectangle(dash, (0, 0), (DASHBOARD_W, 60), (30, 30, 30), -1)
        self.draw_text_centered(dash, "TRAFFIC ANALYTICS", DASHBOARD_W // 2, 40, 0.9, (0, 255, 255), 2)
        
        curr_y = 80
        # Color Legend
        legend_items = list(CLASS_COLORS.items())
        col_start_x = [60, 220]
        row_height = 30
        for i, (cls_name, color) in enumerate(legend_items):
            col = i % 2
            row = i // 2
            x = col_start_x[col]
            y = curr_y + (row * row_height)
            cv2.circle(dash, (x, y), 6, color, -1, cv2.LINE_AA)
            cv2.putText(dash, cls_name, (x + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        
        curr_y += (3 * row_height) + 20
        # Histogram
        cv2.line(dash, (20, curr_y), (DASHBOARD_W-20, curr_y), (100,100,100), 1)
        curr_y += 30
        self.draw_text_centered(dash, "TOTAL TRAFFIC VOLUME", DASHBOARD_W // 2, curr_y, 0.6, (255, 255, 255), 1)
        curr_y += 30
        
        totals = [sum(c.values()) for c in counts_per_region]
        max_val = max(totals) if totals and max(totals) > 0 else 10
        
        bar_area_h = 100
        num_regions = max(1, len(self.regions))
        bar_w = (DASHBOARD_W - 80) // num_regions
        if bar_w > 80: bar_w = 80
        
        for i, total in enumerate(totals):
            h_bar = int((total / max_val) * bar_area_h)
            x_bar = 40 + (i * (bar_w + 15))
            y_base = curr_y + bar_area_h
            
            color = (0, 255, 0)
            if total > 15: color = (0, 255, 255)
            if total > 30: color = (0, 0, 255)
            
            cv2.rectangle(dash, (x_bar, y_base - h_bar), (x_bar + bar_w, y_base), color, -1)
            cv2.putText(dash, str(total), (x_bar + (bar_w//2) - 5, y_base - h_bar - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(dash, f"R{i+1}", (x_bar + (bar_w//2) - 10, y_base + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        curr_y += bar_area_h + 50
        # Pie Charts
        cv2.line(dash, (20, curr_y - 20), (DASHBOARD_W-20, curr_y - 20), (100,100,100), 1)
        self.draw_text_centered(dash, "REGIONAL BREAKDOWN", DASHBOARD_W // 2, curr_y, 0.6, (255, 255, 255), 1)
        curr_y += 20

        remaining_h = TARGET_H - curr_y
        if num_regions > 0:
            slot_height = remaining_h // num_regions
            for i, counts in enumerate(counts_per_region):
                slot_y = curr_y + (i * slot_height)
                center_y = slot_y + (slot_height // 2)
                
                cv2.putText(dash, f"Region {i+1}", (20, slot_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                pie_radius = min(45, (slot_height // 2) - 30)
                pie_center = (80, center_y + 10)
                self.draw_pie_chart(dash, counts, pie_center, pie_radius)
                
                list_x = 150
                list_y_start = center_y - pie_radius + 10
                sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                
                if sum(counts.values()) == 0:
                     cv2.putText(dash, "No Traffic", (list_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1, cv2.LINE_AA)
                else:
                    line_h = 20
                    for idx, (cls_name, cnt) in enumerate(sorted_items):
                        if cnt == 0: continue
                        if idx > 4: break 
                        color = CLASS_COLORS.get(cls_name, (255, 255, 255))
                        cv2.circle(dash, (list_x, list_y_start + (idx*line_h) - 5), 4, color, -1, cv2.LINE_AA)
                        cv2.putText(dash, f"{cnt} {cls_name}", (list_x + 15, list_y_start + (idx*line_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        return dash

    def process(self):
        cap = cv2.VideoCapture(self.args.source)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.args.source}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Interactive drawing mode
        if self.args.draw:
            success, first_frame = cap.read()
            if not success:
                print("Error: Could not read first frame for drawing regions.")
                cap.release()
                return
            
            # Resize first frame to target display width/height
            first_frame_resized = cv2.resize(first_frame, (VIDEO_W, TARGET_H))
            self.draw_regions_interactively(first_frame_resized)
            self.init_buffers()
            
            # Save configuration if file path is specified
            if self.args.save_config:
                self.save_regions_to_config(self.args.save_config)
            
            # Reset capture position to start from beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        output_path = self.args.output
        if output_path.endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_W, TARGET_H))
        
        print(f"Starting processing: {total_frames} frames.")
        
        with tqdm(total=total_frames, unit="frames", desc="Processing") as pbar:
            while True:
                success, frame = cap.read()
                if not success: break
                
                frame_disp = cv2.resize(frame, (VIDEO_W, TARGET_H))
                results = self.model.predict(frame_disp, conf=self.args.conf, verbose=False, classes=TARGET_CLASSES)
                
                # --- STEP 1: GATHER RAW DATA (No Smoothing yet) ---
                frame_counts = [defaultdict(int) for _ in self.regions]
                frame_boxes = [[] for _ in self.regions]

                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        raw_name = self.model.names[cls_id]
                        label = CLASS_NAME_MAP.get(raw_name, raw_name)
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        for i, poly in enumerate(self.regions):
                            if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                                frame_counts[i][label] += 1
                                frame_boxes[i].append((x1, y1, x2, y2))
                                
                                color = CLASS_COLORS.get(label, (255, 255, 255))
                                cv2.rectangle(frame_disp, (x1, y1), (x2, y2), color, 2)
                                break

                # --- STEP 2: APPLY SMOOTHING ---
                smoothed_counts = [defaultdict(int) for _ in self.regions]

                # 2a. Smooth Dashboard Numbers
                for i in range(len(self.regions)):
                    for cls_name in CLASS_COLORS.keys():
                        val = frame_counts[i].get(cls_name, 0)
                        self.region_class_buffers[i][cls_name].append(val)
                        smoothed_counts[i][cls_name] = int(np.mean(self.region_class_buffers[i][cls_name]))

                # --- STEP 3: LOGIC & DRAWING ---
                overlay = frame_disp.copy()
                for i, poly in enumerate(self.regions):
                    # 3a. Update Congestion Buffers
                    raw_total = sum(frame_counts[i].values()) # Raw total for check
                    raw_ratio = self.calculate_occupancy(frame_disp.shape, poly, frame_boxes[i])
                    
                    self.total_count_buffers[i].append(raw_total)
                    self.occupancy_buffers[i].append(raw_ratio)
                    
                    avg_total = np.mean(self.total_count_buffers[i])
                    avg_ratio = np.mean(self.occupancy_buffers[i])
                    
                    # 3b. Determine Status (Green/Orange/Red)
                    fill_color = (0, 255, 0) # Green
                    alpha = 0.2
                    
                    if avg_total >= self.args.min_count:
                        if avg_ratio >= self.args.congestion_ratio:
                            fill_color = (0, 0, 255) # RED
                            alpha = 0.4
                        else:
                            fill_color = (0, 165, 255) # ORANGE
                    
                    # 3c. Draw Region
                    cv2.fillPoly(overlay, [poly], fill_color)
                    cv2.addWeighted(overlay, alpha, frame_disp, 1 - alpha, 0, frame_disp)
                    cv2.polylines(frame_disp, [poly], True, fill_color, 2, cv2.LINE_AA)
                    
                    # 3d. Draw Region Label
                    M = cv2.moments(poly)
                    if M["m00"] != 0:
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        label_text = f"Region {i+1}"
                        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(frame_disp, (cx - w//2 - 10, cy - h - 10), (cx + w//2 + 10, cy + 10), (0, 0, 0), -1)
                        cv2.rectangle(frame_disp, (cx - w//2 - 10, cy - h - 10), (cx + w//2 + 10, cy + 10), (255, 255, 255), 1)
                        cv2.putText(frame_disp, label_text, (cx - w//2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                # --- STEP 4: FINAL COMPOSITION ---
                self.draw_status_legend(frame_disp)
                
                # Pass SMOOTHED counts to dashboard to fix flickering
                dashboard = self.create_dashboard_frame(smoothed_counts)
                
                final_output = np.hstack((frame_disp, dashboard))
                out.write(final_output)
                pbar.update(1)

        cap.release()
        out.release()
        print(f"Processing Complete. Video saved to {self.args.output}")

if __name__ == "__main__":
    args = parse_arguments()
    renderer = ServerRenderer(args)
    renderer.process()
