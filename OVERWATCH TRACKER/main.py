"""
OVERWATCH TRACKER - Real-time Person Tracking System
With Threat Level Analysis & Pose Estimation
coded by konpep
"""

import sys
import subprocess

# Auto-install missing packages
def check_and_install():
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'torch': 'torch',
        'ultralytics': 'ultralytics'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"[*] Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing + ['-q'])
        print("[OK] Packages installed!\n")

check_and_install()

import cv2
import numpy as np
import time
import threading
import os
from collections import deque

# Check available backends
HAS_CUDA = False
HAS_DIRECTML = False

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except:
    pass

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    HAS_DIRECTML = 'DmlExecutionProvider' in providers
except:
    pass

from ultralytics import YOLO

# Load config
try:
    from config import CAMERA_IP, CAMERA_PORT, TRAIL_LENGTH
except ImportError:
    CAMERA_IP = "192.168.1.111"
    CAMERA_PORT = 8080
    TRAIL_LENGTH = 20

# Colors
ESP_BOX = (0, 255, 0)
ESP_HEAD = (0, 0, 255)
ESP_BONE = (255, 0, 255)
ESP_JOINT = (0, 255, 255)
UNARMED_COLOR = (0, 255, 0)
HOLDING_COLOR = (0, 165, 255)

# Threat level colors
THREAT_RELAXED = (0, 255, 0)      # Green
THREAT_ALERT = (0, 255, 255)      # Yellow
THREAT_DANGER = (0, 165, 255)     # Orange
THREAT_CRITICAL = (0, 0, 255)     # Red

# Objects that can be detected
HOLDABLE = {
    67: "PHONE", 77: "PHONE", 73: "LAPTOP", 74: "MOUSE", 
    76: "KEYBOARD", 75: "REMOTE", 72: "TV",
    39: "BOTTLE", 41: "CUP", 43: "KNIFE", 44: "SPOON",
    42: "FORK", 40: "GLASS", 46: "BANANA", 47: "APPLE",
    48: "SANDWICH", 49: "ORANGE", 53: "PIZZA", 54: "DONUT",
    84: "BOOK", 85: "CLOCK", 87: "SCISSORS", 79: "HAIRDRYER",
}

# Dangerous objects (higher threat level)
DANGEROUS_OBJECTS = {"KNIFE", "SCISSORS"}

# Skeleton connections for pose visualization
SKELETON = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

# Quality presets
PRESETS = {
    '1': {
        'name': 'LOW',
        'pose_model': 'yolov8n-pose.pt',
        'obj_model': 'yolov8n.pt',
        'pose_size': 256,
        'obj_size': 256,
        'skip_frames': 2,
        'obj_skip': 5,
    },
    '2': {
        'name': 'MEDIUM',
        'pose_model': 'yolov8n-pose.pt',
        'obj_model': 'yolov8n.pt',
        'pose_size': 320,
        'obj_size': 384,
        'skip_frames': 1,
        'obj_skip': 3,
    },
    '3': {
        'name': 'HIGH',
        'pose_model': 'yolov8s-pose.pt',
        'obj_model': 'yolov8s.pt',
        'pose_size': 416,
        'obj_size': 480,
        'skip_frames': 1,
        'obj_skip': 2,
    },
    '4': {
        'name': 'ULTRA',
        'pose_model': 'yolov8m-pose.pt',
        'obj_model': 'yolov8m.pt',
        'pose_size': 640,
        'obj_size': 640,
        'skip_frames': 1,
        'obj_skip': 1,
    },
}

# Global state
track_data = {}
head_trails = {}
objects_data = []
running = True
settings = None
device = 'cpu'


def calculate_threat_level(person_data, movement_speed):
    """Calculate threat level based on multiple factors"""
    threat = 0
    
    # Factor 1: Holding something
    holding = person_data.get('holding')
    if holding:
        threat += 30
        if holding in DANGEROUS_OBJECTS:
            threat += 40
    
    # Factor 2: Movement speed (fast = more alert)
    if movement_speed > 50:
        threat += 30
    elif movement_speed > 25:
        threat += 15
    
    # Factor 3: Distance (closer = more threat) - based on box size
    box = person_data.get('box', (0, 0, 100, 100))
    box_height = box[3] - box[1]
    if box_height > 300:  # Very close
        threat += 20
    elif box_height > 200:  # Close
        threat += 10
    
    return min(threat, 100)


def get_threat_info(threat_level):
    """Get threat status text and color"""
    if threat_level >= 70:
        return "CRITICAL", THREAT_CRITICAL
    elif threat_level >= 50:
        return "DANGER", THREAT_DANGER
    elif threat_level >= 25:
        return "ALERT", THREAT_ALERT
    else:
        return "RELAXED", THREAT_RELAXED


def draw_threat_bar(frame, x1, y1, x2, threat_level):
    """Draw threat level bar above person"""
    bar_width = x2 - x1
    bar_height = 8
    bar_y = y1 - 45
    
    # Background
    cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_height), (50, 50, 50), -1)
    
    # Fill based on threat
    fill_width = int(bar_width * threat_level / 100)
    _, color = get_threat_info(threat_level)
    cv2.rectangle(frame, (x1, bar_y), (x1 + fill_width, bar_y + bar_height), color, -1)
    
    # Border
    cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_height), (100, 100, 100), 1)


def draw_head_trail(frame, trail):
    """Draw trail behind head showing movement path"""
    if len(trail) < 2:
        return
    
    points = list(trail)
    for i in range(1, len(points)):
        # Fade effect - older points are more transparent
        alpha = i / len(points)
        thickness = max(1, int(3 * alpha))
        
        # Color gradient from dark red to bright red
        color = (0, 0, int(100 + 155 * alpha))
        
        cv2.line(frame, points[i-1], points[i], color, thickness)


def show_menu():
    """Display settings menu"""
    global device
    
    print("\n" + "="*55)
    print("          OVERWATCH TRACKER v1.0")
    print("             coded by konpep")
    print("="*55)
    
    # Hardware detection
    print("\n[HARDWARE DETECTION]")
    if HAS_CUDA:
        import torch
        print(f"  ✓ NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ✗ NVIDIA GPU: Not found")
    
    if HAS_DIRECTML:
        print("  ✓ AMD/Intel GPU: DirectML available")
    else:
        print("  ✗ AMD/Intel GPU: DirectML not available")
    
    try:
        import torch
        print(f"  ✓ CPU Threads: {torch.get_num_threads()}")
    except:
        print("  ✓ CPU: Available")
    
    # Device selection
    print("\n" + "-"*55)
    print("[SELECT DEVICE]")
    print("  [1] CPU only (works everywhere)")
    if HAS_CUDA:
        print("  [2] NVIDIA GPU (CUDA) - Fastest")
    else:
        print("  [2] NVIDIA GPU - Not supported")
    if HAS_DIRECTML:
        print("  [3] AMD/Intel GPU (DirectML)")
    else:
        print("  [3] AMD/Intel GPU - Not supported")
    print()
    
    while True:
        choice = input("Device (1-3): ").strip() or '1'
        if choice == '1':
            device = 'cpu'
            break
        elif choice == '2':
            if HAS_CUDA:
                device = 'cuda'
                break
            else:
                print("  NVIDIA GPU not available!")
                continue
        elif choice == '3':
            if HAS_DIRECTML:
                device = 'directml'
                break
            else:
                print("  DirectML not available!")
                continue
        print("Invalid choice!")
    
    # Quality selection
    print("\n" + "-"*55)
    print("[SELECT QUALITY]")
    print("  [1] LOW    - Best FPS, lower accuracy")
    print("  [2] MEDIUM - Balanced (recommended)")
    print("  [3] HIGH   - Better accuracy, needs good PC")
    print("  [4] ULTRA  - Maximum accuracy, needs GPU")
    print()
    
    while True:
        choice = input("Quality (1-4) [2]: ").strip() or '2'
        if choice in PRESETS:
            return PRESETS[choice]
        print("Invalid choice!")


class ObjectDetector(threading.Thread):
    """Separate thread for object detection"""
    def __init__(self, model, img_size, use_directml=False):
        super().__init__(daemon=True)
        self.model = model
        self.img_size = img_size
        self.use_directml = use_directml
        self.frame = None
        self.lock = threading.Lock()
        self.new_frame = threading.Event()
        
    def set_frame(self, frame):
        with self.lock:
            self.frame = frame.copy()
        self.new_frame.set()
    
    def run(self):
        global objects_data
        while running:
            self.new_frame.wait(timeout=0.1)
            self.new_frame.clear()
            
            with self.lock:
                if self.frame is None:
                    continue
                frame = self.frame
            
            try:
                dev = 'cpu' if device == 'directml' else device
                results = self.model(frame, verbose=False, imgsz=self.img_size, conf=0.25, device=dev)
                
                new_objects = []
                if results[0].boxes is not None:
                    for box, cls, conf in zip(
                        results[0].boxes.xyxy.cpu().numpy(),
                        results[0].boxes.cls.cpu().numpy().astype(int),
                        results[0].boxes.conf.cpu().numpy()
                    ):
                        if cls in HOLDABLE:
                            new_objects.append({
                                'box': tuple(map(int, box)),
                                'name': HOLDABLE[cls],
                                'conf': conf
                            })
                objects_data = new_objects
            except:
                pass


class FrameGrabber(threading.Thread):
    """Separate thread for frame grabbing to reduce latency"""
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.frame = None
        self.lock = threading.Lock()
        
    def run(self):
        while running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
    
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


def draw_all(frame, persons, fps, info):
    """Draw all ESP elements on frame"""
    global head_trails
    
    for pid, p in persons.items():
        x1, y1, x2, y2 = p['box']
        hx, hy = p['head']
        kpts = p.get('kpts')
        holding = p.get('holding')
        threat = p.get('threat', 0)
        
        # Update head trail
        if pid not in head_trails:
            head_trails[pid] = deque(maxlen=TRAIL_LENGTH)
        head_trails[pid].append((hx, hy))
        
        # Draw head trail
        draw_head_trail(frame, head_trails[pid])
        
        # Draw skeleton
        if kpts is not None:
            pts = kpts.reshape(-1, 3)
            for i, j in SKELETON:
                if i < len(pts) and j < len(pts) and pts[i][2] > 0.3 and pts[j][2] > 0.3:
                    p1 = (int(pts[i][0]), int(pts[i][1]))
                    p2 = (int(pts[j][0]), int(pts[j][1]))
                    if p1[0] > 0 and p2[0] > 0:
                        cv2.line(frame, p1, p2, ESP_BONE, 2)
            for pt in pts:
                if pt[2] > 0.3 and pt[0] > 0:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, ESP_JOINT, -1)
        
        # Get threat color for box
        threat_status, threat_color = get_threat_info(threat)
        
        # Draw corner-style ESP box with threat color
        c = min(x2-x1, y2-y1) // 4
        for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (px, py), (px + c*dx, py), threat_color, 2)
            cv2.line(frame, (px, py), (px, py + c*dy), threat_color, 2)
        
        # Draw threat bar
        draw_threat_bar(frame, x1, y1, x2, threat)
        
        # Draw cross on head
        cv2.line(frame, (hx - 15, hy), (hx + 15, hy), ESP_HEAD, 2)
        cv2.line(frame, (hx, hy - 15), (hx, hy + 15), ESP_HEAD, 2)
        
        # Status label with threat
        cx = (x1 + x2) // 2
        if holding:
            text = f"[{pid}] {holding}"
        else:
            text = f"[{pid}] {threat_status}"
        
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx, ty = cx - tw//2, y1 - 55
        cv2.rectangle(frame, (tx-3, ty-th-3), (tx+tw+3, ty+3), (0,0,0), -1)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, threat_color, 1)
    
    # Clean up old trails for persons no longer tracked
    current_ids = set(persons.keys())
    for pid in list(head_trails.keys()):
        if pid not in current_ids:
            del head_trails[pid]
    
    # Info overlay
    cv2.rectangle(frame, (5, 5), (200, 80), (0,0,0), -1)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    cv2.putText(frame, f"Targets: {len(persons)}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
    
    return frame


def match_objects_to_persons():
    """Match detected objects to persons based on hand proximity"""
    global track_data, objects_data
    
    for pid, pdata in track_data.items():
        px1, py1, px2, py2 = pdata['box']
        pw, ph = px2 - px1, py2 - py1
        
        # Get hand positions from keypoints
        hands = []
        kpts = pdata.get('kpts')
        if kpts is not None:
            pts = kpts.reshape(-1, 3)
            for idx in [9, 10]:  # Wrist keypoints
                if idx < len(pts) and pts[idx][2] > 0.2:
                    hands.append((int(pts[idx][0]), int(pts[idx][1])))
        
        # Fallback to body center if no hands detected
        if not hands:
            hands = [(px1 + pw//2, py1 + int(ph*0.6))]
        
        found = None
        best_dist = 120
        
        for obj in objects_data:
            ox1, oy1, ox2, oy2 = obj['box']
            ocx, ocy = (ox1+ox2)//2, (oy1+oy2)//2
            
            # Check if object is inside person box
            if px1 - 20 < ocx < px2 + 20 and py1 < ocy < py2 + 20:
                for hx, hy in hands:
                    dist = ((ocx - hx)**2 + (ocy - hy)**2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        found = obj['name']
        
        # Update holding status with persistence
        if found:
            track_data[pid]['holding'] = found
            track_data[pid]['hold_frames'] = 15
        elif pdata.get('hold_frames', 0) > 0:
            track_data[pid]['hold_frames'] -= 1
        else:
            track_data[pid]['holding'] = None
        
        # Calculate movement speed from trail
        speed = 0
        if pid in head_trails and len(head_trails[pid]) >= 2:
            trail = list(head_trails[pid])
            dx = trail[-1][0] - trail[0][0]
            dy = trail[-1][1] - trail[0][1]
            speed = (dx**2 + dy**2) ** 0.5 / len(trail) * 10
        
        # Update threat level
        track_data[pid]['threat'] = calculate_threat_level(pdata, speed)


def main():
    global track_data, running, settings, device
    
    # Show settings menu
    settings = show_menu()
    
    use_directml = (device == 'directml')
    actual_device = 'cpu' if use_directml else device
    
    print(f"\n[LOADING]")
    print(f"  Quality: {settings['name']}")
    print(f"  Device: {device.upper()}")
    print(f"  Pose Model: {settings['pose_model']}")
    print(f"  Object Model: {settings['obj_model']}")
    
    print("\nLoading models (this may take a moment)...")
    
    # Load models
    if use_directml:
        pose_model = YOLO(settings['pose_model'])
        obj_model = YOLO(settings['obj_model'])
        
        # Export to ONNX for DirectML
        pose_onnx = settings['pose_model'].replace('.pt', '.onnx')
        obj_onnx = settings['obj_model'].replace('.pt', '.onnx')
        
        if not os.path.exists(pose_onnx):
            print("  Exporting pose model to ONNX...")
            pose_model.export(format='onnx', imgsz=settings['pose_size'])
        if not os.path.exists(obj_onnx):
            print("  Exporting object model to ONNX...")
            obj_model.export(format='onnx', imgsz=settings['obj_size'])
        
        pose_model = YOLO(pose_onnx)
        obj_model = YOLO(obj_onnx)
    else:
        pose_model = YOLO(settings['pose_model'])
        obj_model = YOLO(settings['obj_model'])
        
        if actual_device == 'cpu':
            pose_model.fuse()
            obj_model.fuse()
    
    # Connect to camera
    url = f"http://{CAMERA_IP}:{CAMERA_PORT}/video"
    print(f"\nConnecting to camera at {url}...")
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("\n[ERROR] Cannot connect to camera!")
        print("Make sure:")
        print("  1. IP Webcam app is running on your phone")
        print("  2. You tapped 'Start Server' in the app")
        print(f"  3. Your phone's IP is {CAMERA_IP}")
        print("  4. Your phone and PC are on the same WiFi network")
        return
    
    # Start background threads
    grabber = FrameGrabber(cap)
    grabber.start()
    
    detector = ObjectDetector(obj_model, settings['obj_size'], use_directml)
    detector.start()
    
    time.sleep(0.5)
    
    # Create window
    cv2.namedWindow("OVERWATCH", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("OVERWATCH", 800, 600)
    
    info_text = f"{settings['name']} | {device.upper()}"
    
    print("\n" + "="*55)
    print("  OVERWATCH TRACKER RUNNING")
    print("  Press Q to quit")
    print("="*55 + "\n")
    
    fps = 30
    prev_t = time.time()
    frame_n = 0
    
    # Main loop
    while running:
        frame = grabber.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        
        frame = cv2.resize(frame, (640, 480))
        
        # Calculate FPS
        t = time.time()
        fps = fps * 0.9 + 0.1 / (t - prev_t + 0.001)
        prev_t = t
        frame_n += 1
        
        # Send frame to object detector
        if frame_n % settings['obj_skip'] == 0:
            detector.set_frame(frame)
        
        # Run pose detection
        if frame_n % settings['skip_frames'] == 0:
            try:
                results = pose_model.track(frame, persist=True, verbose=False, 
                                          imgsz=settings['pose_size'], conf=0.4, device=actual_device)
                
                new_data = {}
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    kpts = results[0].keypoints.data.cpu().numpy() if results[0].keypoints else None
                    ids = results[0].boxes.id
                    ids = ids.cpu().numpy().astype(int) if ids is not None else list(range(len(boxes)))
                    
                    for i, (box, tid) in enumerate(zip(boxes, ids)):
                        x1, y1, x2, y2 = map(int, box)
                        k = kpts[i] if kpts is not None and i < len(kpts) else None
                        
                        # Get head position from nose keypoint or estimate
                        if k is not None and len(k) > 0 and k[0][2] > 0.3:
                            hx, hy = int(k[0][0]), int(k[0][1])
                        else:
                            hx, hy = (x1+x2)//2, y1 + (y2-y1)//10
                        
                        # Preserve previous state
                        prev = track_data.get(tid, {})
                        new_data[tid] = {
                            'box': (x1, y1, x2, y2),
                            'head': (hx, hy),
                            'kpts': k,
                            'holding': prev.get('holding'),
                            'hold_frames': prev.get('hold_frames', 0),
                            'threat': prev.get('threat', 0)
                        }
                
                track_data = new_data
            except:
                pass
        
        # Match objects to persons
        if frame_n % settings['obj_skip'] == 0:
            match_objects_to_persons()
        
        # Draw everything
        frame = draw_all(frame, track_data, fps, info_text)
        cv2.imshow("OVERWATCH", frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nOVERWATCH closed.")


if __name__ == "__main__":
    main()
