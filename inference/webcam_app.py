import cv2
import numpy as np
from collections import deque, Counter
import time
import ctypes
import os
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
WEIGHTS_DIR = ROOT_DIR / "weights"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "asl_crossdomain_matplotlib"))

from ultralytics import YOLO
from pyvi.ViUtils import add_accents
from PIL import ImageFont, ImageDraw, Image
import torch
import timm
from torchvision import transforms

# ===================== 1. Load models =====================
STAGE1_IMGSZ = 640
STAGE1_CONF = 0.25

hand_model = YOLO(str(WEIGHTS_DIR / "yolov8s_stage1_hand_detector.pt"))

with open(BASE_DIR / "label.txt", "r", encoding="utf-8") as f:
    asl_labels = [line.strip() for line in f if line.strip()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asl_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(asl_labels))
checkpoint = torch.load(WEIGHTS_DIR / "ViT.pth", map_location=device)
state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
asl_model.load_state_dict(state_dict, strict=False)
asl_model.to(device)
asl_model.eval()

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def predict_asl_vit(hand_crop):
    rgb_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_crop)
    input_tensor = vit_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = asl_model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, class_id = torch.max(probs, dim=0)
    return asl_labels[int(class_id)], float(conf)

# ===================== 2. Initialize webcam =====================
def open_webcam(max_index=4):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        for camera_index in range(max_index):
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                print(f"Opened webcam index {camera_index} with backend {backend}")
                return cap
            cap.release()
    raise RuntimeError("Cannot open webcam. Please check camera permission or change the camera index.")

cap = open_webcam()

# ===================== 3. Variables =====================
chat_text = ""         # chữ chưa dấu, hiển thị ở Top box
completed_text = ""    # chữ đã dấu, hiển thị ở Bottom box
current_word = ""      # từ đang nhập
gesture_window = deque(maxlen=30)
HOLD_TIME = 1.5
gesture_start_time = None
current_gesture = None
progress = 0

# Scroll riêng cho top/bottom
top_scroll = 0
bottom_scroll = 0
scroll_step = 30

# ===================== 4. Window =====================
window_name = "YOLO Hand + ASL Chat"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
window_w, window_h = 1280, 720
cv2.resizeWindow(window_name, window_w, window_h)
user32 = ctypes.windll.user32
screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
cv2.moveWindow(window_name, (screen_w - window_w)//2, (screen_h - window_h)//2)

# ===================== 5. Font =====================
font_path = "C:\\Windows\\Fonts\\arial.ttf"
font_size = 24
font = ImageFont.truetype(font_path, font_size)

# ===================== 6. Refresh button =====================
refresh_button = {"width": 120, "height": 40, "clicked": False}
def update_refresh_button_position(chat_h):
    refresh_button["x1"] = 10
    refresh_button["x2"] = 10 + refresh_button["width"]
    refresh_button["y1"] = chat_h - refresh_button["height"] - 10
    refresh_button["y2"] = refresh_button["y1"] + refresh_button["height"]
update_refresh_button_position(window_h)

# ===================== 7. Wrap text =====================
def wrap_text_pil(text, font, max_width):
    words = text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = font.getbbox(test_line)
        text_w = bbox[2] - bbox[0]
        if text_w <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

# ===================== 8. Mouse callback =====================
def mouse_callback(event, x, y, flags, param):
    global top_scroll, bottom_scroll
    frame_w = param['frame_w']
    x_rel = x - frame_w
    top_box_h = param['top_box_h']
    bottom_box_start = param['bottom_box_start']
    bottom_box_h = param['bottom_box_h']

    if event == cv2.EVENT_LBUTTONDOWN:
        # Refresh button
        if refresh_button["x1"] <= x_rel <= refresh_button["x2"] and refresh_button["y1"] <= y <= refresh_button["y2"]:
            refresh_button["clicked"] = True
    elif event == cv2.EVENT_MOUSEWHEEL:
        # Scroll riêng
        if y < top_box_h:  # trên Top box
            if flags > 0:
                top_scroll = max(0, top_scroll - scroll_step)
            else:
                top_scroll += scroll_step
        elif bottom_box_start <= y <= bottom_box_start + bottom_box_h:  # trong Bottom box
            if flags > 0:
                bottom_scroll = max(0, bottom_scroll - scroll_step)
            else:
                bottom_scroll += scroll_step

cv2.setMouseCallback(window_name, mouse_callback, param={'frame_w': 640, 'top_box_h':0, 'bottom_box_start':0, 'bottom_box_h':0})

# ===================== 9. Main loop =====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Tính toán vị trí Top/Bottom box để callback mouse
    top_box_h = h//2 - 40
    bottom_box_start = h//2
    bottom_box_h = h - bottom_box_start - 100
    cv2.setMouseCallback(window_name, mouse_callback, param={'frame_w': w,
                                                             'top_box_h': top_box_h,
                                                             'bottom_box_start': bottom_box_start,
                                                             'bottom_box_h': bottom_box_h})

    detected_gesture = None

    # ---------- Detect hand ----------
    hand_results = hand_model.predict(frame, imgsz=STAGE1_IMGSZ, conf=STAGE1_CONF, verbose=False)[0]
    if hand_results.boxes is not None and len(hand_results.boxes) > 0:
        for box in hand_results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            margin = 10
            x1, y1 = max(0, x1-margin), max(0, y1-margin)
            x2, y2 = min(w, x2+margin), min(h, y2+margin)
            hand_crop = frame[y1:y2, x1:x2]
            if hand_crop.size == 0:
                continue
            gesture, conf = predict_asl_vit(hand_crop)
            detected_gesture = gesture
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,200,0), 2)
            cv2.putText(frame, f"{gesture} ({conf*100:.1f}%)", (x1, y1-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)
            break

    # ---------- Update gesture ----------
    if detected_gesture not in [None, "nothing"]:
        gesture_window.append(detected_gesture)
        non_nothing = [g for g in gesture_window if g != "nothing"]
        if non_nothing:
            most_common_gesture, count = Counter(non_nothing).most_common(1)[0]
            if most_common_gesture != current_gesture:
                current_gesture = most_common_gesture
                gesture_start_time = time.time()
                progress = 0
            else:
                progress = min(int((time.time() - gesture_start_time)/HOLD_TIME*100), 100)
                if time.time() - gesture_start_time >= HOLD_TIME:
                    
                    # ======================================================
                    # ---------- Handle gestures (PHẦN ĐÃ SỬA) ----------
                    # ======================================================
                    
                    if most_common_gesture == "del":
                        if current_word:
                            # TH1: Đang gõ dở 1 từ
                            current_word = current_word[:-1]
                            chat_text = chat_text[:-1]
                        elif chat_text:
                            # TH2: Vừa gõ space xong, xóa dấu space
                            chat_text = chat_text[:-1]
                        # Không làm gì completed_text
                            
                    elif most_common_gesture == "space":
                        # 1. Thêm space vào top box và reset current_word
                        chat_text += " "
                        current_word = "" 
                        
                        # 2. (LOGIC MỚI) Lấy *toàn bộ* text ở top box để convert
                        text_to_convert = chat_text.strip() # Lấy text, bỏ dấu space cuối
                        
                        if text_to_convert:
                            # Chuyển đổi toàn bộ câu
                            converted_sentence = add_accents(text_to_convert.lower())
                            
                            # Viết hoa chữ cái đầu CÂU và gán cho bottom box
                            completed_text = converted_sentence.capitalize()
                            
                            # Thêm space vào cuối bottom box để đồng bộ
                            completed_text += " "
                        else:
                            # Nếu top box chỉ là space, bottom box rỗng
                            completed_text = ""

                    else: # (Các cử chỉ chữ cái: A, B, C...)
                        current_word += most_common_gesture
                        chat_text += most_common_gesture
                        
                    # ======================================================
                    # ---------- (HẾT PHẦN SỬA) ----------
                    # ======================================================
                    
                    gesture_window.clear()
                    gesture_start_time = None
                    current_gesture = None
                    progress = 0
    else:
        progress = 0
        current_gesture = None
        gesture_start_time = None

    # ---------- Draw chat frame ----------
    chat_w = 350
    chat_h = h
    chat_frame = np.ones((chat_h, chat_w, 3), dtype=np.uint8) * 245
    img_pil = Image.fromarray(chat_frame)
    draw = ImageDraw.Draw(img_pil)
    max_width = chat_w - 20

    # Top box (chưa dấu)
    draw.rectangle([5,5,chat_w-5, top_box_h], outline=(0,0,0), width=2)
    wrapped_top = wrap_text_pil(chat_text, font, max_width)
    for i, line in enumerate(wrapped_top[-100:]):
        y_pos = 10 + i*30 - top_scroll
        if 10 <= y_pos <= top_box_h-30:
            draw.text((10, y_pos), line, font=font, fill=(70,130,180))

    # Bottom box (đã dấu)
    draw.rectangle([5,bottom_box_start,chat_w-5,bottom_box_start+bottom_box_h], outline=(0,0,0), width=2)
    wrapped_bottom = wrap_text_pil(completed_text, font, max_width)
    for i, line in enumerate(wrapped_bottom[-100:]):
        y_pos = bottom_box_start + 10 + i*30 - bottom_scroll
        if bottom_box_start <= y_pos <= bottom_box_start+bottom_box_h-30:
            draw.text((10, y_pos), line, font=font, fill=(70,180,70))

    # Progress bar
    progress_y_start = chat_h - 80
    draw.rectangle([10, progress_y_start, chat_w-10, progress_y_start+25], fill=(200,200,200))
    draw.rectangle([10, progress_y_start, 10 + int((chat_w-20)*progress/100), progress_y_start+25], fill=(70,180,120))
    draw.text((chat_w//2-20, progress_y_start+5), f"{progress}%", font=font, fill=(50,50,50))

    # Refresh button
    update_refresh_button_position(chat_h)
    draw.rectangle([refresh_button["x1"], refresh_button["y1"], refresh_button["x2"], refresh_button["y2"]], fill=(200,50,50))
    draw.text((refresh_button["x1"]+10, refresh_button["y1"]+5), "Refresh", font=font, fill=(255,255,255))
    if refresh_button["clicked"]:
        chat_text = ""
        completed_text = ""
        current_word = ""
        top_scroll = 0
        bottom_scroll = 0
        refresh_button["clicked"] = False

    chat_frame = np.array(img_pil)
    if frame.shape[0] != chat_frame.shape[0]:
        chat_frame = cv2.resize(chat_frame, (chat_w, frame.shape[0]))
    combined_frame = np.hstack((frame, chat_frame))
    cv2.imshow(window_name, combined_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
