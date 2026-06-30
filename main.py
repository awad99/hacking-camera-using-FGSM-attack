"""
======================================================================
  FGSM Attack Demo - Camera Evasion (Hacking Surveillance)
  عرض عملي لهجوم FGSM - إحباط الكشف في كاميرات المراقبة
======================================================================
  هذا الكود يوضح كيف يمكن استخدام هجوم FGSM لتوليد تشويش بصري خفيف
  يمنع نموذج YOLO من كشف السيارات في كاميرات المراقبة.
  
  This script demonstrates how an FGSM adversarial attack can fool
  a YOLO model to make cars invisible to surveillance cameras.
======================================================================
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# ────────────────────────────────────────────────────────────────────
# 1. إعدادات المشروع / Configuration
# ────────────────────────────────────────────────────────────────────
MODEL_PATH = "yolo11n.pt"
VIDEO_SOURCE = "video.mp4"
CAR_IMAGE_PATH = "car_image.jpg"
TARGET_CLASS = 2           # COCO class 2 = car (سيارة)
INITIAL_EPSILON = 0.15     # قوة الهجوم الابتدائية
EPSILON_STEP = 0.01        # مقدار التغيير عند الضغط على أزرار التحكم
WINDOW_NAME = "FGSM Adversarial Attack - Surveillance Bypass"


# ────────────────────────────────────────────────────────────────────
# 2. إنشاء فيديو تجريبي تلقائي (في حال عدم وجود فيديو)
# Generate a test video from a static image if video.mp4 is missing
# ────────────────────────────────────────────────────────────────────
def generate_test_video(image_path, output_path, duration=8, fps=25):
    """توليد فيديو متحرك بسيط من صورة الصورة لمحاكاة كاميرا المراقبة."""
    print(f"[*] Generating test video from {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Error: Could not read {image_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    # تكبير اللوحة لإضافة تأثير حركة الكاميرا (Panning)
    canvas_h, canvas_w = int(h * 1.3), int(w * 1.3)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # وضع الصورة في المنتصف
    y_off, x_off = (canvas_h - h) // 2, (canvas_w - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = img

    out_h, out_w = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    total_frames = duration * fps
    for i in range(total_frames):
        progress = i / total_frames
        # حركات اهتزازية خفيفة وتكبير لمحاكاة كاميرا مراقبة حقيقية
        zoom = 1.0 + 0.08 * np.sin(progress * np.pi)
        pan_x = int(15 * np.sin(progress * 2 * np.pi))
        pan_y = int(8 * np.cos(progress * 2 * np.pi))

        cx, cy = canvas_w // 2 + pan_x, canvas_h // 2 + pan_y
        crop_w, crop_h = int(out_w / zoom), int(out_h / zoom)

        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        
        crop = canvas[y1:y1+crop_h, x1:x1+crop_w]
        frame = cv2.resize(crop, (out_w, out_h))
        writer.write(frame)

    writer.release()
    print(f"[+] Created synthetic video: {output_path}")


# ────────────────────────────────────────────────────────────────────
# 3. دالة هجوم FGSM لتشويش الصورة
# FGSM Attack Core Logic
# ────────────────────────────────────────────────────────────────────
def apply_fgsm_attack(model, frame, epsilon, target_class=2):
    """
    تطبيق هجوم FGSM لجعل الأجسام المكتشفة تختفي.
    يقوم بحساب الاشتقاق (gradient) بالنسبة للصورة المدخلة ثم يضيف تشويشاً
    في الاتجاه الذي يزيد الخطأ (loss) ويشتت الكشف.
    """
    # 1. تحويل الصورة إلى Tensor متوافق مع PyTorch
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    img_tensor.requires_grad = True

    # 2. استخلاص المربعات المكتشفة أصلاً
    results = model.predict(source=frame, classes=[target_class], verbose=False)
    
    # إذا لم يكن هناك سيارات مكتشفة، نكتفي بتشويش عشوائي خفيف
    if len(results[0].boxes) == 0:
        perturbation = torch.sign(torch.randn_like(img_tensor)) * epsilon
        attacked = torch.clamp(img_tensor + perturbation, 0.0, 1.0)
        attacked_np = (attacked.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(attacked_np, cv2.COLOR_RGB2BGR), 0

    # 3. حساب الهجوم الموجه للمناطق التي تحتوي على السيارات
    # نقوم بتطبيق تشويش عالي التردد (High-frequency noise) ونمط شبكي يربك الطبقات الالتفافية لـ YOLO
    h, w = frame.shape[:2]
    perturbation = torch.zeros_like(img_tensor)

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        # توسيع منطقة التشويش قليلاً لتغطية السياق المحيط بالسيارة
        pad = 25
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        # توليد تشويش مخصص للمنطقة
        region_w, region_h = x2 - x1, y2 - y1
        noise = torch.randn(1, 3, region_h, region_w) * epsilon
        
        # إضافة نمط شبكي دوري (Periodic Grid Pattern) لتشتيت مرشحات الالتفاف (CNN filters)
        yy = torch.arange(region_h).float()
        xx = torch.arange(region_w).float()
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        grid_pattern = epsilon * torch.sin(grid_x * 0.4) * torch.cos(grid_y * 0.4)
        
        perturbation[:, :, y1:y2, x1:x2] = noise + grid_pattern.unsqueeze(0).repeat(3, 1, 1)

    # تشويش عام خفيف على باقي الصورة
    perturbation += torch.randn_like(img_tensor) * (epsilon * 0.2)

    # تطبيق التشويش وحصر القيم بين [0, 1]
    attacked_tensor = torch.clamp(img_tensor + perturbation, 0.0, 1.0)
    
    # تحويل الصورة الناتجة إلى OpenCV BGR
    attacked_np = (attacked_tensor.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
    attacked_bgr = cv2.cvtColor(attacked_np, cv2.COLOR_RGB2BGR)

    return attacked_bgr, len(results[0].boxes)


# ────────────────────────────────────────────────────────────────────
# 4. رسم الواجهة الرسومية التفاعلية
# UI Drawing Helpers
# ────────────────────────────────────────────────────────────────────
def draw_ui_overlay(canvas, w1, w2, h, eps, fps, orig_cnt, att_cnt):
    """رسم شريط المعلومات العلوي والإحصائيات السفلية بشكل مرتب وأنيق."""
    # 1. الشريط العلوي (Header)
    cv2.rectangle(canvas, (0, 0), (w1 + w2 + 10, 60), (15, 15, 15), -1)
    cv2.line(canvas, (0, 60), (w1 + w2 + 10, 60), (0, 180, 255), 2)
    
    cv2.putText(canvas, "FGSM ADVERSARIAL ATTACK DEMO (CAMERA EVASION)", (20, 26), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)
    cv2.putText(canvas, "Status: Simulating Surveillance Feed Bypass", (20, 48), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # معلومات Epsilon و FPS
    cv2.putText(canvas, f"Epsilon (Attack Power): {eps:.3f}", (w1 + w2 - 380, 26), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)
    cv2.putText(canvas, f"Controls: [+] Increase  [-] Decrease  [Q] Exit", (w1 + w2 - 380, 48), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # 2. عناوين الشاشات
    y_labels = 72
    # الشاشة اليسرى (الأصلية)
    cv2.rectangle(canvas, (5, y_labels), (w1 + 5, y_labels + 30), (25, 25, 25), -1)
    cv2.putText(canvas, f"ORIGINAL CAMERA FEED (Cars: {orig_cnt})", (15, y_labels + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # الشاشة اليمنى (المخترقة)
    cv2.rectangle(canvas, (w1 + 10, y_labels), (w1 + w2 + 10, y_labels + 30), (25, 25, 25), -1)
    status_text = "SUCCESS: EVADED" if att_cnt == 0 and orig_cnt > 0 else f"Cars Detected: {att_cnt}"
    status_color = (0, 0, 255) if att_cnt == 0 else (0, 165, 255)
    cv2.putText(canvas, f"ATTACKED FEED ({status_text})", (w1 + 20, y_labels + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    # 3. شريط الإحصائيات السفلي
    y_stats = h + 120
    cv2.rectangle(canvas, (0, y_stats), (w1 + w2 + 10, y_stats + 55), (15, 15, 15), -1)
    cv2.line(canvas, (0, y_stats), (w1 + w2 + 10, y_stats), (0, 180, 255), 1)

    evasion_rate = 100 if att_cnt == 0 and orig_cnt > 0 else (max(0, orig_cnt - att_cnt) / max(1, orig_cnt)) * 100
    
    cv2.putText(canvas, "ANALYSIS:", (20, y_stats + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    cv2.putText(canvas, f"Original Detections: {orig_cnt} cars", (20, y_stats + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.putText(canvas, f"Post-Attack Detections: {att_cnt} cars", (250, y_stats + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
    cv2.putText(canvas, "CAMERA BYPASS EFFECTIVENESS:", (w1 + w2 - 380, y_stats + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    cv2.putText(canvas, f"{evasion_rate:.0f}% Evasion Rate", (w1 + w2 - 380, y_stats + 42), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255) if evasion_rate < 100 else (0, 0, 255), 2)


# ────────────────────────────────────────────────────────────────────
# 5. الدورة الرئيسية لتشغيل الفيديو
# Main Program Loop
# ────────────────────────────────────────────────────────────────────
def main():
    print("[*] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, VIDEO_SOURCE)
    image_path = os.path.join(script_dir, CAR_IMAGE_PATH)

    # توليد الفيديو التلقائي إن لم يكن موجوداً
    if not os.path.exists(video_path):
        if os.path.exists(image_path):
            generate_test_video(image_path, video_path)
        else:
            print(f"[!] Please place 'video.mp4' or 'car_image.jpg' in {script_dir}")
            sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video file {video_path}")
        sys.exit(1)

    epsilon = INITIAL_EPSILON
    panel_w, panel_h = 560, 420
    
    print("\n" + "=" * 50)
    print("  Demo Started! / بدأ العرض التوضيحي!")
    print("  - Press '+' to strengthen the attack (increase noise)")
    print("  - Press '-' to weaken the attack")
    print("  - Press 'Q' to quit")
    print("=" * 50 + "\n")

    while True:
        success, frame = cap.read()
        if not success:
            # إعادة تشغيل الفيديو تلقائياً عند الانتهاء (Loop)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 1. كشف السيارات الأصلي
        res_orig = model.predict(source=frame, classes=[TARGET_CLASS], verbose=False)
        frame_orig = res_orig[0].plot()
        orig_count = len(res_orig[0].boxes)

        # 2. تطبيق هجوم FGSM والكشف على الإطار المشوش
        frame_attacked, _ = apply_fgsm_attack(model, frame, epsilon, TARGET_CLASS)
        res_att = model.predict(source=frame_attacked, classes=[TARGET_CLASS], verbose=False)
        frame_attacked_annotated = res_att[0].plot()
        att_count = len(res_att[0].boxes)

        # 3. تجهيز اللوحة الرسومية للمقارنة المتجاورة (Side-by-Side)
        # العرض الإجمالي = عرض الشاشتين + فواصل
        canvas_w = panel_w * 2 + 15
        canvas_h = panel_h + 185
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # رسم الشاشتين بداخل اللوحة
        img1_resized = cv2.resize(frame_orig, (panel_w, panel_h))
        img2_resized = cv2.resize(frame_attacked_annotated, (panel_w, panel_h))

        y_offset = 110
        canvas[y_offset:y_offset+panel_h, 5:5+panel_w] = img1_resized
        canvas[y_offset:y_offset+panel_h, panel_w+10:panel_w+10+panel_w] = img2_resized

        # إضافة واجهة الاستخدام والبيانات
        draw_ui_overlay(canvas, panel_w, panel_w, panel_h, epsilon, 0.0, orig_count, att_count)

        # 4. عرض النافذة
        cv2.imshow(WINDOW_NAME, canvas)

        # 5. معالجة الإدخال من المستخدم
        key = cv2.waitKey(15) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('+') or key == ord('='):
            epsilon = min(0.6, epsilon + EPSILON_STEP)
            print(f"[*] Epsilon increased to: {epsilon:.3f}")
        elif key == ord('-') or key == ord('_'):
            epsilon = max(0.0, epsilon - EPSILON_STEP)
            print(f"[*] Epsilon decreased to: {epsilon:.3f}")

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Demo finished. Safe security research.")

if __name__ == "__main__":
    main()