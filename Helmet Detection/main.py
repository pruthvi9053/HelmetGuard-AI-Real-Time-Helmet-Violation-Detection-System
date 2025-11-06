import os
import cv2
import threading
from datetime import datetime
from ultralytics import YOLO

# -------- CONFIG --------
MODEL_PATH = r"Helmet Detection\best.pt"
VIDEO_PATH = r"Helmet Detection\Traffic1.mp4"
ALERT_SOUND = r"Helmet Detection\alert.wav"
VIOLATIONS_DIR = r"Helmet Detection\violations"
COOLDOWN_FRAMES = 30   # won't save again for same scene for N frames
# ------------------------

os.makedirs(VIOLATIONS_DIR, exist_ok=True)

# quick path checks
print("CHECK PATHS:")
print(" MODEL exists:", os.path.exists(MODEL_PATH), MODEL_PATH)
print(" VIDEO exists:", os.path.exists(VIDEO_PATH), VIDEO_PATH)
print(" SOUND exists:", os.path.exists(ALERT_SOUND), ALERT_SOUND)
print(" VIOLATIONS DIR:", VIOLATIONS_DIR)
print()

# load model with try/except to print errors
try:
    print("Loading model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    print("Model loaded. Class names:", model.names)
except Exception as e:
    print("ERROR loading model:", e)
    raise

# helper: safe play (non-blocking)
def play_alert(path):
    try:
        if not os.path.exists(path):
            print("Alert sound file not found:", path)
            return
        # try simpleaudio (non-blocking)
        try:
            import simpleaudio as sa
            wave = sa.WaveObject.from_wave_file(path)
            wave.play()
            return
        except Exception:
            pass
        # fallback playsound in thread
        try:
            from playsound import playsound
            threading.Thread(target=playsound, args=(path,), daemon=True).start()
            return
        except Exception:
            pass
        # fallback winsound on windows
        try:
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return
        except Exception:
            pass
        print("No working sound backend available.")
    except Exception as e:
        print("Exception in play_alert:", e)

# open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"ERROR: cannot open video: {VIDEO_PATH}")

print("Starting processing... press 'q' to quit.")
frame_no = 0
last_saved_frame = -COOLDOWN_FRAMES

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Video finished or cannot read more frames.")
            break
        frame_no += 1

        # run detection -- protect in try/except
        try:
            results = model(frame)   # returns list-like
        except Exception as e:
            print(f"[frame {frame_no}] ERROR running model(frame):", e)
            break

        violation_this_frame = False

        # iterate detections
        for res in results:
            # if res has no boxes skip
            boxes_attr = getattr(res, "boxes", None)
            if boxes_attr is None:
                continue

            # if .boxes.data exists, it's an array; else iterate boxes
            # we'll try both safe ways
            try:
                # preferred: iterate each Box object (ultralytics)
                for box in boxes_attr:
                    try:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                    except Exception:
                        # fallback to boxes.data
                        raise
                    label = str(model.names.get(cls_id, str(cls_id))).strip()
                    label_low = label.lower()
                    # debug: print detection occasionally
                    if frame_no % 100 == 0:
                        print(f"[frame {frame_no}] Detected: '{label}' conf:{conf:.2f} box:{x1,y1,x2,y2}")
                    # draw and decision
                    if "without" in label_low and "helmet" in label_low:
                        # red & mark violation
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
                        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                        violation_this_frame = True
                    elif "with" in label_low and "helmet" in label_low:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
                        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    else:
                        # unknown label: draw gray
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (200,200,200), 2)
                        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

            except Exception:
                # fallback: use boxes.data numpy array rows
                try:
                    data = boxes_attr.data.cpu().numpy()  # Nx6: x1,y1,x2,y2,conf,cls
                    for row in data:
                        x1, y1, x2, y2, conf, cls = row
                        cls = int(cls)
                        x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
                        label = str(model.names.get(cls, str(cls))).strip()
                        label_low = label.lower()
                        if frame_no % 100 == 0:
                            print(f"[frame {frame_no}] Detected (ndarray): '{label}' conf:{conf:.2f} box:{x1,y1,x2,y2}")
                        if "without" in label_low and "helmet" in label_low:
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
                            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                            violation_this_frame = True
                        elif "with" in label_low and "helmet" in label_low:
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
                            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                        else:
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (200,200,200), 2)
                            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
                except Exception as e:
                    print(f"[frame {frame_no}] Failed to read boxes: {e}")

        # if violation and cooldown passed => save & sound
        if violation_this_frame and (frame_no - last_saved_frame) >= COOLDOWN_FRAMES:
            last_saved_frame = frame_no
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"no_helmet_{ts}_f{frame_no}.jpg"
            outpath = os.path.join(VIOLATIONS_DIR, fname)
            try:
                cv2.imwrite(outpath, frame)
                print(f"[SAVED] {outpath} (frame {frame_no})")
            except Exception as e:
                print("ERROR saving image:", e)
            # play alert sound in background
            try:
                threading.Thread(target=play_alert, args=(ALERT_SOUND,), daemon=True).start()
            except Exception as e:
                print("ERROR starting sound thread:", e)

        # show
        cv2.imshow("Helmet Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit pressed.")
            break

    except KeyboardInterrupt:
        print("KeyboardInterrupt - exiting")
        break
    except Exception as e:
        print("Unexpected error in main loop:", e)
        break

cap.release()
cv2.destroyAllWindows()
print("Finished.")
