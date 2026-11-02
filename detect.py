"""
Unified YOLO Real-Time Detection & Tracking CLI Launcher
Powered by modern YOLO11 & YOLOv8 with NVIDIA RTX GPU Acceleration
"""
import argparse
import os
import sys
import time
import cv2
from yolo_detector import YOLODetector


def clean_path(path_str):
    """
    Cleans up user-entered or pasted file paths:
    - Removes surrounding double/single quotes ("...", '...')
    - Strips whitespace, stray newlines, and trailing quote fragments
    - Normalizes separators
    """
    if not path_str:
        return ""
    s = str(path_str).strip()
    s = s.strip('"').strip("'").strip()
    lines = [line.strip().strip('"').strip("'") for line in s.splitlines() if line.strip()]
    if lines:
        s = " ".join(lines)
    s = s.strip('"').strip("'").strip()
    return os.path.normpath(s) if os.path.isabs(s) or ("/" in s or "\\" in s) else s


def browse_file():
    """
    Opens standard Windows File Dialog to select video or image file.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        file_path = filedialog.askopenfilename(
            title="Select Video or Image for YOLO Detection",
            filetypes=[
                (
                    "All Supported Media",
                    "*.mp4;*.avi;*.mov;*.mkv;*.wmv;*.jpg;*.jpeg;*.png;*.bmp;*.webp",
                ),
                ("Video Files", "*.mp4;*.avi;*.mov;*.mkv;*.wmv"),
                ("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
                ("All Files", "*.*"),
            ],
        )
        root.update()
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"[Warning] GUI File Dialog could not open: {e}")
        return None


def open_webcam(camera_index=0):
    """
    Opens webcam using DirectShow on Windows for maximum reliability,
    falling back to standard backends if needed.
    """
    backends = []
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            cap.release()
    return None


def run_detection(
    source="0",
    model="yolo11n",
    conf=0.35,
    iou=0.45,
    loop=True,
    save=False,
    output_dir="output",
):
    source = clean_path(source)
    os.makedirs(output_dir, exist_ok=True)
    detector = YOLODetector(model=model, conf_threshold=conf, iou_threshold=iou)

    # Check if source is webcam
    is_webcam = source.isdigit() or source == "webcam"
    is_image = False

    if not is_webcam:
        if not os.path.exists(source):
            print(f"\n[Error] File not found: '{source}'")
            print("Tip: If the path contains spaces, use option [5] to select it using the file picker!")
            return
        ext = os.path.splitext(source)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            is_image = True

    # 1. Image Mode
    if is_image:
        print(f"\n[Processing Image] {source}")
        image = cv2.imread(source)
        if image is None:
            print(f"[Error] Failed to read image: {source}")
            return

        detections, t_ms = detector.detect(image, track=False)
        result = detector.draw_detections(image, detections, inference_time_ms=t_ms)

        print(f"-> Detected {len(detections)} objects in {t_ms:.1f}ms on {detector.device_name}:")
        for d in detections:
            print(f"   * {d['label']} ({d['confidence']*100:.1f}%) at {d['box']}")

        if save:
            out_path = os.path.join(output_dir, f"detected_{os.path.basename(source)}")
            cv2.imwrite(out_path, result)
            print(f"-> Saved annotated image to: {out_path}")

        h, w = result.shape[:2]
        win_w = min(1280, w)
        win_h = int(h * (win_w / w))
        win_title = f"YOLO Detection - {os.path.basename(source)} ({model})"
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_title, win_w, win_h)
        cv2.imshow(win_title, result)
        print("\nPress any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 2. Video / Webcam Stream Mode
    if is_webcam:
        cam_idx = int(source) if source.isdigit() else 0
        print(f"\n[Opening Webcam] index={cam_idx} with DirectShow backend...")
        cap = open_webcam(cam_idx)
        if cap is None:
            print(f"[Error] Could not open webcam at index {cam_idx}.")
            print("Make sure your webcam is plugged in and permissions are enabled.")
            return
        window_title = f"YOLO Real-Time Detection - Webcam ({model})"
    else:
        print(f"\n[Opening Video] {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[Error] Could not open video file: {source}")
            return
        window_title = f"YOLO Real-Time Detection - {os.path.basename(source)} ({model})"

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Auto-fit display window size to user screen (max 1280x720)
    win_w = min(1280, src_w)
    win_h = int(src_h * (win_w / src_w)) if src_w > 0 else 720

    # Video writer setup if save is requested
    writer = None
    if save and not is_image:
        out_filename = f"detected_{os.path.splitext(os.path.basename(source))[0]}.mp4"
        out_path = os.path.join(output_dir, out_filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_in, (src_w, src_h))
        print(f"-> Recording output video to: {out_path}")

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, win_w, win_h)

    fps = 0.0
    frame_count = 0
    start_time = time.time()
    paused = False
    annotated = None

    print("\nKeyboard Controls in Video Window:")
    print("  [P]   - Pause / Resume")
    print("  [S]   - Save Screenshot to output/")
    print("  [L]   - Toggle Video Looping")
    print("  [ESC] or [Q] - Exit")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                if not is_webcam and loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    start_time = time.time()
                    frame_count = 0
                    continue
                else:
                    print("\nEnd of video stream reached.")
                    break

            frame_count += 1
            now = time.time()
            elapsed = now - start_time
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = now

            # Detection with persistent object tracking
            detections, t_ms = detector.detect(frame, track=True)
            annotated = detector.draw_detections(
                frame, detections, fps=fps, inference_time_ms=t_ms
            )

            if writer is not None:
                writer.write(annotated)

        if annotated is not None:
            # Resize display frame if resolution exceeds 1280x720 for ultra-smooth rendering
            if src_w > 1280 or src_h > 720:
                scale = min(1280.0 / src_w, 720.0 / src_h)
                disp_w = int(src_w * scale)
                disp_h = int(src_h * scale)
                display_frame = cv2.resize(annotated, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            else:
                display_frame = annotated.copy()

            if paused:
                h, w = display_frame.shape[:2]
                cv2.putText(
                    display_frame,
                    "PAUSED",
                    (w // 2 - 100, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
            cv2.imshow(window_title, display_frame)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        elif key in (ord("p"), ord("P")):
            paused = not paused
        elif key in (ord("s"), ord("S")):
            ss_name = f"screenshot_{int(time.time())}.jpg"
            ss_path = os.path.join(output_dir, ss_name)
            cv2.imwrite(ss_path, annotated)
            print(f"-> Screenshot saved: {ss_path}")
        elif key in (ord("l"), ord("L")):
            loop = not loop
            print(f"-> Video loop: {'ON' if loop else 'OFF'}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def interactive_menu():
    current_model = "yolo11n"

    while True:
        print("=" * 60)
        print("    YOLO Real-Time Detection & Tracking (RTX GPU)")
        print("=" * 60)
        print("  [1] UK Road Traffic Video (uk.mp4)")
        print("  [2] USA Street Video (usa-street.mp4)")
        print("  [3] Static Room Image (src_room.jpg)")
        print("  [4] Live Webcam Feed (DirectShow)")
        print("  [5] Browse Computer for Video / Image (File Picker Popup)")
        print(f"  [6] Switch Model [Current: {current_model.upper()}]")
        print("  [0] Exit")
        print("=" * 60)

        choice = input("Select an option [0-6]: ").strip()

        if choice == "1":
            run_detection(source="uk.mp4", model=current_model, loop=True)
        elif choice == "2":
            run_detection(source="usa-street.mp4", model=current_model, loop=True)
        elif choice == "3":
            run_detection(source="src_room.jpg", model=current_model, save=True)
        elif choice == "4":
            run_detection(source="0", model=current_model)
        elif choice == "5":
            print("\nOpening File Explorer dialog...")
            selected = browse_file()
            if selected:
                print(f"Selected: {selected}")
                run_detection(source=selected, model=current_model)
            else:
                print("No file selected.")
        elif choice == "6":
            print("\nAvailable Models:")
            print("  [1] yolo11n   (YOLO11 Nano - Latest, Best Accuracy & Speed, 5.4MB) [Recommended]")
            print("  [2] yolov8n   (YOLOv8 Nano - 6.2MB, Very Fast)")
            print("  [3] tiny      (Legacy YOLOv3-tiny, 34MB, OpenCV CPU)")
            print("  [4] full      (Legacy YOLOv3 Full, 237MB, OpenCV CPU)")
            mod_choice = input("Select model [1-4]: ").strip()
            if mod_choice == "1":
                current_model = "yolo11n"
            elif mod_choice == "2":
                current_model = "yolov8n"
            elif mod_choice == "3":
                current_model = "tiny"
            elif mod_choice == "4":
                current_model = "full"
            print(f"-> Active model set to: {current_model.upper()}")
        elif choice in ("0", "exit", "quit"):
            print("Exiting.")
            break
        else:
            print("[Invalid Option] Please enter a number between 0 and 6.")


def main():
    parser = argparse.ArgumentParser(description="YOLO Real-Time Object Detection & Tracking")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to video/image file or camera index (e.g., '0' for webcam)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n",
        help="YOLO model: 'yolo11n' (default), 'yolov8n', 'tiny', 'full'",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IOU threshold for NMS",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Do not loop video playback",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated video/image to output directory",
    )
    args = parser.parse_args()

    if args.source is None:
        interactive_menu()
    else:
        run_detection(
            source=args.source,
            model=args.model,
            conf=args.conf,
            iou=args.iou,
            loop=not args.no_loop,
            save=args.save,
        )


if __name__ == "__main__":
    main()
