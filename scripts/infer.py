import cv2
from ultralytics import YOLO

MODEL_PATH = './weights/best.pt'
VIDEO_PATH = './test_video.mp4'
OUTPUT_PATH = './results/output.mp4'

def main():
    print(f"Loading model: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    print("Model loaded!")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video {VIDEO_PATH}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print(f"Processing {total_frames} frames...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames}")

        results = model(frame, verbose=False)

        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints

            if boxes is not None and keypoints is not None:
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if keypoints.xy is not None:
                        pts = keypoints.xy[i].cpu().numpy()
                        for pt in pts:
                            cx, cy = int(pt[0]), int(pt[1])
                            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Saved to: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()