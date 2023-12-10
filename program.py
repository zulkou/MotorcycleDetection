import cv2
import argparse
import imutils
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

CONFIDENCE_THRESHOLD = 0.50
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Fungsi untuk parsing argumen
def parse_args():
    parser = argparse.ArgumentParser(description='Object detection using YOLO and OpenCV')
    parser.add_argument('-v', '--video', help='Path to video file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    video_path = args.video

    # Load model dan tracker DeepSort
    model = YOLO('model/best.onnx', task='detect')
    tracker = DeepSort(max_age=5)

    if video_path is None:
        vs = cv2.VideoCapture(0)
    else:
        vs = cv2.VideoCapture(video_path)

    # Variabel untuk menyimpan ID motor
    motorcycle_ids = set()

    while True:
        (grabbed, frame) = vs.read()

        # Jika frame tidak ada maka berhenti
        if not grabbed:
            break

        # Resize frame
        frame = imutils.resize(frame, width=1080)

        # Deteksi objek menggunakan YOLO
        detections = model(frame)[0]

        # Variabel untuk menyimpan hasil deteksi
        results = []
                
        # Looping untuk setiap deteksi
        for data in detections.boxes.data.tolist():
            confidence = data[4]

            # Jika confidence < 0.5 maka skip
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # Ambil label dari deteksi
            label = detections.names[int(data[5])]
            # Jika label adalah motor
            if label == 'motor':
                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                class_id = label
                
                # Simpan hasil deteksi
                results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        # Update tracker
        tracks = tracker.update_tracks(results, frame=frame)
        
        # Looping untuk setiap track
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])

            # Gambar bounding box dan ID
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmax, ymin), GREEN, -1)
            cv2.putText(frame, "ID: {}".format(str(track_id)), (xmin + 5, ymin - 8),
                        cv2.FONT_HERSHEY_PLAIN, 1, WHITE)
            motorcycle_ids.add(track_id)

        # Gambar jumlah motor
        cv2.rectangle(frame, (10, 10), (150, 35), BLACK, -1)
        cv2.putText(frame, "Motor: {}".format(len(motorcycle_ids)), (15, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1, WHITE)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
