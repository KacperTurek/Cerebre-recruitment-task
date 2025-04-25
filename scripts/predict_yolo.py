import argparse
from ultralytics import YOLO

def main(opt):
    print(f"Loading model from {opt.weights}")
    model = YOLO(opt.weights)

    print(f"Running prediction on: {opt.source}")
    results = model.predict(
        source=opt.source,
        imgsz=opt.imgsz,
        conf=opt.conf,
        save=opt.save,
        save_txt=opt.save_txt,
        save_crop=opt.save_crop,
        project=opt.project,
        name=opt.name,
    )

    print("Prediction completed.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/yolo/weights/best.pt', help='Path to model weights')
    parser.add_argument('--source', type=str, default='../data/test/images', help='Image/dir/video path')
    parser.add_argument('--imgsz', type=int, default=416, help='Image size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save annotated results')
    parser.add_argument('--save_txt', action='store_true', help='Save detection labels to TXT files')
    parser.add_argument('--save_crop', action='store_true', help='Save cropped detected objects')
    parser.add_argument('--project', type=str, default='runs/predict')
    parser.add_argument('--name', type=str, default='exp')
    opt = parser.parse_args()

    main(opt)
