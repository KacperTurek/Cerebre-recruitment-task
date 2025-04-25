import argparse
from ultralytics import YOLO
from pathlib import Path

def main(opt):
    print(f"Loading model from {opt.weights}")
    model = YOLO(opt.weights)

    print(f"Evaluating on {opt.split} set...")
    metrics = model.val(
        data=Path(opt.data).resolve(),
        split=opt.split,
        imgsz=opt.imgsz,
        batch=opt.batch,
        save_json=opt.save_json,
        save_hybrid=opt.save_hybrid,
    )

    print("Evaluation Results:")
    for key, value in metrics.results_dict.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/yolo/weights/best.pt', help='Path to model weights')
    parser.add_argument('--data', type=str, default='../data/data.yaml', help='Path to data.yaml')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Which split to evaluate on')
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--save_json', action='store_true', help='Save COCO-style JSON results')
    parser.add_argument('--save_hybrid', action='store_true', help='Save labels + predictions')
    opt = parser.parse_args()

    main(opt)
