import argparse
from ultralytics import YOLO
from pathlib import Path

def main(opt):
    model = YOLO("yolov8n.pt")
    
    model.train(
        data=Path(opt.data).resolve(),
        epochs=opt.epochs,
        imgsz=opt.imgsz,
        batch=opt.batch,
        lr0=opt.lr0,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
        project=opt.project,
        name=opt.name,
        verbose=True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/data.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='yolo')
    opt = parser.parse_args()

    main(opt)
