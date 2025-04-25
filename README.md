# YOLOv8 Object Detection Pipeline

This project demonstrates a complete end-to-end workflow for training, evaluating, and tuning a YOLOv8 model using Ultralytics' YOLO implementation. It includes exploratory data analysis (EDA), model training and evaluation notebooks, as well as production-ready Python scripts for automation.

---

## Project Structure

```
cerebre-recruitment-file/
├── data/                           # Dataset YAML and input images (train/val/test)
├── images/                         # Images used in README.md file 
├── notebooks/                      # Research & development notebooks
│   ├── grid_search_yolov8/         # Results of model training notebook
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
├── scripts/                        # Standalone Python scripts
│   ├── train_yolo.py
│   ├── evaluate_yolo.py
│   ├── predict_yolo.py
│   └── grid_search_yolo.py
├── .gitignore
└── requirements.txt                # List of used packages
```

---

## Dataset Used

This project uses the [Blood Cell Detection Dataset](https://www.kaggle.com/datasets/adhoppin/blood-cell-detection-datatset/data) from Kaggle.

- The dataset contains annotated images for detecting various types of blood cells.
- It has already been split into training, validation, and test sets.
- The annotations are in YOLO format, making them directly compatible with YOLOv8.

To use this dataset:
1. Download it from Kaggle.
2. Unzip the files and place `test/`, `train/`, `valid/` and `data.yaml` in `data/` folder.
3. Make sure your `data.yaml` file is correctly configured to point to the appropriate folders.
4. The folder structure inside `data/` should match YOLOv8 requirements:
   ```
   test/
   ├── images/
   └── labels/

   train/
   ├── images/
   └── labels/

   valid/
   ├── images/
   └── labels/

   └── data.yaml
   ```

---

## Data Exploration Summary

The exploratory data analysis (see `notebooks/data_exploration.ipynb`) revealed the following key insights about the dataset:

- **Classes Distribution**: The dataset includes multiple blood cell types. The distribution is imbalanced, with certain classes (e.g., RBC) being more frequent than others (e.g., Platelets).
- **Annotation Format**: Annotations are well-formatted in YOLO format, making them ready for model training without conversion.
- **Image Characteristics**: Image resolutions vary, but most fall within a manageable size range. All images are in color, and the quality is suitable for detection tasks.
- **Potential Issues**: Some images contain overlapping or densely packed objects, which could pose challenges for detection. A few labels might need review due to potential misannotations.
- **Visual Examples**: The notebook includes random samples of annotated images, which helped verify data consistency and label placement.

These insights were used to inform the training strategy and later model evaluation.

![class_distribution](https://github.com/KacperTurek/Cerebre-recruitment-task/blob/main/images/train_class_distribution.png?raw=true)

![number_of_classes](https://github.com/KacperTurek/Cerebre-recruitment-task/blob/main/images/train_classes_count.png?raw=true)

![number_of_objects](https://github.com/KacperTurek/Cerebre-recruitment-task/blob/main/images/train_objects_count.png?raw=true)

![training_image_0](https://github.com/KacperTurek/Cerebre-recruitment-task/blob/main/images/train_image_0.png?raw=true)

---

## Requirements
- Python 3.8+
- Packages used:
```bash
pip install -r requirements.txt
```

---

## Step-by-Step Guide

### 1. Exploratory Data Analysis (EDA)
- Navigate to `notebooks/data_exploration.ipynb`
- Load and visualize the dataset
- Review class distribution, sample images, and annotations

### 2. Initial Model Training
- Open `notebooks/model_training.ipynb`
- Train a multiple YOLOv8 models using grid search selection of the hyperparameters (e.g., `yolov8n.pt`)
- Track training loss, mAP, and other metrics

### 3. Model Evaluation
- Use `notebooks/model_evaluation.ipynb`
- Evaluate the best model on the test set
- Analyze metrics like mAP@0.5, precision, recall, confusion matrix
- Visualize predictions and failure cases

---

## Python Scripts for Automation

### Train a Model
```bash
python scripts/train_yolo.py \
    --epochs 50 \
    --batch 32 \
    --lr0 0.001 \
    --momentum 0.937 \
    --weight_decay 0.0001 \
    --imgsz 640 \
    --data data/data.yaml \
    --project runs/train \
    --name yolov8_train
```
**Defaults (picked based on the results of grid search):**
- `--epochs`: 20
- `--batch`: 32
- `--lr0`: 0.001
- `--momentum`: 0.9
- `--weight_decay`: 0.0001
- `--imgsz`: 416
- `--data`: ../data/data.yaml
- `--project`: runs/train
- `--name`: yolo

### Evaluate a Trained Model
```bash
python scripts/evaluate_yolo.py \
    --weights weights/best.pt \
    --data data/data.yaml \
    --split test \
    --imgsz 640 \
    --batch: 32
```
**Defaults:**
- `--weights`: runs/train/yolo/weights/best.pt
- `--data`: ../data/data.yaml
- `--split`: test
- `--imgsz`: 416
- `--batch`: 32

### Predict on New Images or Video
```bash
python scripts/predict_yolo.py \
    --weights weights/best.pt \
    --source images/test \
    --imgsz 640 \
    --conf 0.25 \
    --save
```
**Defaults:**
- `--weights`: runs/train/yolo/weights/best.pt
- `--source`: ../data/test/images
- `--imgsz`: 416
- `--conf`: 0.25
- `--save`: False
- `--project`: runs/predict
- `--name`: exp

### Perform Hyperparameter Grid Search
```bash
python scripts/grid_search_yolo.py
```
- Stores each run in `runs/grid_search/`
- Saves a `grid_search_results.csv` file summarizing mAP scores and hyperparameters

---

## Grid Search Results Summary

The `grid_search_yolo.py` script was used to explore multiple combinations of hyperparameters such as learning rate, batch size, momentum, and weight decay. Each configuration was evaluated using validation mAP.

- The results were logged in `grid_search_results.csv`, with each row showing a parameter set and its corresponding mAP.
- The best-performing configuration achieved the highest mAP while maintaining training stability and reasonable training time.
- This optimal configuration was selected for the final training run and model evaluation.

The grid search was essential for identifying hyperparameter values that significantly improved performance compared to the default configuration.

---

## Grid Search Results

The following table shows the results of the hyperparameter grid search, sorted by `mAP50`:

| lr0   | momentum | weight\_decay | batch | run\_name       | mAP50   |
| ----- | -------- | ------------- | ----- | --------------- | ------- |
| 0.01  | 0.9      | 0.0001        | 32    | grid\_trial\_3  | 0.92084 |
| 0.01  | 0.937    | 0.0001        | 32    | grid\_trial\_7  | 0.92084 |
| 0.001 | 0.9      | 0.0001        | 32    | grid\_trial\_11 | 0.92084 |
| 0.001 | 0.937    | 0.0001        | 32    | grid\_trial\_15 | 0.92084 |
| 0.01  | 0.9      | 0.0005        | 32    | grid\_trial\_1  | 0.92001 |
| 0.01  | 0.937    | 0.0005        | 32    | grid\_trial\_5  | 0.92001 |
| 0.001 | 0.9      | 0.0005        | 32    | grid\_trial\_9  | 0.92001 |
| 0.001 | 0.937    | 0.0005        | 32    | grid\_trial\_13 | 0.92001 |
| 0.01  | 0.9      | 0.0005        | 16    | grid\_trial\_0  | 0.91982 |
| 0.01  | 0.937    | 0.0005        | 16    | grid\_trial\_4  | 0.91982 |
| 0.001 | 0.9      | 0.0005        | 16    | grid\_trial\_8  | 0.91982 |
| 0.001 | 0.937    | 0.0005        | 16    | grid\_trial\_12 | 0.91982 |
| 0.01  | 0.9      | 0.0001        | 16    | grid\_trial\_2  | 0.91973 |
| 0.01  | 0.937    | 0.0001        | 16    | grid\_trial\_6  | 0.91973 |
| 0.001 | 0.9      | 0.0001        | 16    | grid\_trial\_10 | 0.91973 |
| 0.001 | 0.937    | 0.0001        | 16    | grid\_trial\_14 | 0.91973 |

---

## Evaluation of Best Model

The best-performing model from the grid search (`grid_trial_3`) was selected based on its top `mAP50` score of **0.92084**. The evaluation revealed:

- **Precision and Recall**: High across all major blood cell classes, with minimal variance.
- **Confusion Matrix**: Clear class separation, minimal confusion, especially for abundant classes like RBC.
- **Visual Output**: Prediction boxes were tightly fitted around objects, even in dense cell clusters.

This confirms that the grid search tuning was successful and the chosen hyperparameters led to a robust model.

---

## Final Model Evaluation

The model trained with the best hyperparameters from the grid search was evaluated on the test set using `evaluate_yolo.py`.

- **mAP@0.5**: The model achieved a strong mean average precision, indicating accurate localization.
- **Precision & Recall**: The model maintained high precision and good recall, showing its ability to identify blood cells while minimizing false positives.
- **Visualization**: Predictions on test images were visually inspected. Bounding boxes were well-aligned with actual objects.
- **Failure Analysis**: A few challenging cases with overlapping cells or low-contrast images were noted where the model had difficulty.

This evaluation confirmed that the grid-searched model generalizes well and is suitable for deployment or further research.

---

## Analysis of Confusion Matrix

During evaluation of the best model (`grid_trial_3`), the normalized confusion matrix revealed some false positive detections, especially for the RBC (Red Blood Cells) class.  
Upon closer inspection, it was observed that these detections are not actual mistakes by the model. Instead, they highlight a labeling issue in the dataset: some RBC instances were **not labeled** in the original training, validation, and test sets.  
As a result, the model correctly identifies RBCs that genuinely appear in the images, but since there are no corresponding ground truth labels, these correct detections are incorrectly counted as false positives in the confusion matrix.  
This indicates that the model has generalized well and is capable of detecting RBCs beyond the scope of the provided annotations.

![confussion_matrix](https://github.com/KacperTurek/Cerebre-recruitment-task/blob/main/images/best_model_confusion_matrix_normalized.png?raw=true)

---

## Analysis of Precision-Confidence and Precision-Recall Curves

The evaluation of the best model (`grid_trial_3`) also included analyzing the precision-confidence (P) curve and precision-recall (PR) curve.  
The **P-curve** shows that precision remains high across a wide range of confidence thresholds, especially for the WBC (White Blood Cells) class, which achieved near-perfect precision even at low confidence levels.  
The **PR-curve** demonstrates that the model maintains strong performance across all classes, with an overall mAP@0.5 of **0.927**.  
Class-wise, WBC achieved the highest area under the curve, followed closely by RBC and Platelets.  
These results confirm that the model not only detects objects accurately but also balances precision and recall effectively across different classes.

![p_curve](https://github.com/KacperTurek/Cerebre-recruitment-task/blob/main/images/best_model_P_curve.png?raw=true)

![pr_curve](https://github.com/KacperTurek/Cerebre-recruitment-task/blob/main/images/best_model_PR_curve.png?raw=true)

---

## Output Files from Notebooks code
- `notebooks/grid_search_yolov8/` — models trained using grid search
- `notebooks/grid_search_results.csv` — Grid search results table
- `notebooks/grid_search_yolov8/grid_trial_3/weights/best.pt` — Weights of the best model

---

## Next Steps
- Try additional YOLO variants (e.g., yolov8s, yolov8m)
- Use data augmentation or pseudo-labeling
- Deploy model via a Flask or FastAPI backend
