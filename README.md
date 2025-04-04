
# Dental Anomaly Detection using YOLOv8 Segmentation
- ![val_batch2_pred.jpg](assets/val_batch2_pred.jpg)

This project uses the **Tufts Dental Database: A Multimodal Panoramic X-Ray Dataset** to detect dental anomalies such as missing teeth, implants, or lesions using a YOLOv8-based pipeline combining both object detection and mask segmentation. We have reported the performance of our model which is greater then some of the best reported acadmeic work on TUFTS dataset.

---

## 📚 Project Overview

Our pipeline involves four sequential processing steps:

1. **Maxillomandibular Region Detection**  
   Isolating the region of interest in the panoramic scan.  
   ![overlay_maxillomandibular.jpg](assets/overlay_maxillomandibular.jpg)


2. **Tooth Segmentation**  
   Extracting the tooth structures from the X-ray for individual inspection.  
   ![overlay_teeth.jpg](assets/overlay_teeth.jpg)

3. **Anomaly Detection**  
   Identifying and segmenting anomalies within the detected teeth regions.  
   ![overlay_anomalies.jpg](assets/overlay_anomalies.jpg)

## 📦 Dataset

**Source**: [Tufts Dental Database](https://arxiv.org/abs/2312.06226)

**Details**:
- Panoramic dental X-rays (high resolution)
- Annotated anomalies under the class label: `anomaly`
- Available in YOLO format for bounding boxes and polygon format for segmentation
- Distribution visualization: The image below shows the distribution of our labeled bounding box dataset. 
![labels.jpg](assets/labels.jpg)

---
## 🧠 Model Pipeline

| Task               | Model     | Description                                      |
|--------------------|-----------|--------------------------------------------------|
| Object Detection   | YOLOv8    | Detect bounding boxes around anomalies          |
| Mask Segmentation  | YOLOv8-seg| Segment anomalies within detected regions       |

---

## 🏋️‍♂️ Training Overview
We have fine tuned the Yolox model for 200 epochs on TUFTS dataset with training observation as shown in the image below.

- **Epochs**: 200
- **Losses**: Classification, Box, Segmentation, and Distribution Focal Loss (DFL)
- Training and validation loss curves show strong convergence:

![results.png](assets/results.png)

---

## 📊 Evaluation Metrics

### 📈 Detection and Segmentation Metrics

| Metric          | Detection (Box) | Segmentation (Mask) |
|------------------|------------------|----------------------|
| mAP@0.5          | 0.946            | 0.942                |
| F1 Score (Max)   | 0.95 @ 0.479     | 0.95 @ 0.478         |
| Precision (Max)  | 1.00 @ 0.826     | 1.00 @ 0.826         |
| Recall (Max)     | 0.95 @ 0.000     | 0.95 @ 0.000         |

---
## 📊 TUFTS comparison

| Metric               | Our Model  | Tufts Paper (Benchmark) |
|----------------------|------------|--------------------------|
| mAP@0.5 (Detection)  | **0.946**  | ~0.82 (YOLOv5s)          |
| mAP@0.5 (Segmentation)| **0.942** | ~0.84 (ResNet-FCN)       |
| Max F1 Score         | 0.95       | N/A                      |
| Max Precision        | 1.00       | N/A                      |
| Max Recall           | 0.95       | N/A                      |

🔍 Our process  outperforms both the YOLOv5s and FCN baselines reported in the Tufts paper.

### 📉 Precision, Recall & F1 Curves


**Bounding Box Metrics**
<div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
  <div style="flex: 1 1 22%; text-align: center; margin: 10px;">
    <p>F1 vs Confidence:</p>
    <img src="BoxF1_curve.png" alt="F1 vs Confidence" style="width: 100%;">
  </div>
  <div style="flex: 1 1 22%; text-align: center; margin: 10px;">
    <p>Precision vs Confidence:</p>
    <img src="BoxP_curve.png" alt="Precision vs Confidence" style="width: 100%;">
  </div>
  <div style="flex: 1 1 22%; text-align: center; margin: 10px;">
    <p>Precision-Recall Curve:</p>
    <img src="BoxPR_curve.png" alt="Precision-Recall Curve" style="width: 100%;">
  </div>
  <div style="flex: 1 1 22%; text-align: center; margin: 10px;">
    <p>Recall vs Confidence:</p>
    <img src="BoxR_curve.png" alt="Recall vs Confidence" style="width: 100%;">
  </div>
</div>


**Mask Segmentation Metrics**
<div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
  <div style="flex: 1 1 22%; text-align: center; margin: 10px;">
    <p>F1 vs Confidence:</p>
    <img src="MaskF1_curve.png" alt="F1 vs Confidence" style="width: 100%;">
  </div>
  <div style="flex: 1 1 22%; text-align: center; margin: 10px;">
    <p>Precision vs Confidence:</p>
    <img src="MaskP_curve.png" alt="Precision vs Confidence" style="width: 100%;">
  </div>
  <div style="flex: 1 1 22%; text-align: center; margin: 10px;">
    <p>Precision-Recall Curve:</p>
    <img src="MaskPR_curve.png" alt="Precision-Recall Curve" style="width: 100%;">
  </div>
  <div style="flex: 1 1 22%; text-align: center; margin: 10px;">
    <p>Recall vs Confidence:</p>
    <img src="MaskR_curve.png" alt="Recall vs Confidence" style="width: 100%;">
  </div>
</div>

---

## 📊 Confusion Matrix
<div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
  <div style="flex: 1 1 45%; text-align: center; margin: 10px;">
    <p>Raw Confusion Matrix:</p>
    <img src="confusion_matrix.png" alt="Raw Confusion Matrix" style="width: 100%;">
  </div>
  <div style="flex: 1 1 45%; text-align: center; margin: 10px;">
    <p>Normalized Confusion Matrix:</p>
    <img src="confusion_matrix_normalized.png" alt="Normalized Confusion Matrix" style="width: 100%;">
  </div>
</div>

**Insights**:
- 93% of anomalies were correctly predicted.
- Background misclassifications were minimal, indicating strong class separation.

---


## 💡 Key Observations

- **Excellent F1 Scores** (0.95) at moderate confidence thresholds.
- **Perfect precision** (1.0) achieved at high confidence, indicating highly trustworthy predictions.
- **High mAP@0.5** for both detection and segmentation proves the model’s capability to localize and delineate anomalies effectively.
- Confusion matrix confirms minimal false positives and false negatives.
- Label distribution is reasonably balanced and spatially consistent across the dataset.

---

## 🧪 Sample Visualizations


### 📦 Ground Truth (Train)
<div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
  <div style="flex: 1 1 25%; text-align: center; margin: 10px;">
    <img src="train_batch2.jpg" alt="Train Batch 2" style="width: 100%;">
  </div>
  <div style="flex: 1 1 25%; text-align: center; margin: 10px;">
    <img src="train_batch6082.jpg" alt="Train Batch 6082" style="width: 100%;">
  </div>
</div>

### 🧾 Ground Truth Labels (Validation)
- ![val_batch2_labels.jpg](assets/val_batch2_labels.jpg)

### 🧠 Model Predictions (Validation)
- ![val_batch2_pred.jpg](assets/val_batch2_pred.jpg)

---


## 🚀 Future Scope

- Introduce multi-label anomaly classification (e.g., caries, implants).
- Build anomaly progression tracking over time from X-ray series.
- Explore 3D reconstruction or CT-assisted learning.

---

## 🛠️ Tools & Frameworks

- **YOLOv8**: Ultralytics
- **Python**: Data preprocessing, training
- **OpenCV + Matplotlib**: Visualization
- **Pandas/Numpy**: Data analysis

---


## 📌 Conclusion

This project demonstrates the viability of deep learning-based anomaly detection in dental radiography using object detection and segmentation techniques. With near-perfect precision and robust recall, this system can significantly assist dental practitioners in early diagnosis and treatment planning.
