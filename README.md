# ✏️ Handwritten Math Symbol Classification

A computer vision project comparing three feature extraction techniques and two classification models for recognising handwritten mathematical symbols.

---

## 📌 Overview

This project classifies handwritten mathematical symbols — `=`, `4`, `9`, `b`, `C`, `cos`, `≥`, `≤`, `M`, `→` — into their correct categories. Rather than jumping straight to a single model, the project systematically compares **three feature representations** against **two classifiers**, producing a 3×2 experimental grid to evaluate which combination generalises best.

The core question explored: *do handcrafted features (HOG, LBP) outperform raw pixel input, and does a simple neural network outperform a classical SVM on each?*

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| OpenCV | Image loading and preprocessing |
| scikit-image | HOG and LBP feature extraction |
| scikit-learn | SVM classifier, train/test split, evaluation |
| TensorFlow / Keras | Neural network classifier |
| Pandas / Matplotlib / Seaborn | Data handling and visualisation |

---

## ⚙️ Pipeline

### 1. Data Loading
- Grayscale images loaded from per-class subfolders and resized to 28×28
- Labels encoded numerically via `LabelEncoder`
- Single stratified 80/20 train/test split applied once, then reused consistently across all feature extraction methods to ensure a fair comparison

### 2. Feature Extraction

Three parallel representations generated from the same images:

| Method | Description |
|--------|-------------|
| **Raw Pixels** | Flattened 28×28 grayscale pixel values (784 features) |
| **HOG** (Histogram of Oriented Gradients) | Captures edge direction and gradient structure — effective for shape-based recognition |
| **LBP** (Local Binary Patterns) | Captures local texture patterns, robust to lighting variation |

### 3. Classification Models

Each feature set is evaluated with two classifiers:

- **SVM** (RBF kernel, `C=100`)
- **ANN** — a simple fully-connected network (one hidden layer, 512 units, softmax output)

This produces **6 total experiments**: {Raw, HOG, LBP} × {SVM, ANN}

### 4. Evaluation
- Train and test accuracy recorded for every combination
- Confusion matrix generated for the SVM + HOG combination
- Training history (accuracy curve) plotted for each ANN run
- Final results consolidated into a comparison table and grouped bar chart

---

## 📊 Results

All six feature/model combinations are summarised in a single comparison table and visualised as a grouped bar chart, ranked by test accuracy. This makes it possible to directly answer:

- Which feature extraction method generalises best to unseen symbols?
- Does the added complexity of a neural network outperform a classical SVM on the same features?
- Is raw pixel input sufficient, or does feature engineering provide a meaningful boost?

*(Run the notebook to populate exact accuracy figures — results depend on the specific dataset split provided.)*

---

## 📁 Repository Structure

```
├── Notebook.ipynb              # Full pipeline notebook
├── student_24956928/           # Dataset folder (per-class image subfolders)
│   ├── =/
│   ├── 4/
│   ├── cos/
│   └── ...
└── README.md
```

---

## 🚀 Future Work

- Replace handcrafted features with a CNN to learn features directly from raw images
- Apply data augmentation (rotation, scaling, noise) to improve generalisation, especially for underrepresented symbol classes
- Expand the dataset size for more robust train/test splits
- Experiment with ensemble methods combining HOG and LBP features

---

## 👤 Author

**Aryan Raheja**  
Bachelor of Computing Science (AI & Data Analytics), University of Technology Sydney
