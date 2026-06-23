# VisualML

**See how a text classifier actually works — one stage at a time.**

VisualML is a SwiftUI iOS app that takes a classic text‑classification pipeline
— **Bag‑of‑Words → TF‑IDF → LSA → linear classifier** — and *draws every
intermediate state* so you can watch raw text turn into numbers, into a 2‑D map,
into a decision boundary. Every stage has live knobs: flip a switch and the
heatmap, the scatter, and the accuracy all recompute in front of you.

The running example classifies short documents as **Sport** vs **Business**.

---

## What it does

Most ML tutorials show you the input and the output and hide everything in
between. VisualML makes the middle visible. You start at a home menu, pick a
stage, and drill into the data representation at that point in the pipeline:

| | |
|---|---|
| <img src="screenshots/home.png" width="240" alt="Home menu"> | **Home** — pick a model family. *Bag of Words* walks the text‑representation stages; *Bag‑of‑Words + Regression* trains a classifier on top. The greyed‑out rows are placeholders for models added later. |
| <img src="screenshots/matrix.png" width="240" alt="Document-Term matrix heatmap"> | **Document‑Term Matrix** — 500 documents × 150 terms drawn as a heatmap, one row per document, grouped by class. Toggle **Raw counts ⇄ TF‑IDF** and **L2‑normalize** and watch the cells re‑weight: ubiquitous words dim, rare telling words brighten. |
| <img src="screenshots/lsa.png" width="240" alt="LSA 2-D scatter"> | **LSA projection** — every document compressed to its top‑2 latent components (truncated SVD) and plotted as a point. The two classes fall into two separated clouds, which is exactly what makes them linearly separable. |
| <img src="screenshots/classifier.png" width="240" alt="Decision boundary"> | **Classifier** — fit **Logistic / Linear / SVM** on the 2‑D LSA points and see the decision boundary `wᵀx + b = 0` drawn through the data. Drag the learning‑rate, iterations, and regularization sliders to see the line move. |
| <img src="screenshots/training.png" width="240" alt="Step-by-step training"> | **Training, step by step** — the boundary starts as an *arbitrary* random line (here: 48% accuracy). Press play or scrub the slider to replay gradient descent iteration‑by‑iteration and watch the line rotate into place as the accuracy climbs. |
| <img src="screenshots/metrics.png" width="240" alt="Accuracy and confusion matrix"> | **Evaluation** — train/test accuracy on a held‑out split, a colour‑coded confusion matrix, and a **“classify your own sentence”** box that runs your text through the whole pipeline and drops it onto the scatter. |

---

## Topics covered so far

The app is a working tour of a full classical‑NLP classification stack:

**Text → numbers**
- **Tokenization & vocabulary building** (with optional stop‑word removal and a min‑document‑frequency / max‑vocabulary cap)
- **Bag‑of‑Words** — the document‑term count matrix
- **TF‑IDF weighting** — `idf = ln((1 + D) / (1 + df)) + 1`, multiplied into the counts to down‑weight common words and up‑weight discriminative ones
- **L2 normalization** — scaling each document vector to unit length so document length stops dominating

**Dimensionality reduction**
- **Latent Semantic Analysis (LSA)** = **truncated SVD** of the weighted matrix
- Computed via the **Gram matrix** `G = W·Wᵀ` and **power iteration with deflation** (`G·v = W·(Wᵀ·v)`), so it scales to the full 500‑document set without a heavy linear‑algebra dependency
- **Document coordinates** `U·Σ`, **term loadings** `Wᵀ·U / σ`, a **scree plot** of the singular values, and **folding‑in** to project a brand‑new sentence into the existing latent space

**Classification**
- **Logistic Regression** — sigmoid + cross‑entropy loss
- **Linear Regression as a classifier** — least‑squares to {0, 1} with a 0.5 threshold, kept deliberately as a *contrast* to logistic regression
- **Linear SVM** — hinge loss with a sub‑gradient update and visible ±1 margins
- **Gradient descent** with adjustable learning rate, iteration count, and L2 regularization
- **Feature standardization** before fitting

**Evaluation**
- **Seeded train/test split**
- **Accuracy**, **confusion matrix**, and a live **decision boundary**

---

## Requirements

- **macOS** with **Xcode** (an SDK that supports **iOS 26.5**)
- An **iPhone 17 Pro** simulator (or any iOS 26.5 simulator / device)
- No third‑party packages — the app uses only **SwiftUI** and the standard library. The matrix/SVD math is hand‑written, so there’s nothing to install.

> The dataset ships in the repo (`Models/dataset.csv`), so the app runs fully
> offline with no setup.

---

## Getting started

```bash
git clone <your-repo-url>
cd VisualML
open VisualML.xcodeproj
```

Then in Xcode:

1. Select the **VisualML** scheme and an **iPhone 17 Pro** simulator.
2. Press **▶ Run** (`⌘R`).
3. From the home screen, tap **Bag of Words** to walk the pipeline, or
   **Bag‑of‑Words + Regression** to jump straight to the classifier.

---

## How it’s built

The app follows **MVVM**. A single `PipelineViewModel` owns the configuration
and caches each artifact; changing a knob recomputes only what depends on it
(a vocabulary change rebuilds everything; a weighting change re‑runs LSA; a
classifier knob just retrains). The heavy LSA compute runs **off the main
thread** and is cached. All the heatmaps and scatter plots are drawn with
SwiftUI `Canvas`.

```
VisualML/
├─ Models/
│  ├─ DataPoint.swift          // one labelled document
│  ├─ PipelineModels.swift     // config + every intermediate type (matrix, SVD result, …)
│  └─ dataset.csv              // 500 docs (250 sport / 250 business)
├─ Services/
│  ├─ DataSetLoader.swift      // CSV parser
│  ├─ NLPProcessor.swift       // tokenize → vocabulary → document-term matrix
│  ├─ Weighting.swift          // TF-IDF + L2 normalization
│  ├─ MatrixMath.swift         // truncated SVD via power iteration + deflation
│  └─ Classifier.swift         // logistic / linear / SVM gradient descent
├─ ViewModels/
│  └─ PipelineViewModel.swift  // holds config + cached artifacts (MVVM)
└─ Views/
   ├─ HomeView.swift           // root menu
   ├─ BagOfWordsFlowView.swift // stage navigation (dataset → matrix → LSA)
   ├─ DatasetView.swift        // the labelled documents
   ├─ MatrixHeatmapView.swift  // the document-term heatmap
   ├─ LSAView.swift            // 2-D latent-space scatter + scree plot
   ├─ ClassifierView.swift     // decision boundary + metrics + live classify
   ├─ TrainingView.swift       // step-by-step gradient-descent playback
   └─ …Info sheets, layout helpers
```

---

## Dataset

`Models/dataset.csv` holds **500 generated documents** (250 sport, 250 business),
two columns: `category,text`. The vocabulary is capped at 150 terms. Document
lengths vary widely on purpose — short documents land near the origin of the LSA
plot while long ones fan outward, which is what gives the 2‑D projection its
characteristic spread and makes the structure easy to see.

---

## Roadmap

The home screen already lists what’s next — each will get the same
“visualize every intermediate state” treatment:

- **Clustering (k‑means)**
- **Decision tree**
- **Neural network**
