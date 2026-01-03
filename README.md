# Z-DNA and G-Quadruplex Prediction in Stem Cells with CNN approach

## Project overview

This repository contains the complete implementation and experimental framework for my Bachelor's thesis:

"Optimizing Applications of Deep Learning Models for Genomics Problems to the Subject Area of Secondary Structures in Stem Cells"

This project bridges computational genomics, deep learning, and stem cell biology to advance understanding of non-canonical DNA structures in development and disease. It focuses on predicting non-canonical DNA secondary structures—specifically Z-DNA and G-quadruplexes (G4)—in pluripotent stem cells using convolutional neural networks (CNNs) with integrated multi-omics data and explainable AI (xAI) interpretation.

## Research context

DNA exists in multiple structural forms beyond the canonical B-DNA. Among them:

Z-DNA: Left-handed zigzag structure linked to Alzheimer’s, cancer, and gene regulation.

G-quadruplexes (G4): Four-stranded structures formed by G-rich sequences, involved in transcription, replication, and neurodegenerative diseases.

These structures are rare (~0.3–2% of the genome) and may be cell-type-specific. Existing models either ignore omics data or are trained genome-wide without cell specificity.

## Key Contributions

1. **Cell-specific modeling:** First application of deep learning for Z-DNA and G4 prediction in pluripotent stem cells.

2. **Omics integration for G4:** First model to incorporate histone modifications, TF binding, chromatin accessibility, etc., for G4 prediction.

3. **Architecture optimization:** Systematic hyperparameter tuning, residual connections, dilation adjustments, and interval-width experiments.

4. **Interpretable AI:** Used an ensemble of xAI methods (Integrated Gradients, InputXGradient, Guided Backpropagation, Deconvolution) to identify biologically relevant omics features.

5. **Biological validation:** Linked top-ranked omics features to known literature, confirming model plausibility.


## Results summary

### G4 prediction

- Best F1-score: 76.77% against 67.86% baseline
- Key omics features: TP53BP1, H3K4me3, BCOR, ATAC-seq, DNase-seq
- Confirmed biological relevance with several top omics features found by model


### Z-DNA prediction

- Best F1-score: 89.96% against 88.15% baseline
- Key omics features: H3K4me3, BRCA1, RNA Polymerase II, H2A.Z, KDM2A
- Comparison with whole-genome model: Showed both shared and stem-cell-specific regulatory features.

## Model architecture

### Baseline CNN (from HSE Bioinformatics Lab)

- Input: One-hot encoded DNA sequence + omics feature matrix
- Convolutional blocks with increasing/decreasing channels
- Fully connected layers with dropout
- Output: Binary classification (structure present/absent)
- This architecture has been used unmodified for Z-DNA (but to stem sells domain only)

### Optimized G4 CNN

- 8 convolutional layers with residual connections
- Kernel size 3, various dilations and strides
- Activation: LeakyReLU
- Optimizer: AdamW (LR=2e-4, weight decay=5e-5)
- Scheduler: ReduceLROnPlateau


## Key Experiments Conducted
- Omics encoding schemes (scaled, binary, max-scaled, etc.)
- Interval width tuning (30-300 nucleotids)
- Hyperparameter optimization with Optuna
- Experiments with dilations, dropout, activation functions
- Add residual connection blocks
- Feature importance analysis using xAI ensemble

## Interpretation Framework

Four xAI methods were used:

1. Integrated Gradients
2. InputXGradient
3. Guided Backpropagation
4. Deconvolution

Features were ranked by mean absolute deviation across methods. Top omics features were validated against biological literature.

## Datasets
- DNA sequence: hg38 (UCSC)
- Omics data: ChIP-Atlas (ATAC-seq, Histone ChIP-seq, TF ChIP-seq, etc.)
- Z-DNA labels: Kouzine dataset
- G4 labels: EndoQuad database (confidence ≥ 4)

## Project structure

```text
ZDNA_G4_Research/
├── data/                                 # Processed datasets (sparse)
├── docs/                                 # Thesis paper and presentation
├── interpretation/                       # Interpretation notebooks and results
├── models/                               # Saved models (including baseline, final and intermediate ones)
├── .gitignore
├── CNN_model_G4_Experiments.ipynb        # Main notebook for G4 experiments
├── CNN_model_G4_Optuning.ipynb           # Notebook with tuning hyperparameters for G4 model before manual experiments
├── CNN_model_ZDNA.ipynb                  # Main notebook for ZDNA experiments
├── DataPreparator.ipynb                  # Notebook for preparing datasets
├── OmicsLoader.ipynb                     # Notebook for loading omics data from ChIP-Atlas
├── README.md
```
