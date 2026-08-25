# Dog vs Cat Image Classification

Binary image classification project that identifies whether an image contains a dog or a cat, using transfer learning with MobileNetV2. This is Project 2 in an ongoing 10-project machine learning series.

## Overview

- **Task**: Binary image classification (cat = 0, dog = 1)
- **Dataset**: [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats) competition dataset
- **Approach**: Transfer learning on top of a pretrained MobileNetV2 backbone
- **Result**: 98.03% accuracy on the held-out test set

#Colab Link:https://colab.research.google.com/drive/1uxgc0bMHuD8b5dqZwvSDv3o7073eViBC#scrollTo=bVyr5uCjW0Qk


## Dataset

- Full dataset: 25,000 labeled training images (12,500 dogs, 12,500 cats), downloaded via the Kaggle API
- For training, a balanced subset of 2,000 images (1,000 cats + 1,000 dogs) was used to keep training time manageable
- All images resized to 224x224x3 and pixel values scaled to the [0, 1] range

## Pipeline

1. Download and extract the dataset from Kaggle
2. Explore the data — verify class balance, visualize sample dog/cat images
3. Resize all images to 224x224
4. Convert images to NumPy arrays and generate labels from filenames
5. Split into train/test sets (80/20)
6. Scale pixel values
7. Build and train the model
8. Evaluate on the test set
9. Run predictions on new, unseen images via a simple predictive system

## Model Architecture

| Layer | Output Shape | Params |
|---|---|---|
| MobileNetV2 (frozen, ImageNet weights, `pooling='avg'`) | (None, 1280) | 2,257,984 |
| Dense (2 units) | (None, 2) | 2,562 |

- **Total params**: 2,260,546
- **Trainable params**: 2,562
- **Non-trainable params**: 2,257,984

The MobileNetV2 backbone is frozen and used purely as a feature extractor; only the final dense classification head is trained.

## Training Configuration

- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy (`from_logits=True`)
- **Metric**: Accuracy
- **Epochs**: 5

## Results

| Metric | Value |
|---|---|
| Training accuracy (final epoch) | 98.65% |
| Test accuracy | 98.03% |
| Test loss | 0.079 |

Training accuracy progressed steadily across epochs: 90.15% → 97.23% → 97.48% → 98.46% → 98.65%.

## Predictive System

The notebook includes an inference cell that:
1. Takes the file path of any image as input
2. Resizes and scales it to match the model's expected input
3. Returns a prediction: cat or dog

This was tested on random dog/cat images sourced outside the training set to confirm the model generalizes.

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn (`train_test_split`)

## How to Run

1. Get a Kaggle API token (`kaggle.json`) and place it in the project directory.
2. Open `Dog_VS_Cat_classification.ipynb` and run the cells in order — the dataset downloads and extracts automatically via the Kaggle API.
3. Once training completes, use the predictive system cell at the end of the notebook to test the model on your own images.

## Notes

This project is part of a self-directed 10-project machine learning series, moving from classical ML on tabular data toward deep learning and computer vision.
