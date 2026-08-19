# Deepfake Face Detection using Vision Transformer

A deep learning project for detecting whether a facial image is real or AI-generated using a Vision Transformer (ViT).

## Project Overview

The increasing quality of AI-generated and manipulated facial images makes it difficult to distinguish between real and fake images using visual inspection alone.

This project develops a binary image classification system using a pretrained Vision Transformer to classify facial images into two categories:

- Real
- Fake

The model uses transfer learning from a pretrained Vision Transformer and adapts it for binary classification.

The project also includes model training, evaluation, prediction, and a Streamlit-based application for using the trained model.

## Approach

The overall pipeline is:

```text
Input Image
     |
     v
Image Preprocessing
     |
     v
Vision Transformer
     |
     v
Binary Classification
     |
     +------------------+
     |                  |
     v                  v
   Real                Fake

The system processes the input image using the same preprocessing pipeline used during model development and passes it through the Vision Transformer.

Model

The project uses:

Vision Transformer (ViT)
google/vit-base-patch16-224
PyTorch
Hugging Face Transformers

The pretrained ViT model is adapted for two-class classification.

The original pretrained classifier is designed for ImageNet classification. For this project, the classification layer is replaced/reconfigured for the two target classes:

Real
Fake

To reduce the amount of training required, the project uses a transfer-learning approach. Most of the pretrained model parameters are kept frozen while selected parts of the transformer are fine-tuned for the deepfake detection task.

Dataset

The training data is organized into three splits:

dataset/
└── real_vs_fake/
    └── real-vs-fake/
        ├── train/
        ├── valid/
        └── test/

The dataset used for the project contains:

100,000 training images
20,000 validation images
20,000 test images

The two classes are:

fake
real

The dataset is not included in this repository because of its size.

Data Preprocessing

Images are resized to the input size required by the Vision Transformer.

The training pipeline includes data augmentation such as:

Image resizing
Random horizontal flipping
Random rotation
Color jitter
Normalization

Validation and test images are processed without the training-specific random augmentations.

Training

The training pipeline is implemented using PyTorch.

The main training components include:

Cross Entropy Loss
AdamW optimizer
Cosine Annealing learning-rate scheduler
Validation-based model monitoring
Checkpoint saving

The training process keeps track of:

Training loss
Training accuracy
Validation loss
Validation accuracy

Model checkpoints can be saved during training so that training progress can be preserved and the best model can be retained.

Evaluation

The project includes a separate evaluation pipeline for testing the trained model.

The evaluation process generates results including:

Classification report
Confusion matrix
ROC curve
Prediction results

The repository contains the generated evaluation files:

classification_report.txt
evaluation_results.txt
confusion_matrix.png
roc_curve.png
predictions.csv

These files provide different views of the model's performance on the test data.

Prediction

The project includes a prediction script that can be used to classify individual images using the trained model.

The prediction workflow is:

Image
  |
  v
Preprocessing
  |
  v
Trained ViT Model
  |
  v
Prediction
  |
  v
Real / Fake
Application

The repository also contains Streamlit application files:

app.py
app2.py

These provide an interface for interacting with the trained deepfake detection model.

The application is intended to make the model easier to use without requiring the user to directly interact with the training or evaluation scripts.

Project Structure
deepfake-face-detection-vit/
│
├── app.py
├── app2.py
│
├── train.py
├── test.py
├── evaluate.py
├── predict.py
│
├── model.py
├── dataset.py
├── config.py
├── utils.py
│
├── check_dataset.py
├── check_model.py
│
├── classification_report.txt
├── evaluation_results.txt
├── predictions.csv
│
├── confusion_matrix.png
├── roc_curve.png
│
├── requirements.txt
└── .gitignore
Hardware and Training Environment

The model was developed using GPU acceleration during training.

The project was moved to Google Colab for CUDA-based training because of the computational requirements of training a Vision Transformer on a large image dataset.

The code also contains device selection logic so that the available hardware can be used where supported.

The general device priority is:

CUDA GPU
   |
   v
Apple MPS
   |
   v
CPU
Installation

Clone the repository:

git clone https://github.com/neeharlpu/deepfake-face-detection-vit.git
cd deepfake-face-detection-vit

Install the required Python packages:

pip install -r requirements.txt
Dataset Setup

The dataset is not included in the repository.

After obtaining the dataset, organize it according to the structure expected by the project:

dataset/
└── real_vs_fake/
    └── real-vs-fake/
        ├── train/
        ├── valid/
        └── test/

Make sure the class folders correspond to:

fake
real

The exact dataset path can be configured through config.py.

Training the Model

After setting up the dataset and configuration, training can be started with:

python train.py

The training script loads the dataset, initializes the Vision Transformer, trains the selected model parameters, evaluates the validation data, and saves checkpoints.

Evaluation

To evaluate the trained model:

python evaluate.py

The evaluation results can be used to inspect the classification performance of the model.

Running Predictions

To run predictions using the trained model:

python predict.py

The prediction script loads the trained model, processes the input image, and returns the predicted class.

Running the Application

The Streamlit application can be started using:

streamlit run app.py

The application provides a simple interface for interacting with the trained deepfake detection model.

Results

The repository includes the evaluation outputs generated during the project:

Classification report
Confusion matrix
ROC curve
Prediction results
Evaluation results

These files are included so that the model's evaluation can be inspected directly from the repository.

No performance value is stated here separately; the actual results should be taken from the evaluation files included in the repository.

Limitations

Deepfake detection is a challenging problem because image-generation and manipulation techniques continue to evolve.

A model trained on a particular dataset may not perform equally well on images produced using different generation or manipulation methods.

Therefore, the predictions produced by this project should be treated as model-based classifications rather than absolute proof that an image is real or fake.

Future Improvements

Possible improvements include:

Training with more diverse deepfake datasets
Fine-tuning more transformer layers
Improving generalization to unseen manipulation techniques
Adding more detailed explainability
Extending the system to video-based deepfake detection
Improving the application interface
Comparing ViT with other deep learning architectures
Project Goal

The goal of this project is to explore the use of Vision Transformers for detecting AI-generated and manipulated facial images.

The project covers the complete machine learning workflow:

Dataset
   |
   v
Preprocessing
   |
   v
Model Development
   |
   v
Transfer Learning
   |
   v
Training
   |
   v
Evaluation
   |
   v
Prediction
   |
   v
Application
Author

Neehar S

MSc Data Science

Lovely Professional University

GitHub: https://github.com/neeharlpu



