# TNSDC-GENERATIVE-AI-NAAN-MUDHALVAN

## Project Title: 
Image Caption Generator using CNN and LSTM on Flickr8K Dataset

## Introduction:
This project implements an image caption generator using Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. The model is trained on the Flickr8K dataset, which contains images along with corresponding descriptions. The goal is to develop a system that can generate accurate and relevant captions for input images.

## Features:
- Utilizes a combination of CNNs and LSTMs: The model leverages a CNN for image feature extraction and an LSTM for generating captions.
- Data Cleaning: Text descriptions are preprocessed by removing punctuation, converting to lowercase, and filtering out non-alphabetic characters.
- Vocabulary Generation: The vocabulary is generated from the cleaned descriptions, and each word is tokenized and mapped to integers.
- GloVe Embeddings: Pre-trained GloVe word embeddings are used to represent words in the vocabulary, enhancing the model's understanding of semantics.
- Training the Model: The model is trained on the training dataset, consisting of images and their corresponding captions, to learn the relationship between visual features and textual descriptions.
- Prediction: Once trained, the model can generate captions for new images by employing a greedy search algorithm.

## Requirements:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV

## Usage:
1. Clone the repository.
2. Download the Flickr8K dataset.
3. Run the provided Jupyter Notebook to preprocess the data, train the model, and generate captions.

## File Structure:
- README.md: Instructions and overview of the project.
- image_caption_generator.ipynb: Python script containing the implementation of the image caption generator.
- Flickr8K dataset: Dataset containing images and their corresponding descriptions.

## References:
- Flickr8K dataset: [Link](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- GloVe word embeddings: [Link](https://www.kaggle.com/datasets/rtatman/glove-global-vectors-for-word-representation)

## Note:
- The dataset used in this project is for research purposes only.
- Make sure to review and adhere to the licensing terms and conditions of the Flickr8K dataset.
