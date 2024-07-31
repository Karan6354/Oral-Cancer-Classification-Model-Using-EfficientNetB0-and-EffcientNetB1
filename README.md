# Oral Cancer Classification Model Using EfficientNetB0 and EfficientNetB1 with Self-Attention Mechanism

## Overview

The oral cancer classification model employs cutting-edge deep learning techniques to precisely identify and categorize various oral diseases. It utilizes EfficientNetB0 and EfficientNetB1, two highly effective convolutional neural network (CNN) architectures, for feature extraction. To enhance the model’s performance further, a self-attention block is integrated to refine feature fusion by emphasizing the most relevant features in the input data. This approach enables the model to capture and interpret the underlying patterns associated with oral cancer more accurately.

## Dataset

The model is trained and evaluated using the MOD (Mouth and Oral Disease) dataset. This dataset includes a range of oral disease categories, with images organized into folders named OC (Oral Cancer), CaS (Cancerous Sores), OT (Other Tumors), CoS (Cold Sores), Gum (Gum Diseases), MC (Mucosal Conditions), and OLP (Oral Lichen Planus). Each folder contains images for training, validation, and testing.

## Model Architecture

1. **Feature Extraction**:
   - **EfficientNetB0 and EfficientNetB1**: These architectures are known for their efficient performance in terms of accuracy and computational cost. Both models are used separately to extract features from the input images. EfficientNetB0 and EfficientNetB1 are designed to capture a range of features from simple textures to complex patterns, which is essential for distinguishing between different oral diseases.

2. **Feature Fusion**:
   - The features extracted by EfficientNetB0 and EfficientNetB1 are combined at a later stage. This fusion process allows the model to leverage the strengths of both architectures, improving the overall feature representation and enhancing the classification accuracy.

3. **Self-Attention Mechanism**:
   - **Self-Attention Layer**: The self-attention block enhances the feature fusion by focusing on the most relevant features in the input data, allowing the model to better capture the underlying patterns associated with oral cancer in the images. This approach is particularly useful in complex tasks where certain features are more indicative of the target outcome than others.

4. **Classification Head**:
   - The fused and attention-enhanced features are fed into a dense layer with ReLU activation, followed by a final dense layer with softmax activation. This setup provides the probabilities for each disease category, allowing for accurate classification.

## Implementation in Python

The model is implemented using TensorFlow and Keras libraries in Python. Key steps include:

- **Model Building**: EfficientNetB0 and EfficientNetB1 are loaded with pre-trained weights and used to extract features.
- **Feature Fusion**: Features from both models are concatenated or merged using a fusion layer.
- **Self-Attention Layer**: Implemented to refine feature representation.
- **Training**: The model is trained using the MOD dataset with appropriate loss functions and optimizers.
- **Evaluation**: The performance is assessed on the validation and test sets, with metrics such as accuracy, precision, recall, and F1 score computed.

## Conclusion

By combining EfficientNetB0 and EfficientNetB1 for feature extraction and incorporating a self-attention mechanism, the oral cancer classification model achieves a robust and accurate classification of various oral diseases. This approach leverages advanced deep learning techniques to address the complexities of oral disease identification, offering improved diagnostic capabilities and supporting better clinical decision-making.

### Average Training Metrics

- **Accuracy**: 99.94169
- **Precision**: 99.94191
- **Recall**: 99.94169
- **F1 Score**: 99.94170

### Average Testing Metrics

- **Accuracy**: 98.34
- **Precision**: 98.35
- **Recall**: 98.34
- **F1 Score**: 98.33

### Average Training and Testing Time
- **Average Training Time**: 294.10 seconds
- **Average Testing Time**: 9.90 seconds

The impressive training metrics reflect the model's high performance and reliability during training. The testing metrics demonstrate strong generalization capabilities, with accuracy and other metrics remaining high even on unseen data. The average testing time indicates efficient processing, making the model suitable for practical applications in clinical settings.

