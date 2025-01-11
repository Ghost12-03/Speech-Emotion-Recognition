**Name** : Advait Shrikant Mulye\
**Company**: CODTECH IT SOLUTIONS\
**ID**: CT08DFP\
**Domain** : Artificial Intelligence\
**Duration** : 12 December 2024 to 12 January 2025\
**Mentor** : N. Santhosh

## **Overview of the Project**
## Project : Speech Emotion Recognition
## Introduction
Emotion recognition from speech is an essential component of creating more intuitive and empathetic human-computer interactions. By understanding the emotional state of users, systems can tailor responses to enhance user experience and engagement. Speech Emotion Recognition (SER) leverages computational techniques to analyze vocal expressions and identify underlying emotions. This project aims to develop an SER system that can reliably detect emotions from spoken language, contributing to advancements in areas such as virtual assistants, automated customer support, mental health diagnostics, and more. Through the integration of machine learning and signal processing, the project explores the capabilities and challenges of emotion detection in speech.

## Background
Emotion recognition encompasses various modalities, including facial expressions, physiological signals, and speech. Among these, speech-based emotion recognition offers unique advantages due to its non-intrusive nature and the richness of information conveyed through vocal characteristics. Previous research in SER has explored diverse feature extraction methods, such as prosodic features (pitch, energy, duration) and spectral features (MFCC, Linear Predictive Coding). Machine learning models, including Support Vector Machines (SVM), Hidden Markov Models (HMM), and more recently, deep learning architectures like Recurrent Neural Networks (RNN) and CNNs, have been employed to classify emotions with varying degrees of success. Despite significant progress, challenges remain in handling diverse datasets, managing variability in speech patterns, and achieving real-time performance.

## System Description
**The proposed SER system consists of the following components:**
- Data Collection: Utilizing publicly available datasets containing speech samples annotated with emotional labels.
- Preprocessing: Cleaning the audio data by removing noise, normalizing volume levels, and segmenting speech into manageable frames.
- Feature Extraction: Extracting pertinent features such as MFCC, pitch, energy, and chroma features that represent the emotional content of speech.
- Model Training: Training a classification model, specifically a CNN, on the extracted features to learn patterns associated with different emotions.
- Emotion Classification: Applying the trained model to new speech inputs to predict the corresponding emotional state.
- Evaluation: Assessing the system's performance using metrics like accuracy, precision, recall, and F1-score.

## Technology Used
- Programming Language: Python
- **Libraries and Frameworks:**
-- Librosa: For audio processing and feature extraction.
-- TensorFlow/Keras: For building and training the CNN model.
- Scikit-learn: For additional machine learning utilities and evaluation metrics.
- Development Environment: Jupyter Notebook for interactive development and experimentation.
- Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

## Discussion
The SER system demonstrated strong performance in recognizing happiness and neutral emotions, likely due to distinct vocal features associated with these states. However, emotions like fear and sadness showed lower precision and recall, indicating challenges in distinguishing these emotions from others with similar prosodic characteristics. The confusion matrix revealed that fear was often misclassified as anger, suggesting overlapping vocal patterns. The use of CNNs proved effective in capturing complex feature interactions, but incorporating additional features or using more advanced architectures like Recurrent Neural Networks (RNNs) could further enhance performance. Additionally, augmenting the dataset with more diverse samples could help the model generalize better across different speakers and contexts.

## Conclusion
This project successfully developed a Speech Emotion Recognition system utilizing deep learning techniques to classify emotions from speech with commendable accuracy. The integration of MFCC, prosodic, and chroma features provided a comprehensive representation of the emotional content in speech. The CNN-based model effectively learned to distinguish between different emotional states, highlighting the potential of machine learning in enhancing human-computer interactions. While the system performs well overall, there is room for improvement in recognizing certain emotions and handling more diverse datasets. The project underscores the viability of SER systems in various applications and sets the foundation for future advancements in this field.
