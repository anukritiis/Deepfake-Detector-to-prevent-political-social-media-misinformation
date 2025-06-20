Deepfake Detection for Political Misinformation
A machine learning application that detects deepfake images of politicians on social media with 97.8% accuracy, using a custom CNN and a multi-layered image processing pipeline. This project addresses the challenge of identifying manipulated images that have been compressed and stripped of metadata by social media platforms.

The Problem
The rise of generative AI has made it easy to create convincing deepfakes of politicians, posing a serious threat to democratic integrity by spreading misinformation. Most existing deepfake detectors perform poorly on the low-quality, highly compressed images found on social media platforms like Twitter and Facebook, as these platforms remove the metadata that forensic tools rely on. This project was created to fill that gap with a robust tool tailored for this specific, challenging environment.

Key Features & Results
High-Accuracy Detection: Achieved 97.8% accuracy on a custom-curated dataset of real and deepfake political images, specifically designed to test performance on compressed, "in-the-wild" examples

Advanced Image Processing Pipeline: Developed and implemented a unique multi-layered image processing pipeline using Laplacian Filtering, Embossing, and Fast Fourier Transform (FFT) in OpenCV to expose subtle manipulation artefacts that are invisible to the naked eye.

Custom CNN Architecture: Built and trained a custom Convolutional Neural Network (CNN) from scratch using TensorFlow. This bespoke model outperformed a fine-tuned ResNet50 model for this specific detection task

Educational Impact: Created an accompanying educational web application with an interactive quiz that improved users' manual detection ability by an average of 131.58%, demonstrating a tangible impact on digital literacy

System Architecture & Methodology
The core of the detection method is a multi-layered image processing pipeline designed to enhance manipulation artefacts before they are passed to the neural network for classification.

The workflow is as follows:

Image Preprocessing: The input image is resized to 224x224 pixels and converted to grayscale

Edge & Texture Enhancement: Laplacian and Emboss filters highlight edges, textures, and depth inconsistencies.

Frequency Domain Analysis: A Fast Fourier Transform (FFT) analyses frequency patterns. Real photos contain high-frequency noise that is often absent in overly smooth, AI-generated images

Classification: The processed data is then passed to the trained CNN for final classification as "Real" or "Fake"

Technology used - 
Backend & Machine Learning: Python, TensorFlow, Keras, OpenCV, NumPy, Matplotlib

GUI Application: Tkinter

Web Application: Flask, HTML/CSS, JavaScript 

Installation & Usage
To run this project locally, please follow these steps:

Clone the repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/anukritiis/Deepfake-Detector-to-prevent-political-social-media-misinformation.git)

Navigate to the project directory:

cd your-repo-name

Run the GUI application:

python gui.py
