# Live-Emotion-Color-Detector

The Live-Emotion-Color-Detector is an application designed to detect human emotions in real-time using a webcam feed. Utilizing the robust RMN model trained on the FER-2013 dataset, this application classifies facial expressions into distinct emotions and highlights them with unique color codes. This application serves as a uselful tool for augmenting interactive experiences and engaging users with intuitive emotion recognition.


## Features

- **Real-time Emotion Detection**: Quickly identifies and classifies emotions from facial expressions as they occur.
- **Color-Coded Feedback**: Assigns unique colors to different emotions for intuitive and immediate understanding.
- **Flexible Application**: Ideal for diverse contexts including interactive installations, educational tools, and enhancing customer service experiences with intuitive emotion recognition.

## Launch Instructions

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or later
- OpenCV
- TensorFlow

### Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/Op27/Live-Emotion-Color-Detector.git
   
2. Install the required dependencies:
   ```bash
    pip install -r requirements.txt

3. Running the Application
To start the application, navigate to the project directory and run:
    ```bash
    python app.py

## Functional Overview

The Live-Emotion-Color-Detector leverages the power of the Residual Masking Network (RMN), a state-of-the-art facial expression recognition model that has shown remarkable accuracy in emotion detection tasks. The RMN model, which is the centerpiece of this application, performed the best on the FER2013 dataset with an impressive accuracy of 76.82% as of 21 February 2024, making it the leading choice for emotion detection.

### About RMN 

Facial Expression Recognition using RMN involves analyzing facial expressions from video feed in real-time to classify them into distinct emotions. This model stands out due to its unique approach to handling the spatial hierarchies between facial parts to recognize emotions accurately. For more technical insights, visit the [RMN GitHub page](https://github.com/phamquiluan/ResidualMaskingNetwork).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.








