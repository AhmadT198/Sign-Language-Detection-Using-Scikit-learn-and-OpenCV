# Sign-Language-Detection-Using-Scikit-learn-and-OpenCV

This repository contains scripts for collecting hand gesture image data, creating datasets, training a classifier, and performing real-time inference for hand gesture recognition. The system leverages OpenCV, MediaPipe, and scikit-learn libraries to facilitate the entire pipeline from data collection to inference.

## Files and Descriptions

- ``collect_imgs.py``: This Python script utilizes OpenCV and the operating system module to facilitate the collection of image data for multiple classes. Upon execution, it first prepares a directory structure to store the captured images. The user specifies the number of classes and the desired dataset size. It then initializes video capture from the default camera and prompts the user to prepare for data collection. Once the user presses 'Q', the script enters a loop to capture images continuously until the specified dataset size is reached for each class. Each captured frame is saved as an image file in the corresponding class directory. After completing data collection for all classes, it releases the video capture device and closes any open windows.

- ``create_dataset.py``: This Python script utilizes MediaPipe and OpenCV to extract hand landmarks from images, storing the data and corresponding labels in a pickle file named "data.pickle". It iterates through images in specified directories, processes them to detect hand landmarks, normalizes the coordinates, and saves them with associated labels. This streamlined approach is valuable for tasks like hand gesture recognition, simplifying data collection and preprocessing.

- ``train_classifier.py``: This Python script employs scikit-learn to train a **Random Forest classifier** for a given dataset loaded from a pickle file, evaluating its performance in a concise manner. Initially, the data and labels are deserialized from "data.pickle". Subsequently, the data is preprocessed, with numpy arrays utilized for efficient manipulation. Through the use of scikit-learn's train_test_split function, the dataset is divided into training and testing sets, ensuring a balanced representation of classes. The script then proceeds to instantiate, train, and evaluate the Random Forest classifier, calculating its accuracy score. Finally, the accuracy percentage is printed, providing a clear indication of the classifier's performance on the test data. Additionally, the trained model is serialized into a pickle file named "model.p" for potential future utilization.

- ``inference_classifier.py``: This Python script integrates a pre-trained machine learning model with live video feed and the MediaPipe library to enable real-time hand gesture recognition. Initially, the model is loaded from a pickle file. Subsequently, the script initializes video capture from the default camera and sets up MediaPipe's hand tracking functionality. In a continuous loop, frames from the video feed are processed to detect hand landmarks, which are then used to predict gestures using the loaded model. Predicted gestures are overlaid onto the video frames alongside bounding boxes around detected hands. This setup enables the script to provide live feedback on recognized hand gestures. Finally, upon termination, the script releases the video capture device and closes all OpenCV windows. This implementation showcases the seamless integration of machine learning models and computer vision techniques for real-time gesture recognition applications.


## Running Locally

To run the system locally, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/AhmadT198/Sign-Language-Detection-Using-Scikit-learn-and-OpenCV.git
   ```

2. Navigate to the repository directory:

   ```
   cd Sign-Language-Detection-Using-Scikit-learn-and-OpenCV
   ```

3. Install the required dependencies using pip:

   ```
   pip install -r requirements.txt
   ```

4. Execute the scripts in the following order:

   - `python collect_imgs.py` (for collecting image data)
   - `python create_dataset.py` (for creating the dataset)
   - `python train_classifier.py` (for training the classifier)
   - `python inference_classifier.py` (for performing real-time inference)

Ensure that you have a camera connected to your system for data collection and real-time inference. Additionally, adjust any parameters or paths in the scripts as needed to suit your setup.

## Contribution

Contributions to this project are welcome. If you have suggestions for improvements, bug fixes, or new features, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License, which means you are free to use, modify, and distribute the code for both commercial and non-commercial purposes. However, attribution is appreciated.

## Acknowledgments

- Thanks to the developers of OpenCV, MediaPipe, and scikit-learn for their excellent libraries.
- Special thanks to the open-source community for their valuable contributions and feedback.
