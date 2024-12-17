# Facial_Recognition

A Python-based system for detecting, tracking, and extracting video clips of a target face from video footage. 

## Prerequisites

- Python 3.7+
- OpenCV
- dlib
- face_recognition
- numpy

## Installation

1. Clone the repository
2. Create a virtual environment (optional but recommended)
3. Install dependencies (requirements.txt)

## Usage

Run the script using the following command:

<code>python video_processing.py --video_path /path/to/video.mp4 --reference_image_path /path/to/reference.jpg --output_dir /path/to/output</code>

## Limitations

- Requires good quality video input for optimal performance
- Requires GPU 
- Maximum tracking duration of 60 frames without redetection
- Code works with MP4 videos and JPG/JPEG image files
