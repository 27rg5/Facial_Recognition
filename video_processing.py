import cv2
import dlib
import face_recognition
import os
import json
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging
import argparse

# Set up logging for informational messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NumpyEncoder(json.JSONEncoder):   # Custom JSON encoder to handle numpy data types

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.ndarray, np.bool_)):
            return obj.item() if isinstance(obj, (np.integer, np.floating, np.bool_)) else obj.tolist()
        return super().default(obj)

@dataclass
class TrackedFace:    # Data class to store information about a tracked face
  
    tracker: dlib.correlation_tracker
    track_id: int
    face_encoding: np.ndarray
    is_target: bool
    last_seen_frame: int
    last_known_bbox: tuple

class FaceTracker:      # Handles detection and tracking of faces in the video frames

    def __init__(self, ref_encoding, face_threshold=0.4, tracking_quality_threshold=8.0, max_frames_to_track=60):
        self.tracked_faces: List[TrackedFace] = []
        self.face_threshold = face_threshold
        self.tracking_quality_threshold = tracking_quality_threshold
        self.max_frames_to_track = max_frames_to_track
        self.ref_encoding = ref_encoding
        self.target_track_id = None
        self.next_track_id = 0

    def start_track(self, frame, face_location, face_encoding, current_frame):        # Starts tracking face using dlib's correlation tracker. Also determines if the face matches the target face.
     
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(face_location[3], face_location[0], face_location[1], face_location[2])
        tracker.start_track(frame, rect)
        
        face_distance = face_recognition.face_distance([self.ref_encoding], face_encoding)[0]
        is_target = face_distance < (self.face_threshold * 1.2)
        
        tracked_face = TrackedFace(tracker, self.next_track_id, face_encoding, is_target, current_frame, face_location)
        if is_target:
            self.target_track_id = self.next_track_id
        self.tracked_faces.append(tracked_face)
        self.next_track_id += 1

    def detect_faces(self, frame):          # Detect faces using face_recognition's CNN model

        return face_recognition.face_locations(frame, number_of_times_to_upsample=1, model='cnn')

    def update_trackers(self, frame, current_frame):        # Update the trackers for all currently tracked faces and remove those that do not meet the quality threshold or have been lost for too long
      
        updated_faces, positions = [], []
        for tf in self.tracked_faces:
            quality = tf.tracker.update(frame)
            frames_since_last_seen = current_frame - tf.last_seen_frame
            q_thresh = self.tracking_quality_threshold * (0.7 if tf.is_target else 1.0)
            
            # Keep tracking if quality is above threshold or if it's a target face still within the allowed re-track window
            if quality > q_thresh or (tf.is_target and frames_since_last_seen <= self.max_frames_to_track and quality > q_thresh * 0.5):
                pos = tf.tracker.get_position()
                bbox = (int(pos.top()), int(pos.right()), int(pos.bottom()), int(pos.left()))
                
                # Update last known good bbox if quality is good
                if quality > q_thresh:
                    tf.last_known_bbox = bbox
                    tf.last_seen_frame = current_frame
                else:
                    # Fall back to last known bbox if quality isn't great
                    bbox = tf.last_known_bbox
                
                updated_faces.append(tf)
                positions.append((bbox, tf.is_target, tf.track_id, quality))
        
        self.tracked_faces = updated_faces
        return positions

def process_video(              # Process the video to detect faces, track a target face, detect scene changes, and extract clips of the target face. Clips and metadata are saved to the specified output directory
    video_path: str, 
    reference_image_path: str, 
    output_dir: str,
    face_threshold: float = 0.4,
    redetection_interval: int = 30,
    min_clip_duration: float = 0.5,
    max_frames_to_track: int = 60
):
   
    # Load reference face and create tracker
    ref_image = face_recognition.load_image_file(reference_image_path)
    ref_encoding = face_recognition.face_encodings(ref_image, num_jitters=2)[0]
    face_tracker = FaceTracker(ref_encoding, face_threshold, max_frames_to_track=max_frames_to_track)

    # Prepare video capture
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    clip_index = 0
    current_clip_frames = []
    clips_metadata = []
    prev_frame = None

    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Check for scene change and finalize previous clip if minimum duration is met
        if prev_frame is not None and detect_scene_change(prev_frame, frame) and current_clip_frames:
            if len(current_clip_frames) > fps * min_clip_duration:
                save_clip(current_clip_frames, output_dir, clip_index, fps)
                metadata = create_clip_metadata(current_clip_frames, clip_index)
                save_metadata(metadata, output_dir, clip_index)
                clips_metadata.append(metadata)
                clip_index += 1
            current_clip_frames = []

        # Periodically re-detect faces to handle drift or lost faces
        if frame_count == 1 or frame_count % redetection_interval == 0:
            face_locations = face_tracker.detect_faces(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=2)
            for loc, enc in zip(face_locations, face_encodings):
                face_tracker.start_track(rgb_frame, loc, enc, frame_count)

        # Update trackers and record target frames
        tracked_positions = face_tracker.update_trackers(rgb_frame, frame_count)
        frame_data = [{"bbox": bbox, "is_target": is_target} for bbox, is_target, _, _ in tracked_positions]
        for bbox, is_target, _, _ in tracked_positions:
            if is_target:
                # Store frame details only if the target face is present
                current_clip_frames.append((frame_count / fps, bbox, frame.copy(), frame_data))

        prev_frame = frame.copy()

    # Finalize any remaining clip at the end of the video if it meets the duration criteria
    if len(current_clip_frames) > fps * min_clip_duration:
        save_clip(current_clip_frames, output_dir, clip_index, fps)
        metadata = create_clip_metadata(current_clip_frames, clip_index)
        save_metadata(metadata, output_dir, clip_index)
        clips_metadata.append(metadata)
        clip_index += 1

    video.release()
    return clips_metadata

def detect_scene_change(prev_frame: np.ndarray, curr_frame: np.ndarray, threshold: float = 30.0) -> bool:
    """
    Detect a scene change by comparing the average difference between the current and previous frame.
    If the difference is greater than the threshold, assume a scene change.
    """
    return np.mean(cv2.absdiff(prev_frame, curr_frame)) > threshold

def create_clip_metadata(frames_data: List[Tuple], clip_index: int) -> Dict:        # Create metadata for each single clip
    
    return {
        "file_name": f"clip_{clip_index}.mp4",
        "start_time": frames_data[0][0],
        "end_time": frames_data[-1][0],
        "frame_count": len(frames_data),
        "face_data": [
            {
                "timestamp": float(t),
                "target_bbox": list(bbox),
                "other_faces": [
                    {
                        "bbox": list(face["bbox"]),
                        "is_target": bool(face["is_target"])
                    } for face in all_faces if not face["is_target"]
                ]
            } 
            for t, bbox, _, all_faces in frames_data
        ]
    }

def save_metadata(metadata: Dict, output_dir: str, clip_index: int):        # Save the metadata for a clip to a JSON file
    
    metadata_file = os.path.join(output_dir, f"clip_{clip_index}_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4, cls=NumpyEncoder)

def save_clip(frames_data: List[Tuple], output_dir: str, clip_index: int, fps: int):        # Save the extracted clip to disk, centering on the target face and applying smoothing for better transition from one frame to another in the cropped clips
    
    if not frames_data:
        return
    output_file = os.path.join(output_dir, f"clip_{clip_index}.mp4")
    
    padding_factor = 0.2
    bboxes = [bbox for _, bbox, _, _ in frames_data]
    smoothed_bboxes = smooth_bbox_sequence(bboxes)

    # Determine maximum dimensions for the crop
    max_width = 0
    max_height = 0
    for bbox in smoothed_bboxes:
        top, right, bottom, left = bbox
        height = bottom - top
        width = right - left
        max_width = max(max_width, width)
        max_height = max(max_height, height)

    final_width = int(max_width * (1 + padding_factor))
    final_height = int(max_height * (1 + padding_factor))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_file, fourcc, fps, (final_width, final_height))
    if not out_writer.isOpened():
        raise IOError(f"Failed to create video writer for clip {clip_index}")

    # Write each frame after cropping and smoothing
    for (_, _, frame, _), bbox in zip(frames_data, smoothed_bboxes):
        top, right, bottom, left = bbox
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        
        crop_top = max(0, center_y - final_height // 2)
        crop_bottom = min(frame.shape[0], crop_top + final_height)
        crop_left = max(0, center_x - final_width // 2)
        crop_right = min(frame.shape[1], crop_left + final_width)
        
        cropped_frame = frame[crop_top:crop_bottom, crop_left:crop_right]
        if cropped_frame.size > 0:
            if cropped_frame.shape[:2] != (final_height, final_width):
                cropped_frame = cv2.resize(cropped_frame, (final_width, final_height))
            # Apply a slight Gaussian blur for smoother visuals
            cropped_frame = cv2.GaussianBlur(cropped_frame, (3, 3), 0)
            out_writer.write(cropped_frame)
    
    out_writer.release()

def smooth_bbox_sequence(bboxes: List[tuple], kernel_size: int = 5) -> List[tuple]:         # Apply Gaussian smoothing to a sequence of bounding boxes to reduce jitter
  
    if len(bboxes) < kernel_size:
        return bboxes

    bbox_array = np.array(bboxes)
    smoothed = np.zeros_like(bbox_array, dtype=float)
    for i in range(4):
        smoothed[:, i] = cv2.GaussianBlur(bbox_array[:, i].reshape(-1, 1),
                                         (kernel_size, 1), 0).reshape(-1)

    return [tuple(map(int, bbox)) for bbox in smoothed]

def main():

    parser = argparse.ArgumentParser(description="Process a video to extract clips of a target face.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--reference_image_path", type=str, required=True, help="Path to the reference image containing the target face.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output clips and metadata.")

    args = parser.parse_args()

    try:
        process_video(
            video_path=args.video_path,
            reference_image_path=args.reference_image_path,
            output_dir=args.output_dir,
            face_threshold=0.6,
            redetection_interval=30,
            min_clip_duration=0.5,
            max_frames_to_track=60
        )
        logging.info("Processing complete.")
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        raise

if __name__ == "__main__":
    main()