import numpy as np
import cv2
import os
import glob

def average(data, window_size):
    filter = np.ones(window_size) / window_size
    padded_data = np.pad(data, (window_size//2, window_size//2), 'edge')
    smoothed_data = np.convolve(padded_data, filter, mode='same')
    smoothed_data = smoothed_data[window_size//2:-window_size//2]
    return smoothed_data 

def smoothMotion(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = average(trajectory[:, i], window_size=50)
    return smoothed_trajectory

def fixImageBorders(image):
    height, width = image.shape[:2]
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), 0, 1.04)
    fixed_image = cv2.warpAffine(image, transform, (width, height))
    return fixed_image

def process_video(input_path, output_path):
    print(f"Stabilizing {input_path}")
    # Open the input video file
    video = cv2.VideoCapture(input_path)
    
    # Get the video properties
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Prepare the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Read the first frame
    success, previous_frame = video.read()
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((num_frames-1, 3), np.float32)

    # Process each frame in the video
    for i in range(num_frames - 2):
        # Detect feature points to track
        feature_points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        success, current_frame = video.read()
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Track feature points using optical flow
        tracked_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, feature_points, None)
        valid_indices = np.where(status == 1)[0]
        feature_points = feature_points[valid_indices]
        tracked_points = tracked_points[valid_indices]

        if len(feature_points) < 4 or len(tracked_points) < 4:
            print("Not enough points to calculate transformation.")
            transforms[i] = transforms[i - 1]
        else:
            # Estimate the affine transformation matrix
            transform_matrix, _ = cv2.estimateAffinePartial2D(feature_points, tracked_points)
            translation_x = transform_matrix[0, 2]
            translation_y = transform_matrix[1, 2]
            rotation_angle = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
            transforms[i] = [translation_x, translation_y, rotation_angle]

        previous_gray = current_gray

    # Calculate the trajectory and smooth it
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smoothMotion(trajectory)
    difference = smoothed_trajectory - trajectory
    smoothed_transforms = transforms + difference

    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Process and save each frame of the stabilized video
    for i in range(num_frames - 2):
        success, frame = video.read()
        if not success:
            print(f"Cannot read the video: {input_path}")
            break
        
        translation_x = smoothed_transforms[i, 0]
        translation_y = smoothed_transforms[i, 1]
        rotation_angle = smoothed_transforms[i, 2]

        # Build the transformation matrix
        transformation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle), translation_x],
            [np.sin(rotation_angle), np.cos(rotation_angle), translation_y]
        ], dtype=np.float32)

        # Apply the transformation to stabilize the frame
        stabilized_frame = cv2.warpAffine(frame, transformation_matrix, (frame_width, frame_height))
        stabilized_frame = fixImageBorders(stabilized_frame)
        video_writer.write(stabilized_frame)
        
        cv2.imshow('Difference Video', np.hstack((frame, stabilized_frame)))
        cv2.waitKey(int(1000 / fps))

    video.release()
    video_writer.release()
    print(f"Stabilized video has been created successfully: {output_path}")

# Folder paths
input_folder = 'input_videos/'
output_folder = 'output_videos/'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of video files
video_files = glob.glob(os.path.join(input_folder, '*.mp4'))

# Process each video file
for video_file in video_files:
    output_file = os.path.join(output_folder, 'stabilized_' + os.path.basename(video_file))
    process_video(video_file, output_file)
