import csv
import cv2
import itertools
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import tqdm

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download model from TF Hub and check out inference code from GitHub
pose_sample_rpi_path = os.path.join(os.getcwd(), '../examples/lite/examples/pose_estimation/raspberry_pi')
sys.path.append(pose_sample_rpi_path)

# Load MoveNet Thunder model
import utils
from data import BodyPart
from ml import Movenet
movenet = Movenet('../movenet_thunder')

# Define function to run pose estimation using MoveNet Thunder.
# You'll apply MoveNet's cropping algorithm and run inference multiple times on
# the input image to improve pose estimation accuracy.
def detect(input_tensor, inference_count=3):
  """Runs detection on an input image.
 
  Args:
    input_tensor: A [height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.
    inference_count: Number of times the model should run repeatly on the
      same input image to improve detection accuracy.
 
  Returns:
    A Person entity detected by the MoveNet.SinglePose.
  """
  image_height, image_width, channel = input_tensor.shape
 
  # Detect pose using the full input image
  movenet.detect(input_tensor.numpy(), reset_crop_region=True)
 
  # Repeatedly using previous detection result to identify the region of
  # interest and only croping that region to improve detection accuracy
  for _ in range(inference_count - 1):
    person = movenet.detect(input_tensor.numpy(), 
                            reset_crop_region=False)

  return person

#@title Code to load the images, detect pose landmarks and save them into a CSV file

class MoveNetPreprocessor(object):
  """Helper class to preprocess pose sample images for classification."""
 #VINCENT: i added a new name for where the labels are 
  def __init__(self,
               images_in_folder,
               csvs_out_path):
    """Creates a preprocessor to detection pose from images and save as CSV.

    Args:
      images_in_folder: Path to the folder with the input images. It should
        follow this structure:
        yoga_poses
        |__ downdog
            |______ 00000128.jpg
            |______ 00000181.bmp
            |______ ...
        |__ goddess
            |______ 00000243.jpg
        ...
      csvs_out_path: Path to write the CSV containing the detected landmark
        coordinates and label of each image that can be used to train a pose
        classification model.
    """
    self._images_in_folder = images_in_folder
    self._csvs_out_path = csvs_out_path
    self._messages = []

    # Create a temp dir to store the pose CSVs per class
    # self._csvs_out_folder_per_class = tempfile.mkdtemp()
    self._csvs_out_folder_per_class = '/tmp/tmpthpnu1hi'
 
    # Get list of pose classes and print image statistics
    self._pose_class_names = sorted(
        [n for n in os.listdir(self._images_in_folder) if not n.startswith('.')]
        )
    # did not fully process 2123
    
  def process(self, per_pose_class_limit=None, detection_threshold=0.1):
    """Preprocesses images in the given folder.
    Args:
      per_pose_class_limit: Number of images to load. As preprocessing usually
        takes time, this parameter can be specified to make the reduce of the
        dataset for testing.
      detection_threshold: Only keep images with all landmark confidence score
        above this threshold.
    """
    # Loop through the classes and preprocess its images
    for pose_class_name in self._pose_class_names:
      print('Preprocessing', pose_class_name, file=sys.stderr)
      
      # Paths for the pose class.
      images_in_folder = os.path.join(self._images_in_folder, pose_class_name, 'rgb')
      csv_out_path = os.path.join(self._csvs_out_folder_per_class,
                                  pose_class_name + '.csv')
      labels_path = os.path.join(self._images_in_folder, pose_class_name, 'labels.csv')
      labels_out_path = os.path.join(self._csvs_out_folder_per_class,
                                  pose_class_name + '_labels.csv')
      
      if os.path.exists(csv_out_path): continue
 
      # Detect landmarks in each image and write it to a CSV file
      with open(csv_out_path, 'w') as csv_out_file:
        # with open(labels_path, 'r') as labels_file:
          # with open(labels_out_path, 'w') as labels_out_file:
            csv_out_writer = csv.writer(csv_out_file, 
                                        delimiter=',', 
                                        quoting=csv.QUOTE_MINIMAL)
            # csv_reader = list(csv.reader(labels_file))
            # labels_out_writer = csv.writer(labels_out_file, 
                                          #  delimiter=',', 
                                          #  quoting=csv.QUOTE_MINIMAL)
            # Get list of images
            image_names = sorted(
                [n for n in os.listdir(images_in_folder) if not n.startswith('.')])
            if per_pose_class_limit is not None:
              image_names = image_names[:per_pose_class_limit]

            valid_image_count = 0

            # Detect pose landmarks from each image
            for idx, image_name in enumerate(tqdm.tqdm(image_names)):
              image_path = os.path.join(images_in_folder, image_name)

              try:
                image = tf.io.read_file(image_path)
                image = tf.io.decode_image(image)
              except:
                self._messages.append('Skipped ' + image_path + '. Invalid image.')
                continue
              else:
                image = tf.io.read_file(image_path)
                image = tf.io.decode_image(image)
                image_height, image_width, channel = image.shape

              # Skip images that isn't RGB because Movenet requires RGB images
              if channel != 3:
                self._messages.append('Skipped ' + image_path +
                                      '. Image isn\'t in RGB format.')
                continue
              person = detect(image)

              # Save landmarks if all landmarks were detected
              min_landmark_score = min(
                  [keypoint.score for keypoint in person.keypoints])
              should_keep_image = min_landmark_score >= detection_threshold
              if not should_keep_image:
                self._messages.append('Skipped ' + image_path +
                                      '. No pose was confidentlly detected.')
                continue

              valid_image_count += 1

              # Get landmarks and scale it to the same size as the input image
              pose_landmarks = np.array(
                  [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                    for keypoint in person.keypoints],
                  dtype=np.float32)

              # Write the landmark coordinates to its per-class CSV file
              coordinates = pose_landmarks.flatten().astype(np.str).tolist()
              csv_out_writer.writerow([image_name] + coordinates)
              # labels_out_writer.writerow([image_name] + [csv_reader[idx][1]])
            if not valid_image_count:
              raise RuntimeError(
                  'No valid images found for the "{}" class.'
                  .format(pose_class_name))
      
    # Print the error message collected during preprocessing.
    print('\n'.join(self._messages))

    # Combine all per-class CSVs into a single output file
    all_landmarks_df = self._all_landmarks_as_dataframe()
    all_landmarks_df.to_csv(self._csvs_out_path, index=False)

  def class_names(self):
    """List of classes found in the training dataset."""
    return self._pose_class_names
  
  def _all_landmarks_as_dataframe(self):
    """Merge all per-class CSVs into a single dataframe."""
    total_df = None
    # total_labels = None
    for class_index, class_name in enumerate(self._pose_class_names):
      csv_out_path = os.path.join(self._csvs_out_folder_per_class,
                                  class_name + '.csv')
      # labels_out_path = os.path.join(self._csvs_out_folder_per_class,
      #                             class_name + '_labels.csv')
      per_class_df = pd.read_csv(csv_out_path, header=None)
      # per_class_labels = pd.read_csv(labels_out_path, header=None)
      
      # Add the labels
      per_class_df['class_no'] = [class_index]*len(per_class_df)
      per_class_df['class_name'] = [class_name]*len(per_class_df)
      
      # Append the folder name to the filename column (first column)
      per_class_df[per_class_df.columns[0]] = (os.path.join(class_name, '') 
        + per_class_df[per_class_df.columns[0]].astype(str))

      if total_df is None:
        # For the first class, assign its data to the total dataframe
        total_df = per_class_df
        # total_labels = per_class_labels
      else:
        # Concatenate each class's data into the total dataframe
        total_df = pd.concat([total_df, per_class_df], axis=0)
        # per_class_labels = pd.concat([total_labels, per_class_labels], axis=0)
 
    list_name = [[bodypart.name + '_x', bodypart.name + '_y', 
                  bodypart.name + '_score'] for bodypart in BodyPart] 
    header_name = []
    for columns_name in list_name:
      header_name += columns_name
    header_name = ['file_name'] + header_name
    header_map = {total_df.columns[i]: header_name[i] 
                  for i in range(len(header_name))}
 
    total_df.rename(header_map, axis=1, inplace=True)

    return total_df

#@markdown Be sure you run this cell. It's hiding the `split_into_train_test()` function that's called in the next code block.

import os

IMAGES_ROOT = '../data'
# You can leave the rest alone:
if not os.path.isdir(IMAGES_ROOT):
  raise Exception(IMAGES_ROOT, "is not a valid directory")

images_in_train_folder = os.path.join(IMAGES_ROOT)
csvs_out_train_path = 'data.csv'
preprocessor = MoveNetPreprocessor(
    images_in_folder=images_in_train_folder,
    csvs_out_path=csvs_out_train_path,
)
preprocessor.process(per_pose_class_limit=None)
