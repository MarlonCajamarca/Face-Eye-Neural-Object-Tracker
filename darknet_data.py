"""
Tool for generating a train and/or test folders with labeled object samples for darknet-based detectors.
"""
import os
import sys
import argparse
import shutil
import json
import re
from tqdm import tqdm

_IMAGE_EXTENSIONS = [".jpg", ".png"]
_ANNOTATION_EXTENSION = ".txt"
_VALIDATION_FOLDER_NAME = "image_samples"

_DATA_FOLDER_NAME = "data"
_OBJECTS_FOLDER_NAME = "obj"
_NAMES_FILE_NAME = "obj.names"
_DATA_FILE_NAME = "obj.data"
_TRAIN_FILE_NAME = "train.txt"
_VALID_FILE_NAME = "valid.txt"

_valid_data_porcentage = 0.10


class DetectorCustomDataset(object):
    def __init__(self, parent_folder: str, output_folder: str, configuration_file: str) -> None:
        self.parent_folder = parent_folder
        self.output_folder = output_folder
        with open(configuration_file) as config_file:
            self.config = json.load(config_file)
        self.subfolders_path_list = [file.path for file in os.scandir(parent_folder) if file.is_dir()]
        self.file_path_list = list()
        self.samples_folder_path_list = list()
        self.raw_image_id = str()
        self.sample_image_id = str()
        self.sample_annotation_id = str()
        self.images_filepath_list = list()
        self.annotations_filepath_list = list()
        self.batch_input_filepaths = list()
        self.samples_counter = 0
        self.train_txt_path = str()
        self.valid_txt_path = str()
        self.output_data_folder = str()
        self.objects_data_folder = str()
        self.train_txt_file = None
        self.valid_txt_file = None

    def run(self) -> None:
        print('Darknet Detection Dataset generator tool LAUNCHED successfully!')
        self.create_training_file()
        self.get_valid_samples_folders()
        self.samples_folder_path_list.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.process_sample_subfolders()
        print('Darknet Detection Dataset generator tool finished successfully!')

    def process_sample_subfolders(self) -> None:
        for samples_folder_path in self.samples_folder_path_list:
            print(f'Current sampling folder : {samples_folder_path}')
            self.file_path_list = [file.path for file in os.scandir(samples_folder_path) if file.is_file()]
            self.images_filepath_list = [file_path for file_path in self.file_path_list if os.path.splitext(file_path)[1] in _IMAGE_EXTENSIONS]
            self.images_filepath_list.sort(key=lambda f: int(re.sub('\D', '', f)))
            self.annotations_filepath_list = [file_path for file_path in self.file_path_list if os.path.splitext(file_path)[1] == _ANNOTATION_EXTENSION]
            self.annotations_filepath_list.sort(key=lambda f: int(re.sub('\D', '', f)))
            if len(self.images_filepath_list) != len(self.annotations_filepath_list):
                print(f'Number of image files : {len(self.images_filepath_list)}')
                print(f'Number of label files : {len(self.annotations_filepath_list)}')
                if len(self.images_filepath_list) > len(self.annotations_filepath_list):
                    self.find_missing_files(self.images_filepath_list)
                else:
                    self.find_missing_files(self.annotations_filepath_list)
                sys.exit()
            assert(len(self.images_filepath_list) == len(self.annotations_filepath_list))
            valid_sample_save_idx = len(self.images_filepath_list) * _valid_data_porcentage
            valid_sample_save_idx = round(len(self.images_filepath_list) / valid_sample_save_idx)
            progress_bar = tqdm(total=len(self.images_filepath_list))
            with open(self.train_txt_path, "a+") as self.train_txt_file:
                with open(self.valid_txt_path, "a+") as self.valid_txt_file:
                    train_sample_counter = 0
                    valid_sample_counter = 0
                    for image_path, label_path in zip(self.images_filepath_list, self.annotations_filepath_list):
                        self.sample_image_id = str(self.samples_counter) + str(os.path.splitext(image_path)[1])
                        image_relative_path = os.path.join(_DATA_FOLDER_NAME, _OBJECTS_FOLDER_NAME, self.sample_image_id)
                        self.sample_annotation_id = str(self.samples_counter) + _ANNOTATION_EXTENSION
                        shutil.copy2(image_path, os.path.join(self.objects_data_folder, self.sample_image_id))
                        shutil.copy2(label_path, os.path.join(self.objects_data_folder, self.sample_annotation_id))
                        if self.samples_counter % valid_sample_save_idx == 0:
                            self.valid_txt_file.write(f'{image_relative_path}\n')
                            valid_sample_counter += 1
                        else:
                            self.train_txt_file.write(f'{image_relative_path}\n')
                            train_sample_counter += 1
                        self.samples_counter += 1
                        progress_bar.update(1)
            progress_bar.close()
        print(f'Total number of samples : {str(self.samples_counter)}')
        print(f'Total number of samples to train : {str(train_sample_counter)}')
        print(f'Total number of samples to validate : {str(valid_sample_counter)}')
        print(f'Train - Validation split %: Training set {(1 - _valid_data_porcentage) * 100}  - Validation set {_valid_data_porcentage * 100}')

    def create_training_file(self):
        self.output_data_folder = os.path.join(self.output_folder, _DATA_FOLDER_NAME)
        if os.path.exists(self.output_data_folder):
            print("--> The folder to process already have a data folder. Creating new data folder!")
            shutil.rmtree(self.output_data_folder)
        os.makedirs(self.output_data_folder)
        object_names_path = os.path.join(self.output_data_folder, _NAMES_FILE_NAME)
        with open(object_names_path, "w+") as object_names_file:
            for i in range(len(self.config["classes"])):
                object_names_file.write(f"{self.config['classes'][str(i)]}\n")
            print("obj.names file CREATED!")
        self.train_txt_path = os.path.join(self.output_data_folder, _TRAIN_FILE_NAME)
        with open(self.train_txt_path, "w+") as self.train_txt_file:
            self.samples_counter = 0
            print("train.txt file CREATED!")
        self.valid_txt_path = os.path.join(self.output_data_folder, _VALID_FILE_NAME)
        with open(self.valid_txt_path, "w+") as self.valid_txt_file:
            print("valid.txt file CREATED!")
        self.objects_data_folder = os.path.join(self.output_data_folder, _OBJECTS_FOLDER_NAME)
        os.makedirs(self.objects_data_folder)
        object_data_path = os.path.join(self.output_data_folder, _DATA_FILE_NAME)
        with open(object_data_path, "w+") as object_data_file:
            object_data_file.write(f'classes = {len(self.config["classes"])}\n')
            object_data_file.write(f'train = {_DATA_FOLDER_NAME}/{_TRAIN_FILE_NAME}\n')
            object_data_file.write(f'valid = {_DATA_FOLDER_NAME}/{_VALID_FILE_NAME}\n')
            object_data_file.write(f'names = {_DATA_FOLDER_NAME}/{_NAMES_FILE_NAME}\n')
            object_data_file.write(f'backup = backup/')

    def get_valid_samples_folders(self) -> None:
        self.samples_folder_path_list = []
        for samples_candidate_folder in self.subfolders_path_list:
            if os.path.exists(os.path.join(samples_candidate_folder, _VALIDATION_FOLDER_NAME)):
                self.samples_folder_path_list.append(os.path.join(samples_candidate_folder, _VALIDATION_FOLDER_NAME))
            else:
                print(f"A IMAGE SAMPLES FOLDER DOES NOT EXISTS!!! Skipping folder {samples_candidate_folder} from dataset creation!")

    @staticmethod
    def find_missing_files(filepath_list):
        missing_filepaths = []
        for file_path in filepath_list:
            target_path = str(os.path.splitext(file_path)[0]) + _ANNOTATION_EXTENSION
            if not os.path.isfile(target_path):
                missing_filepaths.append(target_path)
        print(missing_filepaths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for generating custom dataset for detection using YOLO Darknet format")
    parser.add_argument("parent_folder", help="Full path to parent folder containing subfolders with human validated  samples")
    parser.add_argument("output_folder", help="Full path to output folder where all validated and post-processed samples will be stored")
    parser.add_argument("configuration", help='Full path to configuration.json file')
    args = parser.parse_args()
    DetectorCustomDataset(args.parent_folder, args.output_folder, args.configuration).run()
