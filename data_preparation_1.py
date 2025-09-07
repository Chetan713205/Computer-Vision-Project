import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class ClothingDataPreparator:
    """
    Data preparation class for Clothing Attributes Dataset
    Handles loading, preprocessing, and organizing the dataset
    """

    def __init__(self, data_dir='data', output_dir='processed_data'):
        """
        Initialize the data preparator
        
        Args:
            data_dir (str): Directory containing the raw dataset
            output_dir (str): Directory to save processed data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.data_dir / 'images'
        self.labels_dir = self.data_dir / 'labels'

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)

        # Define the 36 clothing attributes: 23 binary + 3 sleeve + 3 neckline + 7 category
        self.attributes = [
            'necktie',          # 1: No necktie; 2: Has necktie
            'collar',           # 1: No collar; 2: Has collar
            'gender',           # 1: Male; 2: Female
            'placket',          # 1: No placket; 2: Has placket
            'skin_exposure',    # 1: Low exposure; 2: High exposure
            'wear_scarf',       # 1: No scarf; 2: Has scarf

            'pattern_solid',    # 1: No; 2: Yes
            'pattern_floral',   # 1: No; 2: Yes
            'pattern_spotted',  # 1: No; 2: Yes (spotted)
            'pattern_graphics', # 1: No; 2: Yes
            'pattern_plaid',    # 1: No; 2: Yes
            'pattern_striped',  # 1: No; 2: Yes

            'color_red',        # 1: No; 2: Yes
            'color_yellow',     # 1: No; 2: Yes
            'color_green',      # 1: No; 2: Yes
            'color_cyan',       # 1: No; 2: Yes
            'color_blue',       # 1: No; 2: Yes
            'color_purple',     # 1: No; 2: Yes
            'color_brown',      # 1: No; 2: Yes
            'color_white',      # 1: No; 2: Yes
            'color_gray',       # 1: No; 2: Yes
            'color_black',      # 1: No; 2: Yes
            'color_many',       # Many (>2) colors: 1: No; 2: Yes

            'sleeve_no',        # 1: No sleeves
            'sleeve_short',     # 2: Short sleeves
            'sleeve_long',      # 3: Long sleeves

            'neckline_v',       # 1: V-shape
            'neckline_round',   # 2: Round
            'neckline_other',   # 3: Other shapes

            'category_1',       # 1: Shirt
            'category_2',       # 2: Sweater
            'category_3',       # 3: T-shirt
            'category_4',       # 4: Outerwear
            'category_5',       # 5: Suit
            'category_6',       # 6: Tank Top
            'category_7'        # 7: Dress
        ]

        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    def load_image_paths(self):
        """
        Load all image paths from the images directory

        Returns:
            list: List of image file paths
        """
        image_paths = []
        if self.images_dir.exists():
            for ext in self.image_extensions:
                image_paths.extend(list(self.images_dir.glob(f'*{ext}')))
        print(f"Found {len(image_paths)} images in {self.images_dir}")
        return sorted(image_paths)

    def load_labels(self):
        """
        Load labels from MATLAB .mat files and map to clothing attributes with encoding as per README

        Returns:
            pd.DataFrame: DataFrame with image names and encoded attributes
        """
        import scipy.io

        image_paths = self.load_image_paths()
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")
        image_names = [p.name for p in image_paths]

        mat_files = sorted(self.labels_dir.glob('*.mat'))
        if not mat_files:
            raise FileNotFoundError(f"No .mat label files found in {self.labels_dir}")

        # Verify number of samples and adjust image_names accordingly
        first_mat = mat_files[0]
        mat_data = scipy.io.loadmat(first_mat)
        arr = None
        for key, val in mat_data.items():
            if isinstance(val, np.ndarray) and val.ndim in (1, 2) and val.dtype != object:
                arr = val
                break
        if arr is None:
            raise ValueError(f"No numeric array found in {first_mat.name}")
        flat = arr.squeeze()
        if flat.ndim != 1:
            raise ValueError(f"Array in {first_mat.name} is not 1D")
        num_samples = flat.shape[0]

        if len(image_names) > num_samples:
            print(f"Warning: Found {len(image_names)} images but labels have {num_samples} samples. Using first {num_samples} images.")
            image_names = image_names[:num_samples]
            image_paths = image_paths[:num_samples]
        elif len(image_names) < num_samples:
            raise ValueError(f"Found {len(image_names)} images but labels have {num_samples} samples.")

        # Initialize DataFrame with image_name and placeholders for all attributes
        labels_df = pd.DataFrame({'image_name': image_names})

        # Initialize columns for all attributes with zeros or zeros (for multi-class we will store numeric label)
        # For multi-class sleeve_length, neckline, category, keep original integer values (1,2,3,...)
        # For binary attributes, map presence=1 else 0

        # Initialize all columns with zeros or NaNs as appropriate
        for attr in self.attributes:
            if attr in ['sleeve_length', 'neckline', 'category']:
                labels_df[attr] = 0
            else:
                labels_df[attr] = 0

        # Mapping from .mat files to attribute columns and their present class values per README
        # For binary attributes: value=2 means presence (map to 1), else 0
        # For multi-class attributes: keep original integer from 1,2,3 etc.

        attr_mapping = {
            'necktie': {'column': 'necktie', 'presence_value': 2},
            'collar': {'column': 'collar', 'presence_value': 2},
            'gender': {'column': 'gender', 'presence_value': 2},
            'placket': {'column': 'placket', 'presence_value': 2},
            'skinexposure': {'column': 'skin_exposure', 'presence_value': 2},
            'wearscarf': {'column': 'wear_scarf', 'presence_value': 2},

            'pattern_solid': {'column': 'pattern_solid', 'presence_value': 2},
            'pattern_floral': {'column': 'pattern_floral', 'presence_value': 2},
            'pattern_spotted': {'column': 'pattern_spotted', 'presence_value': 2},
            'pattern_graphics': {'column': 'pattern_graphics', 'presence_value': 2},
            'pattern_plaid': {'column': 'pattern_plaid', 'presence_value': 2},
            'pattern_striped': {'column': 'pattern_striped', 'presence_value': 2},

            'red': {'column': 'color_red', 'presence_value': 2},
            'yellow': {'column': 'color_yellow', 'presence_value': 2},
            'green': {'column': 'color_green', 'presence_value': 2},
            'cyan': {'column': 'color_cyan', 'presence_value': 2},
            'blue': {'column': 'color_blue', 'presence_value': 2},
            'purple': {'column': 'color_purple', 'presence_value': 2},
            'brown': {'column': 'color_brown', 'presence_value': 2},
            'white': {'column': 'color_white', 'presence_value': 2},
            'gray': {'column': 'color_gray', 'presence_value': 2},
            'black': {'column': 'color_black', 'presence_value': 2},
            'manycolor': {'column': 'color_many', 'presence_value': 2},

            'sleevelength': {'column': 'sleeve_length'},  # multiclass: 1,2,3
            'neckline': {'column': 'neckline'},         # multiclass: 1,2,3
            'category': {'column': 'category'}           # multiclass: 1 to 7
        }

        # Process each mat file to fill labels_df appropriately
        for mat_path in mat_files:
            attr_name = mat_path.stem.replace('_GT', '')
            if attr_name not in attr_mapping:
                print(f"Skipping unknown attribute file: {mat_path.name}")
                continue
            print(f"Loading attribute '{attr_name}' from {mat_path.name}")
            mat_data = scipy.io.loadmat(mat_path)
            arr = None
            for key, val in mat_data.items():
                if isinstance(val, np.ndarray) and val.ndim in (1, 2) and val.dtype != object:
                    arr = val
                    break
            if arr is None:
                print(f"Warning: No numeric array found in {mat_path.name}")
                continue
            flat = arr.squeeze()
            if flat.ndim != 1 or flat.shape[0] != num_samples:
                print(f"Warning: Attribute '{attr_name}' length {flat.shape} does not match expected {num_samples}")
                continue

            map_info = attr_mapping[attr_name]
            col = map_info['column']

            if attr_name in ['sleevelength', 'neckline', 'category']:
                # Multi-class attributes: copy values as-is (with 0 allowed if missing)
                labels_df[col] = flat.astype(int)
            else:
                # Binary attribute: presence if value == presence_value, else 0
                presence_val = map_info['presence_value']
                labels_df[col] = (flat == presence_val).astype(int)

        print(f"Loaded and mapped labels for {len(labels_df)} samples with attributes: {self.attributes}")

        # Handle missing values and one-hot encode multi-class attributes
        labels_df.replace(-9223372036854775808, np.nan, inplace=True)

        # One-hot encode sleeve_length
        labels_df['sleeve_no'] = (labels_df['sleeve_length'] == 1).astype(int)
        labels_df['sleeve_short'] = (labels_df['sleeve_length'] == 2).astype(int)
        labels_df['sleeve_long'] = (labels_df['sleeve_length'] == 3).astype(int)

        # One-hot encode neckline
        labels_df['neckline_v'] = (labels_df['neckline'] == 1).astype(int)
        labels_df['neckline_round'] = (labels_df['neckline'] == 2).astype(int)
        labels_df['neckline_other'] = (labels_df['neckline'] == 3).astype(int)

        # One-hot encode category
        for i in range(1, 8):
            labels_df[f'category_{i}'] = (labels_df['category'] == i).astype(int)

        # Drop original multi-class columns
        labels_df.drop(columns=['sleeve_length', 'neckline', 'category'], inplace=True)

        # Fill NaN with 0
        labels_df.fillna(0, inplace=True)

        return labels_df

    def validate_data(self, labels_df):
        """
        Validate that images exist for all labels and vice versa

        Args:
            labels_df (pd.DataFrame): DataFrame containing labels

        Returns:
            pd.DataFrame: Validated and cleaned DataFrame
        """
        image_paths = self.load_image_paths()
        image_names = {path.stem for path in image_paths}  # Remove extensions

        # Determine image name column
        label_image_col = None
        for col in ['image_name', 'filename', 'image', 'id', 'image_id']:
            if col in labels_df.columns:
                label_image_col = col
                break
        if label_image_col is None:
            print("Warning: Could not find image name column in labels. Using index to create image_name.")
            labels_df['image_name'] = [f"img_{i}.jpg" for i in range(len(labels_df))]
            label_image_col = 'image_name'

        # Remove extensions to compare names
        labels_df['clean_name'] = labels_df[label_image_col].str.replace(r'\.[^.]*$', '', regex=True)
        label_names = set(labels_df['clean_name'])
        common_names = image_names.intersection(label_names)

        print(f"Images found: {len(image_names)}")
        print(f"Labels found: {len(label_names)}")
        print(f"Common samples: {len(common_names)}")

        # Filter DataFrame for common samples only
        validated_df = labels_df[labels_df['clean_name'].isin(common_names)].copy()
        validated_df = validated_df.drop(columns=['clean_name'])

        print(f"Validated dataset size: {len(validated_df)}")
        return validated_df

    def preprocess_images(self, image_paths, target_size=(224, 224), save_processed=True):
        """
        Preprocess images: resize, normalize, and optionally save them

        Args:
            image_paths (list): List of image file paths
            target_size (tuple): Target (width, height) size for resizing
            save_processed (bool): Whether to save the processed images

        Returns:
            np.array: Array of processed images
        """
        processed_images = []
        processed_dir = self.output_dir / 'processed_images'
        if save_processed:
            processed_dir.mkdir(exist_ok=True)

        print(f"Processing {len(image_paths)} images...")
        for i, img_path in enumerate(image_paths):
            try:
                image = cv2.imread(str(img_path))        # OpenCV to load each image
                if image is None:
                    print(f"Warning: Could not load image {img_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                image = cv2.resize(image, target_size)     # Ensuring const. input size
                image = image.astype(np.float32) / 255.0   # Normalization of pixel values 0-1
                processed_images.append(image)

                if save_processed:
                    processed_path = processed_dir / f"processed_{img_path.name}"
                    plt.imsave(processed_path, image)

                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(image_paths)} images")  # Progress tracking

            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                continue

        print(f"Successfully processed {len(processed_images)} images")
        return np.array(processed_images)

    def create_data_splits(self, labels_df, test_size=0.2, val_size=0.1, random_state=42):
        """
        Create train/validation/test splits

        Args:
            labels_df (pd.DataFrame): DataFrame containing labels
            test_size (float): Proportion of data for test split
            val_size (float): Proportion of train data for validation split
            random_state (int): Random seed

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        train_val_df, test_df = train_test_split(
            labels_df,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            random_state=random_state,
            stratify=None
        )

        print(f"Data splits created:")
        print(f" Training: {len(train_df)} samples")
        print(f" Validation: {len(val_df)} samples")
        print(f" Test: {len(test_df)} samples")

        return train_df, val_df, test_df

    def analyze_attributes(self, labels_df):
        """
        Analyze attribute distributions and correlations

        Args:
            labels_df (pd.DataFrame): DataFrame containing labels
        """
        # Attribute columns exclude image name columns
        attr_columns = [col for col in labels_df.columns if col not in ['image_name', 'filename', 'image', 'id', 'image_id']]

        if not attr_columns:
            print("No attribute columns found for analysis")
            return

        print("\n=== ATTRIBUTE ANALYSIS ===")
        print("\nAttribute Distributions:")

        for col in attr_columns:
            if labels_df[col].dtype in ['int64', 'float64']:
                unique_vals = set(labels_df[col].unique())
                if unique_vals.issubset({0, 1}):
                    pos_count = (labels_df[col] == 1).sum()
                    total_count = len(labels_df)
                    print(f" {col}: {pos_count}/{total_count} ({pos_count/total_count*100:.1f}%)")
                else:
                    # Multi-class attributes
                    print(f" {col}: mean={labels_df[col].mean():.3f}, std={labels_df[col].std():.3f}")

        # Correlation heatmap
        if len(attr_columns) > 1:
            plt.figure(figsize=(15, 12))
            correlation_matrix = labels_df[attr_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                        square=True, fmt='.2f')
            plt.title('Attribute Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'attribute_correlations.png', dpi=300, bbox_inches='tight')
            plt.show()

            correlation_matrix.to_csv(self.output_dir / 'attribute_correlations.csv')

        # Attribute co-occurrence analysis for binary attributes
        binary_attrs = [col for col in attr_columns if set(labels_df[col].unique()).issubset({0, 1})]

        if len(binary_attrs) >= 2:
            print("\nTop Attribute Combinations:")
            combo_counts = {}
            for idx, row in labels_df.iterrows():
                active_attrs = [attr for attr in binary_attrs if row[attr] == 1]
                if len(active_attrs) >= 2:
                    combo = tuple(sorted(active_attrs))
                    combo_counts[combo] = combo_counts.get(combo, 0) + 1

            sorted_combos = sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)
            for combo, count in sorted_combos[:10]:
                print(f" {' + '.join(combo)}: {count} times")

    def save_processed_data(self, train_df, val_df, test_df, images=None):
        """
        Save processed data files and statistics

        Args:
            train_df, val_df, test_df: DataFrames with data splits
            images (np.array): Processed image array (optional)
        """
        print("\n=== SAVING PROCESSED DATA ===")

        train_df.to_csv(self.output_dir / 'train_labels.csv', index=False)
        val_df.to_csv(self.output_dir / 'val_labels.csv', index=False)
        test_df.to_csv(self.output_dir / 'test_labels.csv', index=False)

        if images is not None:
            np.save(self.output_dir / 'processed_images.npy', images)
            print(f"Saved {len(images)} processed images to processed_images.npy")

        attr_columns = [col for col in train_df.columns if col not in ['image_name', 'filename', 'image', 'id', 'image_id']]
        with open(self.output_dir / 'attribute_names.txt', 'w') as f:
            for attr in attr_columns:
                f.write(f"{attr}\n")

        stats = {
            'total_samples': len(train_df) + len(val_df) + len(test_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'num_attributes': len(attr_columns),
            'attribute_names': attr_columns,
            'image_shape': list(images.shape[1:]) if images is not None else None
        }

        with open(self.output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved dataset splits and statistics to {self.output_dir}")

    def create_data_loaders_config(self):
        """
        Create configuration JSON for PyTorch data loaders
        """
        config = {
            'image_transforms': {
                'train': [
                    'resize_224',
                    'random_horizontal_flip_0.5',
                    'random_rotation_15',
                    'normalize_imagenet'
                ],
                'val_test': [
                    'resize_224',
                    'normalize_imagenet'
                ]
            },
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': True
        }

        with open(self.output_dir / 'dataloader_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("Saved data loader configuration")

    def run_complete_preparation(self):
        """
        Run complete data preparation pipeline
        """
        print("=== CLOTHING ATTRIBUTES DATASET PREPARATION ===")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")

        # 1: Load Labels
        labels_df = self.load_labels()

        # 2: Validate Data
        validated_df = self.validate_data(labels_df)
        if len(validated_df) == 0:
            print("No valid samples found. Please check your data directory structure.")
            return

        # 3: Analyze Attributes
        self.analyze_attributes(validated_df)

        # 4: Create Data Splits
        train_df, val_df, test_df = self.create_data_splits(validated_df)

        # 5: Process Images (optional, set False to avoid memory issues)
        process_images = False
        images = None
        if process_images:
            image_paths = self.load_image_paths()
            if image_paths:
                images = self.preprocess_images(image_paths, target_size=(224, 224))

        # 6: Save All Processed Data
        self.save_processed_data(train_df, val_df, test_df, images)

        # 7: Create Data Loader Configuration
        self.create_data_loaders_config()

        print("\n=== DATA PREPARATION COMPLETE ===")
        print(f"Check the '{self.output_dir}' directory for all processed files")


if __name__ == "__main__":
    prep = ClothingDataPreparator(data_dir='../data', output_dir='../processed_data')
    prep.run_complete_preparation()
