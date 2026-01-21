
# Dataset Usage Instructions

## Data Directory Structure

The dataset used in this project should be organized according to the following structure:

```
datasets/
└── processed/
    ...
    └── chb055/
        ├── train.csv
        ├── test.csv
        └── test_label.csv
    └── chb066/
        ├── train.csv
        ├── test.csv
        └── test_label.csv
    ...
```

## Data Acquisition

### 1. Core Dataset Files
The project repository already includes core dataset files:
- `datasets/processed/chb055/train.csv` - Training data
- `datasets/processed/chb055/test.csv` - Test data  
- `datasets/processed/chb055/test_label.csv` - Test labels

### 2. Complete Dataset Download
The complete EEG dataset files are large in size. Please download them from Google Drive:

**[Click here to download the complete dataset](Please add Google Drive link here)**

After downloading, extract the files and place them in the directory structure mentioned above.

## Dataset Path Configuration

When you need to use a different dataset, you must modify the dataset path configuration in the code. The configuration file is located at:

**Line 155 in `utils.py`**

Specific location: https://github.com/qiumiao30/EEG-SemLM/blob/main/utils.py#L155

This line of code sets the root directory path for the dataset. Modifying this line allows you to switch between different datasets.

## Quick Start

1. Download the complete dataset and place it in the correct directory
2. If you need to use a different dataset, modify the path configuration at line 155 in `utils.py`
3. Run the training script to begin using the data

## Important Notes

- Ensure all CSV files follow the same format as the existing `chb055` dataset
- The dataset should include standard EEG features and label columns
- For any data-related issues, please refer to the original paper's data preprocessing methods
