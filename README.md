# The Wayfinder

*An all-seeing analytical tool for tracking and identifying objects across the realms of data*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The Wayfinder is an advanced object identification analysis system inspired by the mystical seeing-stones from Tolkien's universe. This tool processes JSON annotation files to track objects across multiple images, identifying when the same object appears in different contexts and providing comprehensive insights about your dataset.

## 🔍 Key Features

- **Object Tracking**: Identifies the same objects across multiple images using their unique IDs
- **Dataset Statistics**: Provides detailed statistics about objects, images, and classes
- **ID Analysis**: Analyzes patterns in object IDs to reveal naming conventions and commonalities
- **Visualizations**: Generates informative charts to help understand object distributions
- **Empty Detection**: Identifies images that contain no annotated objects
- **JSON Support**: Works with Label Studio JSON annotations format

## 📁 Repository Structure

```
wayfinder/
├── main.py              # Main script for dataset analysis
├── input/               # Input directory for JSON annotation files
├── output/              # Output directory for results
└── requirements.txt     # Python dependencies
```

## ⚙️ Requirements

- Python 3.6+
- Dependencies listed in `requirements.txt`:
  - numpy
  - pandas
  - matplotlib
  - tqdm
  - Pillow

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wayfinder.git
cd wayfinder

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Usage

### Basic Usage

```bash
python main.py --json-dir ./input/your-annotations.json --output-dir ./output
```

### Additional Options

```bash
# Process all JSON files in a directory
python main.py --json-dir ./input/ --output-dir ./output

# Also identify images without objects
python main.py --json-dir ./input/your-annotations.json --output-dir ./output --find-empty
```

## 📋 Expected JSON Format

The Palantír is designed to work with Label Studio JSON format where object IDs are stored in the metadata:

```json
{
  "annotations": [
    {
      "result": [
        {
          "meta": {
            "text": ["IT000104"]  // Object ID here
          },
          "value": {
            "rectanglelabels": ["Class-Name"],  // Object class
            "x": 54.2,
            "y": 58.0,
            "width": 5.6,
            "height": 5.4
          }
        }
      ]
    }
  ]
}
```

## 📈 Output Files

The analysis generates several output files:

- `summary.json` - Overall statistics about objects, images, and classes
- `object_appearances.json` - Detailed tracking information for each object
- `all_objects.csv` - CSV file with all object instances
- `images_without_objects.txt` - List of images without objects (with `--find-empty` flag)
- Visualization charts in the `visualizations/` subdirectory:
  - Object appearance frequencies
  - Objects per image distribution
  - Class distribution
  - ID pattern analysis

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

*"Even the smallest person can change the course of the future." — J.R.R. Tolkien*
