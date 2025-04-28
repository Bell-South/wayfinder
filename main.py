#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter

class ObjectIDAnalyzer:
    def __init__(self):
        """
        Initialize the Object ID Analyzer to process Label Studio annotations
        and identify objects with consistent IDs across images.
        """
        self.object_appearances = defaultdict(list)
        self.image_objects = defaultdict(list)
        self.class_distribution = defaultdict(int)
        self.id_prefix_counts = defaultdict(int)
        self.id_length_counts = defaultdict(int)
        self.total_objects = 0
        self.total_images_with_objects = 0
        self.total_images = 0
        self.objects_per_image_counts = defaultdict(int)
    
    def extract_object_id(self, result_item):
        """
        Extract object ID from the result item metadata
        
        Args:
            result_item: Result item from annotation
            
        Returns:
            Object ID or None if not found
        """
        # Check if there's metadata with text field (containing ID)
        if isinstance(result_item, dict) and 'meta' in result_item:
            meta = result_item['meta']
            if isinstance(meta, dict) and 'text' in meta and isinstance(meta['text'], list) and len(meta['text']) > 0:
                # Return the first text entry in the metadata as the ID
                return meta['text'][0]
        
        return None
    
    def extract_object_info(self, result_item, image_name):
        """
        Extract object information from a result item
        
        Args:
            result_item: Result item from annotation
            image_name: Name of the image
            
        Returns:
            Dictionary with object information or None if not valid
        """
        if not isinstance(result_item, dict):
            return None
        
        # Get object ID
        object_id = self.extract_object_id(result_item)
        if not object_id:
            return None
        
        # Get object class
        object_class = None
        if 'value' in result_item and 'rectanglelabels' in result_item['value']:
            labels = result_item['value']['rectanglelabels']
            if isinstance(labels, list) and len(labels) > 0:
                object_class = labels[0]
        
        # Get bounding box
        bbox = None
        if 'value' in result_item:
            value = result_item['value']
            if ('x' in value and 'y' in value and 
                'width' in value and 'height' in value):
                x = value['x']
                y = value['y']
                width = value['width']
                height = value['height']
                bbox = [x, y, width, height]
        
        # Create object info
        object_info = {
            'id': object_id,
            'class': object_class,
            'bbox': bbox,
            'image': image_name
        }
        
        return object_info
    
    def process_json_file(self, json_path):
        """
        Process a single JSON file
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            Number of objects found in the file
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON file: {json_path}")
            return 0
        
        # Handle both single object and list formats
        if isinstance(data, dict):
            data_list = [data]
        elif isinstance(data, list):
            data_list = data
        else:
            print(f"Unknown JSON format in {json_path}")
            return 0
        
        objects_found = 0
        
        # Process each item in the data
        for item in data_list:
            if not isinstance(item, dict):
                continue
                
            # Get the image name
            image_name = None
            if 'file_upload' in item:
                image_name = item['file_upload']
            elif 'data' in item and 'image' in item['data']:
                image_path = item['data']['image']
                image_name = os.path.basename(image_path)
            
            if not image_name:
                continue
                
            self.total_images += 1
            has_objects = False
            image_object_count = 0
            
            # Get annotations
            annotations = []
            if 'annotations' in item and isinstance(item['annotations'], list):
                for ann in item['annotations']:
                    if 'result' in ann and ann['result']:
                        annotations.extend(ann['result'] if isinstance(ann['result'], list) else [ann['result']])
            
            # Process each annotation
            for result_item in annotations:
                object_info = self.extract_object_info(result_item, image_name)
                
                if object_info:
                    # Add to object tracking
                    object_id = object_info['id']
                    self.object_appearances[object_id].append(object_info)
                    self.image_objects[image_name].append(object_id)
                    
                    # Update class distribution
                    if object_info['class']:
                        self.class_distribution[object_info['class']] += 1
                    
                    # Update ID prefix and length counts
                    if len(object_id) >= 2:
                        prefix = object_id[:2]  # First two characters as prefix
                        self.id_prefix_counts[prefix] += 1
                        
                    id_length = len(object_id)
                    self.id_length_counts[id_length] += 1
                    
                    objects_found += 1
                    image_object_count += 1
                    has_objects = True
            
            if has_objects:
                self.total_images_with_objects += 1
                self.objects_per_image_counts[image_object_count] += 1
        
        return objects_found
    
    def process_json_dataset(self, json_path, output_dir=None):
        """
        Process JSON files - can handle both directory and single file
        
        Args:
            json_path: Path to JSON file or directory with JSON files
            output_dir: Directory to save output files
            
        Returns:
            Summary dictionary with analysis results
        """
        # Reset counters
        self.object_appearances.clear()
        self.image_objects.clear()
        self.class_distribution.clear()
        self.id_prefix_counts.clear()
        self.id_length_counts.clear()
        self.total_objects = 0
        self.total_images_with_objects = 0
        self.total_images = 0
        self.objects_per_image_counts.clear()
        
        path = Path(json_path)
        
        if path.is_file():
            # Handle single file
            print(f"Processing single JSON file: {path}")
            objects_in_file = self.process_json_file(path)
            self.total_objects += objects_in_file
        elif path.is_dir():
            # Handle directory
            json_files = []
            for ext in ['json']:
                json_files.extend(list(path.glob(f"**/*.{ext}")))
            
            print(f"Found {len(json_files)} JSON annotation files")
            
            # Process all files
            for json_path in tqdm(json_files):
                objects_in_file = self.process_json_file(json_path)
                self.total_objects += objects_in_file
        else:
            print(f"Error: {json_path} is not a valid file or directory")
            return {
                'summary': {
                    'total_images': 0,
                    'images_with_objects': 0,
                    'images_without_objects': 0,
                    'total_objects': 0,
                    'unique_objects': 0,
                    'classes': {},
                    'object_appearances': {}
                },
                'dataframe': pd.DataFrame()
            }
        
        # Analyze ID patterns
        id_patterns = self.analyze_id_patterns()
        
        # Generate summary
        summary = {
            'total_images': self.total_images,
            'images_with_objects': self.total_images_with_objects,
            'images_without_objects': self.total_images - self.total_images_with_objects,
            'total_objects': self.total_objects,
            'unique_objects': len(self.object_appearances),
            'classes': dict(self.class_distribution),
            'object_appearances': {k: len(v) for k, v in self.object_appearances.items()},
            'id_patterns': id_patterns,
            'id_prefixes': dict(sorted(self.id_prefix_counts.items(), key=lambda x: x[1], reverse=True)),
            'id_lengths': dict(sorted(self.id_length_counts.items())),
            'objects_per_image': dict(sorted(self.objects_per_image_counts.items()))
        }
        
        # Create DataFrame with all object appearances
        rows = []
        for obj_id, appearances in self.object_appearances.items():
            for appearance in appearances:
                rows.append({
                    'object_id': obj_id,
                    'image': appearance['image'],
                    'class': appearance['class']
                })
        
        df = pd.DataFrame(rows)
        
        # Save results if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save summary to JSON
            with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save object appearances
            appearances_data = {}
            for obj_id, appearances in self.object_appearances.items():
                appearances_data[obj_id] = [
                    {'image': a['image'], 'class': a['class']} for a in appearances
                ]
            
            with open(os.path.join(output_dir, 'object_appearances.json'), 'w') as f:
                json.dump(appearances_data, f, indent=2)
            
            # Save DataFrame to CSV
            if not df.empty:
                df.to_csv(os.path.join(output_dir, 'all_objects.csv'), index=False)
            
            # Generate visualizations
            self.generate_visualizations(output_dir, df)
        
        return {
            'summary': summary,
            'dataframe': df
        }
    
    def analyze_id_patterns(self):
        """
        Analyze patterns in the object IDs
        
        Returns:
            Dictionary with ID pattern information
        """
        if not self.object_appearances:
            return {}
            
        # Get all unique IDs
        all_ids = list(self.object_appearances.keys())
        
        # Calculate average ID length
        avg_length = sum(len(id) for id in all_ids) / len(all_ids)
        
        # Check for common patterns
        patterns = {}
        
        # Check if IDs have common prefixes
        prefixes = {}
        for id in all_ids:
            if len(id) >= 2:
                prefix = id[:2]
                if prefix not in prefixes:
                    prefixes[prefix] = 0
                prefixes[prefix] += 1
        
        # Find the most common prefix
        most_common_prefix = None
        max_count = 0
        for prefix, count in prefixes.items():
            if count > max_count:
                max_count = count
                most_common_prefix = prefix
        
        patterns['most_common_prefix'] = most_common_prefix
        patterns['prefix_coverage'] = max_count / len(all_ids) if all_ids else 0
        
        # Check for numeric parts
        numeric_count = 0
        for id in all_ids:
            # Check if the ID ends with numbers
            numeric_suffix = ''
            for char in reversed(id):
                if char.isdigit():
                    numeric_suffix = char + numeric_suffix
                else:
                    break
            
            if numeric_suffix:
                numeric_count += 1
        
        patterns['ids_with_numeric_suffix'] = numeric_count
        patterns['numeric_suffix_percentage'] = numeric_count / len(all_ids) if all_ids else 0
        
        # Check for format consistency
        format_types = {
            'all_numeric': 0,
            'mixed': 0,
            'alpha_prefix_numeric_suffix': 0
        }
        
        for id in all_ids:
            if id.isdigit():
                format_types['all_numeric'] += 1
            elif any(c.isdigit() for c in id) and any(c.isalpha() for c in id):
                # Check if it follows alpha prefix + numeric suffix pattern
                alpha_part = ''
                numeric_part = ''
                for char in id:
                    if char.isalpha():
                        if numeric_part:  # Already found numeric part, so it's mixed
                            alpha_part = ''
                            break
                        alpha_part += char
                    elif char.isdigit():
                        numeric_part += char
                
                if alpha_part and numeric_part:
                    format_types['alpha_prefix_numeric_suffix'] += 1
                else:
                    format_types['mixed'] += 1
            else:
                format_types['mixed'] += 1
        
        patterns['format_types'] = {k: v / len(all_ids) for k, v in format_types.items()}
        
        # Calculate min, max, and median lengths
        lengths = [len(id) for id in all_ids]
        patterns['min_length'] = min(lengths)
        patterns['max_length'] = max(lengths)
        patterns['avg_length'] = avg_length
        patterns['median_length'] = sorted(lengths)[len(lengths) // 2]
        
        return patterns
    
    def generate_visualizations(self, output_dir, df):
        """
        Generate visualizations of the analysis results
        
        Args:
            output_dir: Directory to save visualizations
            df: DataFrame with object appearances
        """
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Skip if DataFrame is empty
        if df.empty:
            print("No data available to generate visualizations")
            return
        
        # 1. Object appearance counts
        obj_counts = df['object_id'].value_counts()
        
        plt.figure(figsize=(10, 6))
        if len(obj_counts) > 30:
            # For many objects, show top 30
            obj_counts.head(30).plot(kind='bar')
            plt.title('Top 30 Object Appearance Counts')
        else:
            obj_counts.plot(kind='bar')
            plt.title('Object Appearance Counts')
        
        plt.xlabel('Object ID')
        plt.ylabel('Number of Appearances')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'object_counts.png'))
        plt.close()
        
        # 2. Objects per image distribution
        img_counts = df.groupby('image')['object_id'].count()
        
        plt.figure(figsize=(10, 6))
        plt.hist(img_counts, bins=min(20, len(set(img_counts))))
        plt.title('Objects per Image Distribution')
        plt.xlabel('Number of Objects')
        plt.ylabel('Number of Images')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'objects_per_image.png'))
        plt.close()
        
        # 3. Class distribution (if available)
        if 'class' in df.columns and df['class'].notna().any():
            class_counts = df['class'].value_counts()
            
            plt.figure(figsize=(10, 6))
            class_counts.plot(kind='bar')
            plt.title('Object Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'class_distribution.png'))
            plt.close()
        
        # 4. Images with most objects
        top_images = img_counts.sort_values(ascending=False).head(20)
        
        plt.figure(figsize=(12, 6))
        top_images.plot(kind='bar')
        plt.title('Images with Most Objects')
        plt.xlabel('Image')
        plt.ylabel('Number of Objects')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'top_images.png'))
        plt.close()
        
        # 5. ID prefix distribution
        id_prefixes = {}
        for obj_id in df['object_id'].unique():
            if len(obj_id) >= 2:
                prefix = obj_id[:2]
                if prefix not in id_prefixes:
                    id_prefixes[prefix] = 0
                id_prefixes[prefix] += 1
        
        if id_prefixes:
            plt.figure(figsize=(10, 6))
            pd.Series(id_prefixes).sort_values(ascending=False).plot(kind='bar')
            plt.title('ID Prefix Distribution')
            plt.xlabel('Prefix')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'id_prefix_distribution.png'))
            plt.close()
        
        # 6. ID length distribution
        id_lengths = [len(obj_id) for obj_id in df['object_id'].unique()]
        
        plt.figure(figsize=(10, 6))
        plt.hist(id_lengths, bins=min(15, len(set(id_lengths))))
        plt.title('ID Length Distribution')
        plt.xlabel('ID Length')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'id_length_distribution.png'))
        plt.close()
        
        # 7. Create a tracking visualization - where do objects appear
        if len(df['object_id'].unique()) <= 50:  # Only if reasonable number of objects
            # Get top objects by appearance count
            top_objects = obj_counts.head(50).index.tolist()
            top_images = img_counts.head(30).index.tolist()
            
            # Create a matrix of appearances
            filtered_df = df[df['object_id'].isin(top_objects) & df['image'].isin(top_images)]
            pivot_data = pd.crosstab(filtered_df['image'], filtered_df['object_id'])
            
            plt.figure(figsize=(15, 10))
            plt.imshow(pivot_data.values, cmap='viridis', aspect='auto')
            plt.colorbar(label='Presence')
            plt.xticks(range(len(pivot_data.columns)), pivot_data.columns, rotation=90)
            plt.yticks(range(len(pivot_data.index)), pivot_data.index)
            plt.xlabel('Object ID')
            plt.ylabel('Image')
            plt.title('Object Tracking Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'tracking_matrix.png'))
            plt.close()
    
    def identify_images_without_objects(self, json_path):
        """
        Identify images that don't have any objects with IDs
        
        Args:
            json_path: Path to JSON file or directory
            
        Returns:
            List of image names without objects
        """
        # Process the dataset first if not already processed
        if not self.image_objects:
            self.process_json_dataset(json_path)
        
        # Find all images that don't have objects
        images_without_objects = []
        
        path = Path(json_path)
        json_files = []
        
        if path.is_file():
            json_files.append(path)
        elif path.is_dir():
            for ext in ['json']:
                json_files.extend(list(path.glob(f"**/*.{ext}")))
        
        # Check each JSON file
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                continue
            
            # Handle both single object and list formats
            if isinstance(data, dict):
                data_list = [data]
            elif isinstance(data, list):
                data_list = data
            else:
                continue
            
            # Check each item
            for item in data_list:
                if not isinstance(item, dict):
                    continue
                    
                # Get the image name
                image_name = None
                if 'file_upload' in item:
                    image_name = item['file_upload']
                elif 'data' in item and 'image' in item['data']:
                    image_path = item['data']['image']
                    image_name = os.path.basename(image_path)
                
                if not image_name:
                    continue
                    
                # Check if image is in image_objects
                if image_name not in self.image_objects or not self.image_objects[image_name]:
                    images_without_objects.append(image_name)
        
        return images_without_objects

def main():
    parser = argparse.ArgumentParser(description='Analyze Object IDs in JSON Annotations')
    parser.add_argument('--json-dir', type=str, required=True,
                        help='Path to JSON file or directory containing JSON files')
    parser.add_argument('--output-dir', type=str, default='id_analysis',
                        help='Output directory for analysis results (default: id_analysis)')
    parser.add_argument('--find-empty', action='store_true',
                        help='Identify images without object IDs')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ObjectIDAnalyzer()
    
    # Process dataset
    print(f"Processing annotations in {args.json_dir}")
    results = analyzer.process_json_dataset(args.json_dir, args.output_dir)
    
    # Print summary
    summary = results['summary']
    print("\nAnalysis Summary:")
    print(f"Total images processed: {summary['total_images']}")
    print(f"Images with objects: {summary['images_with_objects']}")
    print(f"Images without objects: {summary['images_without_objects']}")
    print(f"Total objects detected: {summary['total_objects']}")
    print(f"Unique object IDs: {summary['unique_objects']}")
    
    # Print class distribution if available
    if summary['classes']:
        print("\nClass distribution:")
        for cls, count in sorted(summary['classes'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {count}")
    
    # Print ID pattern information
    if 'id_patterns' in summary and summary['id_patterns']:
        patterns = summary['id_patterns']
        print("\nID Pattern Analysis:")
        print(f"  ID Length: Min={patterns['min_length']}, Max={patterns['max_length']}, Avg={patterns['avg_length']:.2f}")
        
        if 'most_common_prefix' in patterns:
            print(f"  Most common prefix: '{patterns['most_common_prefix']}' " +
                  f"(found in {patterns['prefix_coverage']*100:.1f}% of IDs)")
        
        if 'format_types' in patterns:
            format_types = patterns['format_types']
            print("  ID Format Types:")
            for fmt, pct in format_types.items():
                print(f"    {fmt}: {pct*100:.1f}%")
    
    # Print objects per image statistics
    if 'objects_per_image' in summary:
        obj_per_img = summary['objects_per_image']
        if obj_per_img:
            total_images = sum(obj_per_img.values())
            print("\nObjects per image:")
            for count, num_images in sorted(obj_per_img.items()):
                print(f"  {count} objects: {num_images} images ({num_images/total_images*100:.1f}%)")
    
    # Print ID prefix information
    if 'id_prefixes' in summary:
        top_prefixes = dict(list(summary['id_prefixes'].items())[:5])
        if top_prefixes:
            print("\nTop ID prefixes:")
            for prefix, count in top_prefixes.items():
                print(f"  '{prefix}': {count} objects")
    
    # Identify images without objects if requested
    if args.find_empty:
        print("\nIdentifying images without objects...")
        images_without_objects = analyzer.identify_images_without_objects(args.json_dir)
        
        print(f"Found {len(images_without_objects)} images without objects")
        
        # Save list of images without objects
        if args.output_dir:
            with open(os.path.join(args.output_dir, 'images_without_objects.txt'), 'w') as f:
                for img in images_without_objects:
                    f.write(f"{img}\n")
            
            print(f"List saved to {os.path.join(args.output_dir, 'images_without_objects.txt')}")
    
    print(f"\nDetailed results saved to {args.output_dir}")

if __name__ == "__main__":
    main()