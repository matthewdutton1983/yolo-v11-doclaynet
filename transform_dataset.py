import json
import os
import shutil
from pathlib import Path

import tqdm
import typer
import yaml


def main(root_folder: Path = Path("./datasets")):
  # Precompute constants
  scaling_factor = 1 / 1025

  # Write YAML configuration
  with open(root_folder / "data.yml", "w") as f:
    yaml.dump(
      {
        "path": "./",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {
                    "0": "Caption",
                    "1": "Footnote",
                    "2": "Formula",
                    "3": "List-item",
                    "4": "Page-footer",
                    "5": "Page-header",
                    "6": "Picture",
                    "7": "Section-header",
                    "8": "Table",
                    "9": "Text",
                    "10": "Title",
      }
    )

  for folder in ["val", "test", "train"]:
    print(f"Converting {folder} dataset...")

    # Prepare paths
    labels_folder = root_folder / "labels" / folder
    images_folder = root_folder / "images" / folder
    coco_json_path = root_folder / "COCO" / f"{folder}.json"
    png_folder = root_folder / "PNG"

    # Create directories if not exist
    labels_folder.mkdir(parents=True, exist_ok=True)
    images_folder.mkdir(parents=True, exist_ok=True)
    
    # Load JSON data
    with open(coco_json_path) as f:
        bigjson = json.load(f)

    # Move images
    for image in tqdm.tqdm(bigjson["images"], desc=f"Moving images for {folder}"):
        image_id = image["id"]
        filename = image["file_name"]
        src_path = png_folder / filename
        dst_path = images_folder / f"{image_id}.png"
        shutil.move(src_path, dst_path)
    
    # Batch process and write labels
    label_data = {}
    for annotation in bigjson["annotations"]:
        image_id = annotation["image_id"]
        filename = f"{image_id}.txt"
        left, top, width, height = annotation["bbox"]
        
        # Scale bounding box values
        left *= scaling_factor
        top *= scaling_factor
        width *= scaling_factor
        height *= scaling_factor
        center_x = left + width / 2
        center_y = top + height / 2
        category_id = annotation["category_id"] - 1
        
        # Append to corresponding label file content
        if filename not in label_data:
            label_data[filename] = []
        label_data[filename].append(f"{category_id} {center_x} {center_y} {width} {height}\n")
    
    # Write labels to files in batch
    for filename, lines in tqdm.tqdm(label_data.items(), desc=f"Writing labels for {folder}"):
        with open(labels_folder / filename, "a") as f:
            f.writelines(lines)

if __name__ == "__main__":
    typer.run(main)
  
