import os
import json
from PIL import Image

# Paths
dataset_root = "/Users/doductai/Desktop/AI and ML/Object-Detection-Drone/data"

# Class mapping
class_names = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor"
}

def write_json_coco(images_dir, annotations_dir, output_json_path):
    # COCO structure
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in class_names.items()]
    }

    # Annotation ID counter
    annotation_id = 1
    num_skip_images=0 # skip images with no annotation
    num_total_images=len(sorted(os.listdir(images_dir)))

    # Process images and annotations

    for image_id, image_file in enumerate(sorted(os.listdir(images_dir))):
        if not image_file.endswith((".jpg", ".png")):
            continue

        image_path = os.path.join(images_dir, image_file)
        annotation_file = os.path.join(annotations_dir, os.path.splitext(image_file)[0] + ".txt")
        
        # Get image details
        img=Image.open(image_path)
        width, height = img.size

        # Add image info to COCO
        coco["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": width,
            "height": height
        })

        # Read annotation file
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                for line in f:
                    # Strip spaces and split the line
                    parts = [p.strip() for p in line.strip().split(",") if p.strip()]
                    # Ensure the line has the expected 8 fields
                    if len(parts) != 8:
                        print(f"Skipping malformed line: {line.strip()}")
                        print(parts)
                        continue

                    x_min, y_min, width, height, _, category_id, _, _ = map(int, parts)

                    # Skip invalid categories
                    if category_id not in class_names:
                        continue

                    # Add annotation info to COCO
                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x_min, y_min, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    annotation_id += 1
        else:
            num_skip_images+=1
            print(f"No annotations found for image: {image_file}")



    # Save COCO JSON
    with open(output_json_path, "w") as f:
        json.dump(coco, f, indent=4)

    print(f"Process {num_total_images} images | Skip {num_skip_images} images | Save {num_total_images-num_skip_images} images")
    print(f"COCO annotations saved to {output_json_path}")

def main():
    # train data
    train_images_dir = os.path.join(dataset_root, "VisDrone2019-DET-train/images")
    train_annotations_dir = os.path.join(dataset_root, "VisDrone2019-DET-train/annotations")
    output_train_json_path = os.path.join(dataset_root, "VisDrone2019-DET-train/train_coco.json")
    write_json_coco(train_images_dir, train_annotations_dir, output_train_json_path)
    
    # validation data
    val_images_dir = os.path.join(dataset_root, "VisDrone2019-DET-val/images")
    val_annotations_dir = os.path.join(dataset_root, "VisDrone2019-DET-val/annotations")
    output_val_json_path = os.path.join(dataset_root, "VisDrone2019-DET-val/val_coco.json")
    write_json_coco(val_images_dir, val_annotations_dir, output_val_json_path)

    # test data
    test_images_dir = os.path.join(dataset_root, "VisDrone2019-DET-test-dev/images")
    test_annotations_dir = os.path.join(dataset_root, "VisDrone2019-DET-test-dev/annotations")
    output_test_json_path = os.path.join(dataset_root, "VisDrone2019-DET-test-dev/test_coco.json")
    write_json_coco(test_images_dir, test_annotations_dir, output_test_json_path)

if __name__=="__main__":
    main()