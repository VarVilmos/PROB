"""
Generate COCO_INC image split files for 8-task class-incremental learning.

COCO_INC splits the TOWOD 80-class list into 8 tasks of 10 classes each:
  T1: classes  0- 9  (first half of TOWOD T1: VOC classes)
  T2: classes 10-19  (second half of TOWOD T1: VOC classes)
  T3: classes 20-29  (first half of TOWOD T2)
  T4: classes 30-39  (second half of TOWOD T2)
  T5: classes 40-49  (first half of TOWOD T3)
  T6: classes 50-59  (second half of TOWOD T3)
  T7: classes 60-69  (first half of TOWOD T4)
  T8: classes 70-79  (second half of TOWOD T4)

Usage (run from the PROB project root):
    python datasets/generate_coco_inc_splits.py
    python datasets/generate_coco_inc_splits.py --data_root ./data/OWOD
"""

import os
import argparse
import xml.etree.ElementTree as ET

# The 80 COCO/TOWOD class names in order (indices 0-79)
ALL_CLASSES = [
    # T1 (TOWOD VOC): indices 0-19
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    # T2 (TOWOD): indices 20-39
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    # T3 (TOWOD): indices 40-59
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    # T4 (TOWOD): indices 60-79
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl",
]

# 8 tasks, 10 classes each
COCO_INC_TASKS = {
    f"t{i+1}": ALL_CLASSES[i*10:(i+1)*10]
    for i in range(8)
}


def get_classes_in_xml(xml_path):
    """Return set of class names found in a VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return {obj.find("name").text for obj in root.findall("object")}


def generate_splits(data_root):
    annotation_dir = os.path.join(data_root, "Annotations")
    image_set_dir = os.path.join(data_root, "ImageSets", "COCO_INC")
    os.makedirs(image_set_dir, exist_ok=True)

    # Collect all XML files once
    print(f"Scanning annotations in {annotation_dir} ...")
    all_files = sorted(
        f[:-4] for f in os.listdir(annotation_dir) if f.endswith(".xml")
    )
    print(f"  Found {len(all_files)} annotation files.")

    # Cache class sets per file to avoid re-parsing
    print("  Parsing class names (this may take a minute)...")
    file_classes = {}
    for fname in all_files:
        xml_path = os.path.join(annotation_dir, fname + ".xml")
        file_classes[fname] = get_classes_in_xml(xml_path)

    all_task_images = set()

    for task_name, task_classes in COCO_INC_TASKS.items():
        task_class_set = set(task_classes)
        matching = [f for f, classes in file_classes.items() if classes & task_class_set]
        all_task_images.update(matching)

        out_path = os.path.join(image_set_dir, f"coco_inc_{task_name}_train.txt")
        with open(out_path, "w") as fp:
            fp.write("\n".join(matching))
        print(f"  {task_name} ({len(task_classes)} classes, {len(matching)} images) -> {out_path}")

    # Test split: all images that contain any of the 80 classes
    all_sorted = sorted(all_task_images)
    test_path = os.path.join(image_set_dir, "coco_inc_all_test.txt")
    with open(test_path, "w") as fp:
        fp.write("\n".join(all_sorted))
    print(f"\n  Test split ({len(all_sorted)} images) -> {test_path}")
    print("\nDone. Split files written to:", image_set_dir)
    print("\nClass assignments:")
    for task_name, task_classes in COCO_INC_TASKS.items():
        idx_start = list(COCO_INC_TASKS.keys()).index(task_name) * 10
        print(f"  {task_name}: indices {idx_start}-{idx_start+9}  {task_classes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data/OWOD", type=str)
    args = parser.parse_args()
    generate_splits(args.data_root)
