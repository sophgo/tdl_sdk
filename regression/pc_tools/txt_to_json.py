import argparse
import os
import json
from pathlib import Path
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--bbox_threshold", type=float, default=0.5)
    parser.add_argument("--score_threshold", type=float, default=0.1)
    parser.add_argument("--position_threshold", type=float, default=0.1)
    parser.add_argument("--image_dir", required=True, type=str)
    parser.add_argument("--chip", required=True, type=str)
    parser.add_argument("--txt_dir", required=True, type=str)
    parser.add_argument("--json_path", required=True, type=str)
    parser.add_argument("--task", required=True, type=str)

    return parser.parse_args()


def load_json(json_path):
    if os.path.exists(json_path):
        json_status = "exist"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"从 {json_path} 加载现有的JSON数据\n")

    else:
        json_status = "new"
        print(f"JSON文件 {json_path} 不存在，将创建一个新文件\n")
        data = {}
    return data, json_status


def save_json(data, json_path, json_status):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    if json_status == "new":
        print(f"JSON数据已成功保存到 {json_path}\n")
    else:
        print(f"JSON数据已成功更新到 {json_path}\n")


def process_detection_txt_files(txt_dir, image_dir=None):
    annotations = {}
    txt_path = Path(txt_dir)

    txt_files = list(txt_path.glob("*.txt"))
    if not txt_files:
        print(f"在目录 {txt_dir} 中未找到任何TXT文件\n")
        return annotations

    for txt_file in txt_files:
        filename = txt_file.stem
        img_file = list(Path(image_dir).glob(f"{filename}.*"))[0].name
        img_file = str(img_file)

        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        annotation_list = []
        for line in lines:
            parts = line.strip().split()
            x1, y1, x2, y2 = map(float, parts[:4])
            if len(parts) == 6:
                class_id = int(parts[4])
                score = float(parts[5])
            elif len(parts) == 5:
                score = float(parts[4])
                class_id = None
            annotation = {
                "bbox": [x1, y1, x2, y2],
                "score": score,
                "class_id": class_id,
            }
            annotation_list.append(annotation)
        annotations[img_file] = annotation_list
    return annotations


def process_classification_txt_files(txt_dir, image_dir=None):
    annotations = {}
    txt_path = Path(txt_dir)

    txt_files = list(txt_path.glob("*.txt"))
    if not txt_files:
        print(f"在目录 {txt_dir} 中未找到任何TXT文件\n")
        return annotations

    for txt_file in txt_files:
        filename = txt_file.stem
        img_file = list(Path(image_dir).glob(f"{filename}.*"))[0].name
        img_file = str(img_file)

        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        annotation_list = []
        for line in lines:
            parts = line.strip().split()
            class_id, score = map(float, parts[:2])
            annotation = {
                "score": score,
                "class_id": int(class_id),
            }
            annotation_list.append(annotation)
        annotations[img_file] = annotation_list
    return annotations


def process_keypoint_txt_files(txt_dir, image_dir=None):
    annotations = {}
    txt_path = Path(txt_dir)
    if not txt_path.is_dir():
        raise NotADirectoryError(f"指定的txt_dir '{txt_dir}' 不是一个目录或不存在\n")

    txt_files = list(txt_path.glob("*.txt"))
    if not txt_files:
        print(f"在目录 {txt_dir} 中未找到任何TXT文件\n")
        return annotations

    for txt_file in txt_files:
        filename = txt_file.stem
        img_path = list(Path(image_dir).glob(f"{filename}.*"))[0]
        img_file = img_path.name

        with Image.open(img_path) as img:
            img_width, img_height = img.size

        img_file = str(img_file)
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        need_normalize = False
        for line in lines:
            parts = line.strip().split()
            if max(map(float, parts)) > 1:
                need_normalize = True
                break

        keypoints_x = []
        keypoints_y = []
        keypoints_score = []
        for line in lines:
            parts = line.strip().split()
            has_score = len(parts) % 3 == 0

            if has_score:
                x, y, score = map(float, parts)
                keypoints_score.append(round(float(score), 2))
            else:
                x, y = map(float, parts)

            if need_normalize:
                x /= img_width
                y /= img_height

            keypoints_x.append(round(x, 4))
            keypoints_y.append(round(y, 4))

        keypoint_entry = {
            "keypoints_x": keypoints_x,
            "keypoints_y": keypoints_y,
            "keypoints_score": keypoints_score,
        }
        annotations[img_file] = keypoint_entry
    return annotations


def update_json(data, args, annotations, json_status, task):
    if json_status == "new":
        if args.model_id == "" or args.model_name == "":
            raise ValueError("json数据首次建立，model_id, model_name不能为空")
    if args.model_id:
        data["model_id"] = args.model_id
    if args.model_name:
        data["model_name"] = args.model_name
    if task == "detection":
        data["bbox_threshold"] = args.bbox_threshold
        data["score_threshold"] = args.score_threshold
    elif task == "classification":
        data["score_threshold"] = args.score_threshold
    elif task == "keypoint":
        data["score_threshold"] = args.score_threshold
        data["position_threshold"] = args.position_threshold
    else:
        raise ValueError("未识别的task")
    img_dir = Path(args.image_dir).resolve()
    data["image_dir"] = img_dir.name
    data[args.chip] = annotations
    return data


def main():
    args = parse_arguments()
    data, json_status = load_json(args.json_path)

    if args.task == "detection":
        annotations = process_detection_txt_files(args.txt_dir, args.image_dir)
    elif args.task == "classification":
        annotations = process_classification_txt_files(args.txt_dir, args.image_dir)
    elif args.task == "keypoint":
        annotations = process_keypoint_txt_files(args.txt_dir, args.image_dir)
    else:
        print("未识别的task\n")
        return

    if not annotations:
        print("未找到任何注释信息可供添加或更新\n")
        return

    updated_data = update_json(data, args, annotations, json_status, args.task)

    save_json(updated_data, args.json_path, json_status)


if __name__ == "__main__":
    main()
