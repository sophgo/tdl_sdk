import sys
import os
from tdl import nn,image

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_face_attribute.py <model_path> <image_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    face_attribute = nn.get_model(nn.ModelType.CLS_ATTRIBUTE_GENDER_AGE_GLASS_MASK, model_path)
    img = image.read(img_path)
    face_attribution = face_attribute.inference(img)
    result = face_attribution[0]
    print(result["mask_score"])
    print(result["is_wearing_mask"])
    print(result["gender_score"])
    print(result["is_male"])
    print(result["age_score"])
    print(result["age"])
    print(result["glass_score"])