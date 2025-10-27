import sys
import os
from tdl import nn, image

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python sample_img_face_attribute.py <model_id_name> <model_path> <image_path>")
        sys.exit(1)
    model_id_name = sys.argv[1]
    model_path = sys.argv[2]
    img_path = sys.argv[3]

    model_id = getattr(nn.ModelType, model_id_name)
    face_attribute = nn.get_model(model_id, model_path)
    img = image.read(img_path)
    face_attribution = face_attribute.inference(img)
    result = face_attribution[0]

    if model_id_name == "CLS_ATTRIBUTE_GENDER_AGE_GLASS_MASK":
        print("gender_score:", result["gender_score"])
        print("is_male:", result["is_male"])
        print("age_score:", result["age_score"])
        print("age:", result["age"])
        print("glass_score:", result["glass_score"])
        print("is_wearing_glass:", result["is_wearing_glass"])
        print("mask_score:", result["mask_score"])
        print("is_wearing_mask:", result["is_wearing_mask"])
    elif model_id_name == "CLS_ATTRIBUTE_GENDER_AGE_GLASS":
        print("gender_score:", result["gender_score"])
        print("is_male:", result["is_male"])
        print("age_score:", result["age_score"])
        print("age:", result["age"])
        print("glass_score:", result["glass_score"])
        print("is_wearing_glass:", result["is_wearing_glass"])
    elif model_id_name == "CLS_ATTRIBUTE_GENDER_AGE_GLASS_EMOTION":
        print("gender_score:", result["gender_score"])
        print("is_male:", result["is_male"])
        print("age_score:", result["age_score"])
        print("age:", result["age"])
        print("glass_score:", result["glass_score"])
        print("is_wearing_glass:", result["is_wearing_glass"])
        print("emotion_score:", result["emotion_score"])
        print("emotion:", result["emotion"])
    else:
        print("Unknown model type:", model_id_name)

# input: python sample_img_face_attribute.py <model_id_name> <model_path> <image_path>
# output: gender_score: <value> is_male: <value> age_score: <value> age: <value> glass_score: <value> is_wearing_glass: <value> mask_score: <value> is_wearing_mask: <value>