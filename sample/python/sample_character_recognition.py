import sys
import os
from tdl import nn,image

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_character_recognition.py <model_path> <image_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    character_recognitor = nn.CharacterRecognitor(nn.ModelType.RECOGNITION_LICENSE_PLATE, model_path)
    img = image.read(img_path)
    chars = character_recognitor.inference(img)
    print(chars)

    