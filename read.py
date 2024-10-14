import gc
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from craft_text_detector import Craft
from PIL import Image
import cv2
import time

craft = Craft(output_dir=None, crop_type="box", cuda=True)

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten')

# Uncomment below lines to use larger, more accurate model
# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

def read(image_path):
    image = cv2.imread(image_path)
    result = craft.detect_text(image_path)
    boxes = result["boxes"]
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    texts = []
    for box in boxes:
        crop = pil_image.crop([box[0][0], box[0][1], box[2][0], box[2][1]])
        pixel_values = processor(crop, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        texts.append(text)
    return texts

image_path = "images/english.png"

start_time = time.time()

texts = read(image_path)
text_data = " ".join(texts)

end_time = time.time()
time_difference = end_time - start_time

print(f"Time: {time_difference} seconds")

print("Data\n")
print(text_data)

craft.unload_craftnet_model()

gc.collect()
torch.cuda.empty_cache()