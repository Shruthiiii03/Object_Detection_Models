# AIzaSyAQhOcA0G4I21CGC2OU7DbY-ssIqz3ebi8

from google import genai

from google.genai import types

import json
import os
import time

from PIL import Image, ImageDraw, ImageFont

configs = [
    # {"temperature": 0.5, "top_p": 0.5, "top_k": 2},
    # {"temperature": 0.7, "top_p": 0.9, "top_k": 40},
    # {"temperature": 0.3, "top_p": 0.95, "top_k": 50},
    # {"temperature": 0.4, "top_p": 0.4, "top_k": 1},
    # {"temperature": 0.2, "top_p": 0.5, "top_k": 2},
    # {"temperature": 0.5, "top_p": 0.6, "top_k": 10},
    # {"temperature": 0.4, "top_p": 0.4, "top_k": 5},
    # {"temperature": 0.5, "top_p": 0.4, "top_k": 4},
    {"temperature": 0.0, "top_p": 0.4, "top_k": 4}
]

# generate bounding boxes 
def generate_image(boundingboxes, image_path, save_path):
    image = Image.open(image_path)  # Load the original image
    draw = ImageDraw.Draw(image)
    print("Image Size:", image.size)
    img_width, img_height = image.size
    scale_x = img_width / 1000
    scale_y = img_height / 1000

    for obj in boundingboxes:
        box = obj.get("box_2d")
        label = obj.get("label", "Unknown") # either label, or unknown

        if box and len(box) == 4:
            x_min, y_min, x_max, y_max = box

            # Transpose
            x_min, y_min = y_min, x_min
            x_max, y_max = y_max, x_max

            # Scaling
            x_min = int(x_min * scale_x)
            x_max = int(x_max * scale_x)
            y_min = int(y_min * scale_y)
            y_max = int(y_max * scale_y)
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            draw.text((x_min, y_min - 15), label, fill="red")
        
        else:
            print("⚠️ Invalid box skipped:", box)

    image.save(save_path)

client = genai.Client(api_key='AIzaSyAQhOcA0G4I21CGC2OU7DbY-ssIqz3ebi8')
IMAGE_MIME_TYPE = 'image/png'
INPUT_FOLDER = "chosen_dataset"
OUTPUT_FOLDER = "gemini_predictions"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


for cfg in configs:
    # loop through configs and save json under there
    folder_name = f"predictions_temp{cfg['temperature']}_topP{cfg['top_p']}_topK{cfg['top_k']}"
    OUTPUT_FOLDER = os.path.join("gemini_predictions", folder_name)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs("output_with_boxes", exist_ok=True)

    # loop through the chosen_dataset 
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".png"):
            image_path = os.path.join(INPUT_FOLDER, filename)
            base_name = os.path.splitext(filename)[0]  # 'ch01' from 'ch01.jpg'

            with open(image_path, 'rb') as f:
                scene_image_bytes= f.read()

            response_text = ""

            for chunk in client.models.generate_content_stream(
                model='gemini-2.0-flash',
                contents=[
                'In the image, identify all instances of small roadside safety pillars with alternating black and white horizontal stripes. These pillars are short (waist height or lower), square or rectangular in shape, and usually placed at the corners or sides of roads or structures. Do NOT detect shadows, fences, poles, or any elongated structures. Provide their bounding box coordinates of the matching in [x1, y1, x2, y2] format (top-left and bottom-right corners). If possible, also provide the object label. Respond ONLY in valid JSON. Do NOT add any extra commentary or text. Your response should be a JSON array directly.',
                types.Part.from_bytes(data=scene_image_bytes,mime_type=IMAGE_MIME_TYPE),
                ],
                config=types.GenerateContentConfig(
                    top_k=cfg["top_k"],
                    top_p=cfg["top_p"],
                    temperature=cfg["temperature"],
                    seed=42,
                ),
            ):

                response_text += chunk.text
                #print(chunk.text, end='')

            # convert to JSON
            clean_response = response_text.strip().split("```json")[-1].split("```")[0].strip()
            print(clean_response)
            bbox_data = json.loads(clean_response)

            output_json_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.json")
            with open(output_json_path, "w") as f:
                f.write(clean_response)
            
            time.sleep(5)

            os.makedirs("output_with_boxes", exist_ok=True)
            generate_image(
                bbox_data,
                image_path=image_path,
                save_path=f"output_with_boxes/{filename}"
            )