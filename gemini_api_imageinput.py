from google import genai

from google.genai import types

import json

from PIL import Image, ImageDraw, ImageFont

def generate_image(boundingboxes):
    image = Image.open('pillar.jpg')  # Load the original image
    draw = ImageDraw.Draw(image)
    print("Image Size:", image.size)
    img_width, img_height = image.size
    scale_x = img_width / 1000
    scale_y = img_height / 1000

    for obj in boundingboxes:
        box = obj.get("box_2d")
        label = obj.get("label", "Unknown") # either label, or unknown

        if box:
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

    image.save("output_with_boxes.jpg")
    image.show()

client = genai.Client(api_key='YOUR KEY')
IMAGE_MIME_TYPE = 'image/jpeg'

with open('bollard.jpg', 'rb') as f:
    ref_image_bytes= f.read()

with open('pillar.jpg', 'rb') as f:
    scene_image_bytes= f.read()

response_text = ""

for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash',
    contents=[
    'In the image, identify all instances of the reference image given to you with a bollard of alternating black and white horizontal stripes. These pillars are short (waist height or lower), square or rectangular in shape, and usually placed at the corners or sides of roads or structures. Do NOT detect shadows, fences, poles, or any elongated structures. Provide their bounding box coordinates of the matching in the second scene image in [x1, y1, x2, y2] format (top-left and bottom-right corners). If possible, also provide the object label. Respond ONLY in valid JSON. Do NOT add any extra commentary or text. Your response should be a JSON array directly.',
    types.Part.from_bytes(data=ref_image_bytes,mime_type=IMAGE_MIME_TYPE),
    types.Part.from_bytes(data=scene_image_bytes,mime_type=IMAGE_MIME_TYPE),
    ],
):

    response_text += chunk.text
    #print(chunk.text, end='')

# convert to JSON
clean_response = response_text.strip().split("```json")[-1].split("```")[0].strip()
print(clean_response)
bbox_data = json.loads(clean_response)
generate_image(bbox_data)
