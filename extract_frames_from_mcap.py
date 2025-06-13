import os
import sys
import capnp
import cv2
import numpy as np
from mcap.reader import make_reader

# Step 1: Import your image.capnp schema from vk_sdk
# sys.path.append(os.path.join(os.path.dirname(__file__), "./vk_sdk/capnp"))
sys.path.append('/opt/vilota/vk_sdk/capnp')
sys.path.append('/opt/vilota/messages')
capnp.add_import_hook()
import image_capnp as eCALImage

# Step 2: Decode image using vk_sdk's schema
def decode_image_msg(msg):
    with eCALImage.Image.from_bytes(msg) as img:
        encoding = img.encoding
        width = img.width
        height = img.height
        data = img.data

        if encoding == "mono8":
            return np.frombuffer(data, dtype=np.uint8).reshape((height, width))
        elif encoding == "mono16":
            return np.frombuffer(data, dtype=np.uint16).reshape((height, width)) 
        elif encoding == "bgr8":
            return np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        elif encoding in ["jpeg", "png"]:
            return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError(f"Unsupported image encoding: {encoding}")

# Step 3: Main frame extraction logic
def process_mcap_to_frames(mcap_path, output_dir):
    os.makedirs(os.path.join(output_dir, "left"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "right"), exist_ok=True)

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)

        for schema, channel, message in reader.iter_messages():
            topic = channel.topic

            if topic in ["S1/stereo1_l", "S1/stereo2_r"]:
                print(f"🔍 Processing message from {topic}...")

                try:
                    img = decode_image_msg(message.data)
                    if img is not None:
                        timestamp = message.log_time
                        direction = "left" if topic == "S1/stereo1_l" else "right"
                        filename = os.path.join(output_dir, direction, f"{direction}_{timestamp}.png")
                        cv2.imwrite(filename, img)
                        print(f"Saved: {filename}")
                    else:
                        print("Failed to decode image")

                except Exception as e:
                    print(f"Error processing message: {e}")

# Step 4: Call the function with your file
if __name__ == "__main__":
    process_mcap_to_frames("videos/forklift.mcap", "frames")
