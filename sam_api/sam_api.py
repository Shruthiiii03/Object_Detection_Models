import os
import torch 
import json
import cv2
import supervision as sv 
import numpy as np 
# from segment_anything import sam_model_registry, SamPredictor 
from groundingdino.util.inference import Model 

# setting up constants
DEVICE = torch.device('cpu')
HOME = os.getcwd()

GROUNDING_DINO_CONFIG_PATH = r"/home/shruthi/mcap_frame_project/sam_api/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = r"/home/shruthi/mcap_frame_project/sam_api/weights/groundingdino_swint_ogc.pth"
# SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")

# load models
grounding_dino_model = Model(
    model_config_path=GROUNDING_DINO_CONFIG_PATH,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
    device="cpu"
)

# sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
# sam_predictor = SamPredictor(sam)

# os.makedirs(os.path.join(HOME, "sam_predictions"), exist_ok=True)

# segment function
# def segment(sam_predictor, image, xyxy):
#     sam_predictor.set_image(image)
#     result_masks = []
#     for box in xyxy: 
#         masks, scores, logits = sam_predictor.predict(
#             box=box,
#             multimask_output=True
#         )
#         result_masks.append(masks[np.argmax(scores)])
#     return np.array(result_masks)

# annotation
CLASSES = [
    "short concrete safety block with diagonal black and white stripes",
    "roadside striped rectangular barrier block",
    "black and white warning bollard near wall",
    "diagonally-striped waist-height concrete post",
    "static roadside obstruction with hazard markings",
    ]
BOX_THRESHOLD = 0.6
TEXT_THRESHOLD = 0.4 

output_dir_name = f"dino_temp{BOX_THRESHOLD}_text{TEXT_THRESHOLD}"
output_dir = os.path.join(HOME, "dino_predictions", output_dir_name)
os.makedirs(output_dir, exist_ok=True)

    #to give better context 
def enhance_class_name(class_names):
    return [f"all {c}s" for c in class_names]    

image_dir = r"/home/shruthi/mcap_frame_project/chosen_dataset"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

for path in image_paths:
    image = cv2.imread(path)
    
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(CLASSES),
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    if len(detections.xyxy) == 0:
        print(f"No bollards detected in {[path]}")
        continue
    
    # convert to masks 
    # detections.mask = segment(
    #     sam_predictor=sam_predictor,
    #     image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    #     xyxy=detections.xyxy
    # )

    # annotate & save 
    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()

    if detections.class_id is None or any(v is None for v in detections.class_id):
        detections.class_id = np.zeros(len(detections), dtype=int)  # default all to class 0

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)

    # output naming 
    base_name = os.path.basename(path).split('.')[0]
    output_img_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
    output_json_path = os.path.join(output_dir, f"{base_name}.json")

    # save image
    cv2.imwrite(output_img_path, annotated_image)
    print(f"✅ Saved: {output_img_path}")    

    # save JSON
    json_data = [
    {
        "box": box.tolist(),  # [x1, y1, x2, y2]
        "class_id": int(cls_id)
    }
    for box, cls_id in zip(detections.xyxy, detections.class_id)
    ]

    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON: {output_json_path}")