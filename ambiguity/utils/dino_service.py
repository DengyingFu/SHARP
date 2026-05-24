import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from typing import List
import numpy as np
from PIL import Image
import io
import os
import sys
import json

# Add parent directory to path to import api_wrappers
# This assumes the service is run from the root of the AmbitiousRobo project
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ambiguity.utils.api_wrappers import DinoWrapper, Owlv2Wrapper

app = FastAPI(title="Dino Service")

# In a real implementation, you would load the model here.
# We initialize it conditionally to avoid double loading when running as a script.

dino_wrapper = None

if __name__ != "__main__":
    # When running via 'uvicorn module:app', __name__ is not __main__
    # So we initialize using environment variables here.
    model_type = os.getenv("DETECTION_MODEL", "grounding_dino")
    if model_type == "owlv2":
        dino_wrapper = Owlv2Wrapper(config_path='ambiguity/configs/models.yaml')
        print("Owlv2 model wrapper initialized for the service (env).")
    else:
        dino_wrapper = DinoWrapper(config_path='ambiguity/configs/models.yaml')
        print("DINO model wrapper initialized for the service (env).")

def convert_detections_to_dict(detections):
    """
    Convert Detections object to a JSON-serializable dictionary.
    Handles various detection object types from different libraries.
    """
    if detections is None:
        return None
    
    # Check if it's already a dict or list
    if isinstance(detections, (dict, list)):
        return detections
    
    # Try to convert Detections object to dict
    try:
        # For supervision Detections or similar objects
        if hasattr(detections, 'xyxy'):
            result = {
                'xyxy': detections.xyxy.tolist() if isinstance(detections.xyxy, np.ndarray) else detections.xyxy,
            }
            if hasattr(detections, 'confidence'):
                result['confidence'] = detections.confidence.tolist() if isinstance(detections.confidence, np.ndarray) else detections.confidence
            if hasattr(detections, 'class_id'):
                result['class_id'] = detections.class_id.tolist() if isinstance(detections.class_id, np.ndarray) else detections.class_id
            if hasattr(detections, 'mask'):
                result['mask'] = detections.mask.tolist() if isinstance(detections.mask, np.ndarray) else detections.mask
            return result
        
        # Try to convert to dict if it has __dict__
        if hasattr(detections, '__dict__'):
            return {k: v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in detections.__dict__.items() 
                    if not k.startswith('_')}
    except Exception as e:
        print(f"Warning: Could not fully convert detections object: {e}")
        return str(detections)
    
    return str(detections)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

@app.post("/get_object_localizations", summary="Detect objects and get masks")
async def get_object_localizations_endpoint(
    object_nouns: str = Form(..., description="Comma-separated list of object nouns to detect."),
    image: UploadFile = File(..., description="Image file to process."),
    results_dir: str = Form(None, description="Directory to save results (optional).")
):
    image_data = await image.read()
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # The form sends a single string, so we split it into a list.
    nouns = [noun.strip() for noun in object_nouns.split(',')]
    
    print(f"Service received request to detect: {nouns}")
    
    detections, mask, object_names = dino_wrapper.get_object_localizations(pil_image, nouns, results_dir)
    
    # Convert detections to JSON-serializable format
    detections_dict = convert_detections_to_dict(detections)
    
    # Convert mask to list if it's a numpy array
    if isinstance(mask, np.ndarray):
        mask_list = mask.tolist()
    else:
        mask_list = mask
    
    response_data = {
        "detections": detections_dict,
        "mask": mask_list,
        "object_names": object_names
    }
    
    return response_data


@app.post("/get_scene_graph", summary="Generate scene graph from detections")
async def get_scene_graph_endpoint(
    object_names: str = Form(..., description="Comma-separated list of object names."),
    mask: UploadFile = File(..., description="JSON file containing the segmentation mask."),
    pcd: UploadFile = File(..., description="JSON file containing the point cloud data."),
    image: UploadFile = File(..., description="Image file used for segmentation."),
    output_folder:str = Form(..., description="Comma-separated list of object names."),
):
    # Read and process inputs
    image_data = await image.read()
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    mask_data = await mask.read()
    mask_array = np.array(json.loads(mask_data))
    
    pcd_data = await pcd.read()
    pcd_array = np.array(json.loads(pcd_data))
    
    names = [name.strip() for name in object_names.split(',')]
    
    print(f"Service received request to generate scene graph for: {names}")
    
    # Use a temporary directory for server-side file generation
    
    objects_info, objects_dict = dino_wrapper.get_scene_graph(
        pil_image, pcd_array, mask_array, names, output_folder=output_folder
    )
    
    response_data = {
        "objects_info": objects_info,
        "objects_dict": objects_dict
    }
    
    # Use the custom NumpyEncoder for serialization
    return json.loads(json.dumps(response_data, cls=NumpyEncoder))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start the Dino/Owlv2 Service")
    parser.add_argument("--model", type=str, default="grounding_dino", choices=["grounding_dino", "owlv2"], help="Choose detection model: grounding_dino or owlv2")
    args, unknown = parser.parse_known_args()

    # Initialize based on CLI args (dino_wrapper is None here)
    if args.model == "owlv2":
        dino_wrapper = Owlv2Wrapper(config_path='ambiguity/configs/models.yaml')
        print("Owlv2 model wrapper initialized for the service (CLI).")
    else:
        dino_wrapper = DinoWrapper(config_path='ambiguity/configs/models.yaml')
        print("DINO model wrapper initialized for the service (CLI).")

    # To run this service:
    # From your project root (d:\西电\Ambitious\AmbitiousRobo), execute:
    # python -m uvicorn ambiguity.utils.dino_service:app --reload --port 8000
    print(f"Starting service with {args.model} model ...")
    uvicorn.run(app, host="0.0.0.0", port=5678)
