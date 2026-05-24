import os
import sys
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

# Add project root to path so we can import ambiguity modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ambiguity.modules.perception_module_image_only_local import (
    ScenePerceptionImageOnlyLocal as ScenePerceptionImageOnly,
)


class AnalyzeRequest(BaseModel):
    image_path: str
    results_dir: str
    use_oracle: bool = False
    output_format: str = "v2"
    return_masks: bool = False


class AnalyzeResponse(BaseModel):
    objects: List[Dict[str, Any]]
    relationships: List[Any]
    bbox: List[Any]
    masks: Optional[List[Any]] = None


def _serialize_masks(masks):
    """Convert masks to JSON-serializable lists if possible."""
    if masks is None:
        return None
    serialized = []
    for m in masks:
        if m is None:
            serialized.append(None)
            continue
        if hasattr(m, "tolist"):
            try:
                serialized.append(m.tolist())
                continue
            except Exception:
                pass
        serialized.append(m)
    return serialized


app = FastAPI(title="Scene Perception API", version="1.0.0")

# Instantiate once at startup
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/models.yaml")
try:
    perception = ScenePerceptionImageOnly(config_path=CONFIG_PATH)
except Exception as exc:
    traceback.print_exc()
    raise RuntimeError(f"Failed to initialize ScenePerceptionImageOnly: {exc}")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=400, detail=f"Image not found: {req.image_path}")

    os.makedirs(req.results_dir, exist_ok=True)

    try:
        scene_graph = perception.analyze_scene(
            image_path=req.image_path,
            results_dir=req.results_dir,
            use_oracle=req.use_oracle,
            output_format=req.output_format,
            return_masks=req.return_masks,
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Perception failed: {exc}")

    response: Dict[str, Any] = {
        "objects": scene_graph.get("Objects", []),
        "relationships": scene_graph.get("relationships", []),
        "bbox": scene_graph.get("bbox", []),
    }

    if req.return_masks and "masks" in scene_graph:
        response["masks"] = _serialize_masks(scene_graph.get("masks"))

    return response


@app.post("/analyze_upload", response_model=AnalyzeResponse)
def analyze_upload(
    image: UploadFile = File(..., description="RGB image file"),
    results_dir: str = Form(..., description="Directory to save outputs"),
    use_oracle: bool = Form(False),
    output_format: str = Form("v2"),
    return_masks: bool = Form(False),
):
    # Save uploaded image to results_dir
    os.makedirs(results_dir, exist_ok=True)
    image_path = os.path.join(results_dir, "uploaded_image.png")
    try:
        content = image.file.read()
        with open(image_path, "wb") as f:
            f.write(content)
    finally:
        image.file.close()

    try:
        scene_graph = perception.analyze_scene(
            image_path=image_path,
            results_dir=results_dir,
            use_oracle=use_oracle,
            output_format=output_format,
            return_masks=return_masks,
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Perception failed: {exc}")

    response: Dict[str, Any] = {
        "objects": scene_graph.get("Objects", []),
        "relationships": scene_graph.get("relationships", []),
        "bbox": scene_graph.get("bbox", []),
    }

    if return_masks and "masks" in scene_graph:
        response["masks"] = _serialize_masks(scene_graph.get("masks"))

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
