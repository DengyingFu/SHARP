"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ambres import DEVICE, ASSETS_DIR


class Sam:
    def __init__(self):
        sam_checkpoint = ASSETS_DIR.WEIGHTS / "sam2.1_hiera_large.pt"
        sam_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"  # do not touch this value
        self.sam_predictor = SAM2ImagePredictor(
            build_sam2(str(sam_model_cfg), sam_checkpoint, device=DEVICE)
        )
        return

    def set_sam_image(self, rgb: np.ndarray):
        self.sam_predictor.set_image(rgb)
        return

    def query_sam_points(self, points: np.ndarray, labels: np.ndarray = None) -> np.ndarray:
        """
        Args:
            points.shape: [N, 2]
            labels.shape: [N]
            label should be 1 if it is a positive point, or zero if it is a negative (i.e. outside of the mask) point
        Returns:
            mask: [1, H, W]
        """
        assert points.shape[1] == 2
        if labels is not None:
            assert len(points) == len(labels)
            assert len(labels.shape) == 1
        else:
            labels = np.ones((len(points)))

        # Prediction
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        # Pick best mask
        mask = masks[:1]
        return mask


if __name__ == "__main__":
    sam = Sam()
    image = Image.open(ASSETS_DIR.IMAGES / "5rhU25AdQW4jADxhp8EYuq.jpeg")
    image = np.array(image.convert("RGB"))
    points = np.array([[1384, 1060]])
    sam.set_sam_image(image)
    mask = sam.query_sam_points(points)

    # Visualize
    import matplotlib.pyplot as plt

    plt.imshow(image)
    plt.scatter(*points.T, c="r", s=50)
    plt.imshow(mask[0], alpha=0.5)
    plt.show()

    print("done")
