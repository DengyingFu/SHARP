"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import matplotlib.pyplot as plt
from PIL import Image
from ambres import ASSETS_DIR, CKPT
from ambres.ambres_model import AmbresFineTuned


image = Image.open(ASSETS_DIR.IMAGES / "real_0.png")
obj_list = ["blue cup"]

model = AmbresFineTuned(adapter_ckpt=CKPT.REAL)
model.reset_chat()
model.set_image([image])
obj_locations = model.detect_pretty(obj_list)

plt.imshow(image)
for _, coords in obj_locations.items():
    first_detection = coords[0]
    plt.plot(*coords, "ro")
plt.show()
