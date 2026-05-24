"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import matplotlib.pyplot as plt
from PIL import Image
from ambres import DATA_DIR
from ambres.sam2_remote import SamRemote
from ambres.ambres_remote import AmbresRemote

save_img = False

ambres_model = AmbresRemote()
sam_model = SamRemote()

image = Image.open(DATA_DIR.get_dir("real", "test") / "SVyiQ6Fz2kfeWnepc52Nh3.jpeg")
obj_list = ["orange", "small pot"]

ambres_model.reset_chat()
ambres_model.set_image(image)
sam_model.set_image(image)

obj_locations = ambres_model.detect(obj_list)
print(obj_locations)

plt.imshow(image)
for _, coords in obj_locations.items():
    first_detection = coords[0]
    mask = sam_model.query_mask(first_detection)
    plt.plot(*first_detection, "ro")
    plt.imshow(mask[0], alpha=0.4)

# Save the image without axes and white border
plt.axis("off")  # Turn off the axis
if save_img:
    plt.savefig("masked_image.png", bbox_inches="tight", pad_inches=0)
plt.show()
