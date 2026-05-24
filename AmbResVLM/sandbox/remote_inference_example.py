"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import matplotlib.pyplot as plt
from PIL import Image
from ambres import ASSETS_DIR
from ambres.sam2_remote import SamRemote
from ambres.ambres_remote import AmbresRemote

model = AmbresRemote()
sam_model = SamRemote()

image = Image.open(ASSETS_DIR.IMAGES / "5rhU25AdQW4jADxhp8EYuq.jpeg")
image = image.reduce(4)  # Reduce the image size if too big
task_description = "move the marker next to the sprite bottle"

model.reset_chat()
output_query = model.handle_query(task_description, image)
if output_query["task_ambiguous"]:
    user_response = input(output_query["clarifying_question"] + "\n")
    output_reply = model.handle_response(user_response)
    obj_list = output_reply["task_objects"]
else:
    obj_list = output_query["task_objects"]

obj_locations = model.detect(obj_list)
print(obj_locations)

sam_model.set_image(image)
plt.imshow(image)
for _, coords in obj_locations.items():
    first_detection = coords[0]
    mask = sam_model.query_mask(first_detection)
    plt.plot(*first_detection, "ro")
    plt.imshow(mask[0], alpha=0.5)
plt.show()
