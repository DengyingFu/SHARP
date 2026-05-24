
import re
import base64
# Helper functions
def load_image_as_base64(image_path):
    with open(image_path, 'rb') as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image



def process_grasping_result(output, text):
    """
    Parses the grasping result output and extracts object ID and class name.

    Supports two formats:
    - "[ID, class name]"  -> Example: "[1, green cylinder]"
    - "[pick object, ID, class name]"  -> Example: "[pick object, 4, blue bolt]"
    """

    # Try first format: [ID, class name]
    match1 = re.search(r'\[(\d+),\s*(.+?)\]', output)
    
    # Try second format: [pick object, ID, class name]
    match2 = re.search(r'\[pick object,\s*(\d+),\s*(.+?)\]', output)

    if match1:
        object_id = int(match1.group(1))
        class_name = match1.group(2).lower()
    elif match2:
        object_id = int(match2.group(1))
        class_name = match2.group(2).lower()
    else:
        class_name = text
        return {"class_name": class_name}
    
    return {
        "selected_object_id": object_id,  # Target object ID
        "class_name": class_name  # Target object class name
    }

