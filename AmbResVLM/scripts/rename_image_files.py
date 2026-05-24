"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import re
import shortuuid
from ambres import DATA_DIR


def is_valid_shortuuid(shortuuid: str) -> bool:
    return bool(re.fullmatch(r"[2-9A-HJ-NP-Za-km-z]{22}", shortuuid))


def main():
    image_files = sorted([image_path for image_path in DATA_DIR.AMBRES.iterdir()])
    for idx in range(len(image_files)):
        if image_files[idx].suffix != ".jpeg":
            raise ValueError(f"{image_files[idx]} is not a jpeg image!")
        if not is_valid_shortuuid(str(image_files[idx].stem)):
            valid_name = f"{shortuuid.uuid()}.jpeg"
            print(f"Renaming {image_files[idx]} with {image_files[idx].parent / valid_name}")
            image_files[idx].rename(image_files[idx].parent / valid_name)


if __name__ == "__main__":
    main()
