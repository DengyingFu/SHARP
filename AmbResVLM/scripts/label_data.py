"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import os
import json
import pathlib
import streamlit as st
from PIL import Image
from ambres.dataset import DataHandler


def main():
    mode = "test"
    data_handler = DataHandler(env="real", mode=mode)
    image_files = sorted(
        [f for f in os.listdir(data_handler.images_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    )
    st.set_page_config(layout="wide")
    st.title("Data Labeling App")
    if "image_index" not in st.session_state:
        st.session_state.image_index = 0
    current_image = image_files[st.session_state.image_index]
    img_id = pathlib.Path(current_image).stem
    image_path = os.path.join(data_handler.images_dir, current_image)

    col1, col2, col3 = st.columns(3)
    with col1:
        with Image.open(image_path) as image:
            image = image.reduce(4)
            progress_str = f"({st.session_state.image_index+1}/{len(image_files)}) \t-\t"
            st.image(image, caption=progress_str + current_image)  # , width=500)

        # Display navigation buttons
        left, right = st.columns((1, 3))
        with left:
            if st.button("Previous", disabled=st.session_state.image_index == 0):
                st.session_state.image_index -= 1
                st.session_state.locations = []
                st.rerun()
        with right:
            if st.button("Next", disabled=st.session_state.image_index == len(image_files) - 1):
                st.session_state.image_index += 1
                st.session_state.locations = []
                st.rerun()

    with col2:
        with st.form(key="data_form"):
            obj_list = st.text_area("OBJ_LIST -> list[str]")
            ambiguity_map = st.text_area("AMBIGUITY_MAP -> dict[str, list[str]]")
            tasks = st.text_area("TASKS -> list[str] (only multi-object ones)")
            save_button = st.form_submit_button("Save")
            if save_button:
                try:
                    parsed_obj_list = json.loads(obj_list)
                except json.JSONDecodeError:
                    st.error("Invalid JSON data in OBJ_LIST!")
                try:
                    parsed_tasks = json.loads(tasks)
                except json.JSONDecodeError:
                    st.error("Invalid JSON data in TASKS!")
                try:
                    parsed_ambiguity_map = json.loads(ambiguity_map)
                except json.JSONDecodeError:
                    st.error("Invalid JSON data in AMBIGUITY_MAP!")
                sample = {
                    "id": img_id,
                    "obj_list": parsed_obj_list,
                    "tasks": parsed_tasks,
                    "ambiguity_map": parsed_ambiguity_map,
                }
                data_handler.append_sample(sample)
                st.rerun()

    with col3:
        with st.container(height=1000):
            # Display the data for the current image
            sample = data_handler.get_data_for_image(img_id)
            st.write(sample)


if __name__ == "__main__":
    main()
