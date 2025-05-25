import os

# Redirect config to a writable place and prevent /.streamlit usage
os.environ["HOME"] = "/tmp"
os.environ["XDG_CONFIG_HOME"] = "/tmp"

import cv2
import numpy as np
import streamlit as st

def remove_watermark(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use thresholding to isolate watermark (bright areas)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Inpaint the watermark area
    inpainted = cv2.inpaint(image, thresh, 3, cv2.INPAINT_TELEA)
    return inpainted

def main():
    st.title("Watermark Remover (Educational Use Only)")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Original Image', channels="BGR")

        if st.button("Remove Watermark"):
            result = remove_watermark(image)
            st.image(result, caption='Processed Image', channels="BGR")

if __name__ == "__main__":
    main()
