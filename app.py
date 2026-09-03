Import streamlit as st
from PIL import Image



st.set_page_config(
  page_title="AI Traffic Sign Recognition",
  page_icon="🚦"
)



st.title("🚦 AI Traffic Sign Recognition")
st.write(
  "Upload a traffic sign image and the AI model will "
  "identify the sign and recommend an autonomous vehicle action."
)

uploaded_file = st.file_uploader(
  "Upload a traffic sign image",
  type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file)

  st.image(image, caption="Uploaded Traffic Sign")

  st.success("Image uploaded successfully!")

  st.write("### AI Prediction")
  st.info("Model will be connected in the next development step.") 

