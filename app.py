import streamlit as st
from PIL import Image
from multimodel import MultiModalSearch

st.set_page_config(layout="wide")

image_data_path = "images"

def resize_image(image, width=400, height=400):
    image = Image.open(image)
    return image.resize((width, height))


def main():
    st.markdown("<h1 style = 'text-align:center;'>Text To Image Recommendation System</h1>",unsafe_allow_html=True)
    multimodalsearch = MultiModalSearch(document_directory=image_data_path)

    query = st.text_input("Enter you query")

    if st.button("Search") and len(query) > 0:
        st.info(f'Query : **{query.strip()}**') 
        results = multimodalsearch.search(query=query)
        st.subheader("Results")
        col1,col2,col3 = st.columns(3)

        with col1:
            resized_img1 = resize_image(results[0].content)
            st.image(resized_img1, use_column_width=True)  
            
        with col2:
            resized_img2 = resize_image(results[1].content)
            st.image(resized_img2, use_column_width=True)

        with col3:
            resized_img3 = resize_image(results[2].content)
            st.image(resized_img3, use_column_width=True)

    else:
        st.warning("Please Enter the query .......")



if __name__ == "__main__":
    main()