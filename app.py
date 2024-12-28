import streamlit as st
from multimodel import MultiModalSearch

st.set_page_config(layout="wide")



def main():
    st.markdown("<h1 style = 'text-align:center; color:brown'>Image Recommendation system</h1>",unsafe_allow_html=True)
    multimodalsearch = MultiModalSearch()

    query = st.text_input("Enter you query")

    if st.button("Search"):
        if len(query) > 0 :
            results = multimodalsearch.search(query=query)
            st.info(f'Query : **{query}**')
            st.subheader("Results")
            col1,col2,col3 = st.columns(3)

            with col1:
                st.write(f"Score : {round(results[0].score*100,2)}")
                st.image(results[0].content, use_container_width=True)

            with col2:
                st.write(f"Score : {round(results[1].score*100,2)}")
                st.image(results[1].content, use_container_width=True)


            with col3:
                st.write(f"Score : {round(results[2].score*100,2)}")
                st.image(results[2].content, use_container_width=True)

        else:
            st.warning("Please Enter the query .......")



if __name__ == "__main__":
    main()