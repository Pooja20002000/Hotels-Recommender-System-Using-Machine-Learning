import pickle
import streamlit as st
import numpy as np


st.header('Hotel Recommender System Using Machine Learning')
model = pickle.load(open('artifacts/model.pkl','rb'))
hotels_name = pickle.load(open('artifacts/hotels_name.pkl','rb'))
final = pickle.load(open('artifacts/final.pkl','rb'))
hotel_pivot = pickle.load(open('artifacts/hotel_pivot.pkl','rb'))


def fetch_poster(suggestion):
    hotel_name = []
    ids_index = []
    poster_url = []

    for hotel_id in suggestion:
        hotel_name.append(hotel_pivot.index[hotel_id])

    for name in hotel_name[0]: 
        ids = np.where(final['Hotel_Name'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final.iloc[idx]['image_url']
        poster_url.append(url)

    return poster_url



def recommend_hotel(hotel_name):
    hotels_list = []
    hotel_id = np.where(hotel_pivot.index == hotel_name)[0][0]
    distance, suggestion = model.kneighbors(hotel_pivot.iloc[hotel_id,:].values.reshape(1,-1), n_neighbors=6 )

    poster_url = fetch_poster(suggestion)
    
    for i in range(len(suggestion)):
            hotels = hotel_pivot.index[suggestion[i]]
            for j in hotels:
                hotels_list.append(j)
    return hotels_list , poster_url       



selected_hotels = st.selectbox(
    "Type or select a book from the dropdown",
    hotel_names
)

if st.button('Show Recommendation'):
    recommended_hotels,poster_url = recommend_hotel(selected_hotels)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_hotels[1])
        # st.image(poster_url[1])
    with col2:
        st.text(recommended_hotels[2])
        # st.image(poster_url[2])

    with col3:
        st.text(recommended_hotels[3])
        # st.image(poster_url[3])
    with col4:
        st.text(recommended_hotels[4])
        # st.image(poster_url[4])
    with col5:
        st.text(recommended_hotels[5])
        # st.image(poster_url[5])