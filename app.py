import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Load the preprocessed DataFrame
def load_data():
    return pd.read_csv("preprocessed_data.zip")
df = load_data()

# Step 1: TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
textual_embeddings = tfidf_vectorizer.fit_transform(df['combined_textual_features'])

# Step 2: Feature Scaling for Numerical Columns
scaler = MinMaxScaler()
numerical_columns = ['number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'beds']  # Add your numerical columns here
scaled_numerical_features = scaler.fit_transform(df[numerical_columns])

# Combine the scaled numerical features with textual embeddings
combined_features = np.hstack((textual_embeddings.toarray(), scaled_numerical_features))

# Step 3: Cosine Similarity
cosine_sim = cosine_similarity(combined_features)

# Recommendation function without Sentence Transformers
def recommend_listings(query, top_n=5):
    # Calculate the cosine similarity between the query text and all listing embeddings
    query_embedding = tfidf_vectorizer.transform([query])  # Transform the query text into TF-IDF vector
    similarity_scores = cosine_similarity(query_embedding, textual_embeddings)

    # Check if the similarity_scores matrix is empty or has unexpected dimensions
    if similarity_scores.shape[0] == 0:
        raise ValueError("No listings found for the given query.")

    # Get the indices of the top-N most similar listings
    top_indices = np.argsort(similarity_scores[0])[-top_n:][::-1]

    # Retrieve the top-N recommended listings
    recommended_listings = df.iloc[top_indices]

    return recommended_listings

def display_recommendations(recommended_listings):
    # Generate HTML representation of the results
    html_output = "<table>"

    for index, row in recommended_listings.iterrows():
        html_output += "<tr>"
        # Display the image as a clickable link to the listing
        html_output += f"<td><a href='{row['listing_url']}' target='_blank'><img src='{row['picture_url']}' style='width:150px;height:150px;'></a></td>"
        html_output += "<td>"
        html_output += f"<b>Name:</b> {row['name']}<br>"
        html_output += f"<b>Property Type:</b> {row['property_type']}<br>"
        html_output += f"<b>Room Type:</b> {row['room_type']}<br>"
        html_output += f"<b>Review Scores Rating:</b> {row['review_scores_rating']}<br>"
        html_output += f"<b>Neighbourhood:</b> {row['neighbourhood']}<br>"
        # Include other numerical columns in the display here
        html_output += "</td>"
        html_output += "</tr>"

    html_output += "</table>"

    # Display the HTML output
    st.markdown(html_output, unsafe_allow_html=True)

# Function to filter recommendations based on user input
def filter_recommendations(df, bedrooms, beds, min_rating, neighborhood_group, neighborhood):
    filtered_df = df[(df['bedrooms'] >= bedrooms) &
                     (df['beds'] >= beds) &
                     (df['review_scores_rating'] >= min_rating) &
                     (df['neighbourhood_group'] == neighborhood_group) &
                     (df['neighbourhood'] == neighborhood)]
    return filtered_df

# Streamlit App
st.title("AirBnB Berlin Recommendation System")

# Add an input text box for the user to enter their query
query = st.text_input("Search:", " ")

# Create Streamlit widgets for filtering
st.sidebar.header("Filter Options")

# Bedrooms filter
bedrooms_filter = st.sidebar.slider("Number of Bedrooms", min_value=0, max_value=int(df['bedrooms'].max()), value=0)

# Beds filter
beds_filter = st.sidebar.slider("Number of Beds", min_value=0, max_value=int(df['beds'].max()), value=0)

# Review Scores Rating filter
min_review_score = st.sidebar.slider("Minimum Review Scores Rating", min_value=0, max_value=int(df['review_scores_rating'].max()), value=0)

# Neighborhood Group filter
neighborhood_group_filter = st.sidebar.selectbox("Neighborhood Group", df['neighbourhood_group'].unique())

# Neighborhood filter
neighborhood_filter = st.sidebar.selectbox("Neighborhood", df['neighbourhood'].unique())

# Button to trigger recommendations update
update_button = st.sidebar.button("Update Recommendations")

if update_button:
    # Filter recommendations based on user input
    filtered_recommendations = filter_recommendations(df, bedrooms_filter, beds_filter, min_review_score, neighborhood_group_filter, neighborhood_filter)
    
    # Display the filtered recommendations
    display_recommendations(filtered_recommendations)


# Define a button to trigger the recommendation
if st.button("Recommend"):
    top_recommendations = recommend_listings(query, top_n=5)
    display_recommendations(top_recommendations)
