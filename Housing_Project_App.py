#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import pandas as pd
from Housing_Project import *

# In[ ]:

st.write("# House Price Prediction")

# In[ ]:

beds_input = int(st.number_input("Enter the number of beds", min_value=1, step=1))
baths_input = int(st.number_input("Enter the number of baths", min_value=1, step=1))
area_input = int(st.number_input("Enter the area in sqft", min_value=100, step=1))
year_built_input = int(st.number_input("Enter the year built", min_value=1000, step=1))
user_input = st.text_area("Enter the description")

# In[ ]:

# Preprocess user input
preprocessed_input = preprocess_text(user_input)

# Convert the User Input into Word Embeddings
input_embedding_df = generate_word_embeddings([preprocessed_input])

# Using TfidfVectorizer
tfidf_input = [" ".join(preprocessed_input)]
tfidf_features_df = extract_tfidf_features(tfidf_input)

# Using CountVectorizer
bow_input = [" ".join(preprocessed_input)]
bow_features_df = extract_bow_features(bow_input)

# Create separate DataFrames for each feature type
input_features_df = pd.DataFrame({
    'beds': [beds_input],
    'baths': [baths_input],
    'area': [area_input],
    'yearBuilt': [year_built_input]
})

# Add a prefix to the column names of each DataFrame to make them unique
input_embedding_df.columns = 'emb_' + input_embedding_df.columns.astype(str)
tfidf_features_df.columns = 'tfidf_' + tfidf_features_df.columns.astype(str)
bow_features_df.columns = 'bow_' + bow_features_df.columns.astype(str)

# Ensure that the columns match with the NLP features in training data
input_embedding_df = input_embedding_df.reindex(columns=word_embeddings_df.columns, fill_value=0)
tfidf_features_df = tfidf_features_df.reindex(columns=tfidf_df.columns, fill_value=0)
bow_features_df = bow_features_df.reindex(columns=bow_df.columns, fill_value=0)

# Concatenate all the feature DataFrames
combined_features_df = pd.concat([input_features_df, input_embedding_df, tfidf_features_df, bow_features_df], axis=1)

# Filter out any columns that are not present in the training data
combined_features_df = combined_features_df[combined_data.columns]

# Load the XGBoost model from the saved file
xgboost_model = joblib.load('xgboost_model.pkl')

# ...

if st.button('Predict'):
    # Convert the DataFrame to a DMatrix for prediction
    input_features_dmatrix = xgb.DMatrix(combined_features_df)

    # Use the trained model to predict the price using the DMatrix object
    predicted_price = xgboost_model.predict(input_features_dmatrix)

    # Round the predicted price to two decimal places
    rounded_predicted_price = np.around(predicted_price, 2)

    st.success(f'Predicted Price: ${rounded_predicted_price[0]:.2f}')
    







# %%
