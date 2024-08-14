import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Warehouse Location Predictor",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("<h1 style='text-align: center;'>Walmart 🌟</h1>", unsafe_allow_html=True)

# Get the absolute path to the current directory
current_dir = os.getcwd()

# Construct the full paths to the pickle files
model_path = os.path.join(current_dir, 'model.pkl')
scaler_path = os.path.join(current_dir, 'scaler.pkl')

# Load the model and scaler
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

# Streamlit app
st.markdown("<h1 style='text-align: center;'>City Suitability Prediction</h1>", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state['prediction_made'] = False
    st.session_state['predicted_score'] = None

# Create a form for input
with st.form(key='predict_form'):
    feature_1 = st.slider('Population', min_value=100000, max_value=5000000, value=2342868)
    feature_2 = st.slider('Road Quality', min_value=100000, max_value=2000000, value=1122336)
    feature_3 = st.slider('Tier Value', min_value=1, max_value=3, value=3)
    feature_4 = st.slider('Average Income', min_value=10000, max_value=100000, value=31009)
    feature_5 = st.slider('Literacy Rate', min_value=1, max_value=10, value=6)
    feature_6 = st.slider('Railways count', min_value=1, max_value=10, value=3)
    feature_7 = st.slider('Average Land Price (per sq feet)', min_value=1000, max_value=10000, value=3293)
    feature_8 = st.slider('Airport Proximity', min_value=1, max_value=100, value=30)

    # Submit button
    submit_button = st.form_submit_button(label='Predict Suitability')

if submit_button:
    # Prepare the input array for prediction
    new_input = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]])
    new_input = scaler.transform(new_input)
    predicted_score = model.predict(new_input)[0]

    # Save the prediction to session state
    st.session_state['predicted_score'] = predicted_score
    st.session_state['prediction_made'] = True

if st.session_state['prediction_made']:
    # Load your data into pandas DataFrames
    output_df = pd.read_csv('output.csv')  # Replace with your actual file path
    cleaned_df = pd.read_csv('combined_data.csv')  # Replace with your actual file path

    # Clean and prepare the location data
    for df in [output_df, cleaned_df]:
        df['location'] = df['location'].str.lstrip(',')  # Remove leading commas
        df['location'] = df['location'].str.strip()
        df['location'] = df['location'].str.replace(r'\(.*\)', '', regex=True).str.strip()  # Remove trailing characters
        df['location'] = df['location'].apply(lambda x: ', '.join([part.strip() for part in x.split(',')]) if isinstance(x, str) else x)

    # Merge the data based on city names
    merged_df = pd.merge(output_df, cleaned_df[['location', 'lats', 'longs']], on='location', how='left')

    # Convert Average Land Price to Acres
    merged_df['average_land_price'] *= 43560

    # Calculate the difference in suitability score
    merged_df['suitability_diff'] = abs(merged_df['suitability_score'] - st.session_state['predicted_score'])

    # Get the top 10 closest cities
    top_10_cities = merged_df.nsmallest(10, 'suitability_diff')

    # Filter the cities with valid latitude and longitude
    valid_cities = top_10_cities.dropna(subset=['lats', 'longs'])

    # Display the results
    st.subheader("Top 10 cities closest to the predicted suitability score:")
    st.dataframe(top_10_cities[['location', 'suitability_score']])

    if not valid_cities.empty:
        st.write("Rendering map...")
        # Create a folium map centered around the mean location
        m = folium.Map(location=[valid_cities['lats'].mean(), valid_cities['longs'].mean()], zoom_start=5)

        # Add city markers to the map with popups containing all relevant information
        for idx, row in valid_cities.iterrows():
            popup_html = f"""
            <div style="width: 300px; font-family: Arial; font-size: 12px;">
                <h4 style="color: #2A9D8F;">{row['location']}</h4>
                <p><strong>Population:</strong> {row['population']:,}</p>
                <p><strong>Road Quality:</strong> {row['dist_road_qual']}</p>
                <p><strong>Tier Value:</strong> {row['tier_value']}</p>
                <p><strong>Literacy Rate:</strong> {row['literacy_rate']}</p>
                <p><strong>Railways Count:</strong> {row['railways_count']}</p>
                <p><strong>Average Land Price (per acre):</strong> {row['average_land_price']:.2f}</p>
                <p><strong>Airport Proximity:</strong> {row['airport_proximity']}</p>
            </div>
            """

            folium.Marker(
                location=[row['lats'], row['longs']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)

        # Render the map in Streamlit
        st_folium(m, width=1000, height=600)
    else:
        st.write("No valid locations found for mapping.")

    # Plot graphs for each parameter across the top locations
    st.subheader("Parameter Comparisons Across Top Locations")
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 16))  # Increase width for better label spacing
    parameters = {
        'Population': 'population',
        'Road Quality': 'dist_road_qual',
        'Tier Value': 'tier_value',
        
        'Literacy Rate': 'literacy_rate',
        'Railways Count': 'railways_count',
        'Average Land Price (per acre)': 'average_land_price',
        'Airport Proximity': 'airport_proximity'
    }

    for ax, (param_name, param_column) in zip(axes.flat, parameters.items()):
        ax.barh(valid_cities['location'], valid_cities[param_column], color='blue')  # Horizontal bars for better label display
        ax.set_title(param_name)
        ax.set_xlabel(param_name)
        ax.set_ylabel('Location')
        ax.tick_params(axis='y', labelsize=10)  # Adjust font size if necessary

    plt.tight_layout()
    st.pyplot(fig)
