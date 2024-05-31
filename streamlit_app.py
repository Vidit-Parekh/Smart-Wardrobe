import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pickle
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Set page configuration
st.set_page_config(page_title="Smart Wardrobe", page_icon=":shirt:", layout="wide")

# Define CSS styles
css = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://cdn.openart.ai/published/ueo80BRd2eKE0y1HoO6O/J0e7nWqi_g-wW_raw.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    opacity: 0.8;
}
[data-testid="stSidebar"] {
    background-color: #e0e0e0;  /* Light gray background */
    border-radius: 10px;
    opacity: 0.8;
}
[data-testid="stSidebarNav"] button {
    background-color: #36597d;  /* Darker blue for buttons */
    color: white;
    border: none;
    border-radius: 5px;
    margin: 5px 0;
    padding: 10px 20px;
    cursor: pointer;
}
[data-testid="stSidebarNav"] button:hover {
    background-color: #29496b;  /* Even darker blue on hover */
}
h1 {
    color: #001F3F;  /* Darker blue for better contrast */
    font-family: 'Georgia', serif;
    text-align: center;
}
input[type=text], .stTextInput > div > input {
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #ccc;
    width: 100%;
    box-sizing: border-box;
}
button {
    background-color: #36597d;  /* Same blue as buttons */
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}
button:hover {
    background-color: #29496b;  /* Even darker blue on hover */
}
.file-uploader {
    padding: 20px;
    border: 2px dashed #ccc;
    border-radius: 5px;
    text-align: center;
}
.rating-star {
    color: #ffc107;  /* Keep gold color for rating */
    font-size: 30px;
}
.stMarkdown a {
    color: #FF5733;  /* Change hyperlink color */
    font-weight: bold;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Create multipage structure
pages = ["Home", "Recommendations", "Feedback"]

if 'page' not in st.session_state:
    st.session_state.page = pages[0]

# Function to toggle sidebar visibility
def toggle_sidebar():
    st.session_state.sidebar_visible = not st.session_state.sidebar_visible

# Initialize sidebar visibility state
if 'sidebar_visible' not in st.session_state:
    st.session_state.sidebar_visible = True

# Navigation function
def navigate_to(page):
    st.session_state.page = page

# Navigation buttons
st.sidebar.title("Navigation")
if st.sidebar.button("Toggle Sidebar"):
    toggle_sidebar()

if st.session_state.sidebar_visible:
    for page in pages:
        if st.sidebar.button(page):
            navigate_to(page)

# Function to classify category
def classify_category(input_item, df):
    df = pd.read_csv('categories.csv')
    default_var = 'casual'
    matching_items = df.loc[df['items'].str.lower() == input_item.lower(), 'category']
    if not matching_items.empty:
        category_name = matching_items.iloc[0]
        return category_name
    else:
        return default_var

# Function to save uploaded file
def save_uploaded_file(uploaded_file, user, option):
    try:
        save_path = os.path.join(user, option, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return f"{uploaded_file.name} saved successfully."
    except PermissionError as e:
        st.error(f"Permission error: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to create the model
def create_model():
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False

    model = tf.keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])

    return model

# Function to extract features
def extract_features(img_path, model):
    if os.path.isfile(img_path):
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            expanded_img_array = np.expand_dims(img_array, axis=0)
            preprocessed_img = preprocess_input(expanded_img_array)
            result = model.predict(preprocessed_img).flatten()
            normalized_result = result / norm(result)
            return normalized_result
        except PermissionError as e:
            st.error(f"Permission error: {e}")
            return None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None
    else:
        st.error(f"{img_path} is not a valid file.")
        return None

# Function to recommend items
def recommend(features, feature_list, k):
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

root = "/home/vidit-parekh/PycharmProjects/test_ipd "
cat_ds = "categories.csv"

# Main logic for each page
if st.session_state.page == "Home":
    st.title("Smart Wardrobe")

    # User input
    user = st.text_input("Enter your name:", st.session_state.get('user', ""))
    option = st.text_input("Enter your Occasion:", st.session_state.get('option', ""))
    dataset_path = os.path.join(root, cat_ds)
    dataset = pd.read_csv(dataset_path)

    button = st.button("Submit")

    if button:
        user_path = os.path.join(root, user)
        option_path = os.path.join(user_path, option)
        os.makedirs(option_path, exist_ok=True)

        input_item = option
        category = classify_category(input_item, dataset_path)
        st.session_state.user = user
        st.session_state.option = option
        st.session_state.category = category
        navigate_to("Recommendations")

elif st.session_state.page == "Recommendations":
    st.title("Recommendations")

    if 'option' not in st.session_state or 'category' not in st.session_state:
        st.write("It seems you have not selected any option or category. Please go back and enter the details.")
    else:
        st.write(f"Recommendation for {st.session_state.option} will be: {st.session_state.category}")

    st.title("Multiple File Uploader")

    uploaded_files = st.file_uploader("Upload multiple files", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            result = save_uploaded_file(uploaded_file, st.session_state.user, st.session_state.option)
        st.write('Images uploaded successfully.')

        model = create_model()

        filenames = []

        user_option_path = os.path.join(st.session_state.user, st.session_state.option)
        if os.path.isdir(user_option_path):
            for file in os.listdir(user_option_path):
                file_path = os.path.join(user_option_path, file)
                if os.path.isfile(file_path):
                    filenames.append(file_path)

        feature_list = []

        if filenames:
            for file in tqdm(filenames):
                features = extract_features(file, model)
                if features is not None:
                    feature_list.append(features)
        else:
            st.write('No images found.')

        user_directory = os.path.join(root, st.session_state.user)
        os.makedirs(user_directory, exist_ok=True)

        feature_list_filename = os.path.join(user_directory, f"{st.session_state.user}.pkl")
        filenames_filename = os.path.join(user_directory, f"{st.session_state.user}_filenames.pkl")

        pickle.dump(feature_list, open(feature_list_filename, 'wb'))
        pickle.dump(filenames, open(filenames_filename, 'wb'))

        feature_list = np.array(pickle.load(open(feature_list_filename, 'rb')))
        filenames = pickle.load(open(filenames_filename, 'rb'))

        photos_length = len(uploaded_files) if uploaded_files else 0

        k = 0
        if photos_length <= 2:
            k = 1
        elif 2 < photos_length < 5:
            k = 2
        elif photos_length >= 5:
            k = 3
        else:
            k = 0
            st.write('Upload at least 1 Image')

        # Initialize session state for storing recommendation indices
        if 'recommendation_indices' not in st.session_state:
            st.session_state.recommendation_indices = []
            st.session_state.k = 0

        if uploaded_files:
            first_file = uploaded_files[0]  # Use the first uploaded file as the seed image
            if k != st.session_state.k:
                st.session_state.recommendation_indices = []
                st.session_state.k = k

            if st.session_state.recommendation_indices is None or len(st.session_state.recommendation_indices) == 0:
                features = extract_features(os.path.join(st.session_state.user, st.session_state.option, first_file.name), model)
                if features is not None:
                    st.session_state.recommendation_indices = recommend(features, feature_list, k)

            recommendation_indices = st.session_state.recommendation_indices

            col1, col2, col3, col4, col5 = st.columns(5)

            if recommendation_indices is not None and len(recommendation_indices) > 0:
                for i in range(k):
                    with locals()[f'col{i+1}']:
                        if recommendation_indices.shape[1] > i:
                            st.image(filenames[recommendation_indices[0][i]], use_column_width='always', output_format='JPEG', caption='Recommendation', width=300)
            else:
                print("Some error occurred in file upload")

    if st.button("Proceed to Feedback"):
        navigate_to("Feedback")

# Feedback page
elif st.session_state.page == "Feedback":
    st.title('Did you find us helpful')

    def star_rating(rating):
        df = pd.read_csv("referral_links.csv")
        if 'category' not in st.session_state:
            st.error("No category found. Please go back and complete the previous steps.")
            return

        if rating == 0 or rating >= 4:
            matching_rows = df.loc[df['category'].str.lower() == st.session_state.category.lower()]

            if not matching_rows.empty:
                random_row = matching_rows.sample(n=1).iloc[0]
                amazon = random_row['amazon']
                ajio = random_row['ajio']
                myntra = random_row['myntra']

                st.markdown(f"[Amazon]({amazon})")
                st.markdown(f"[Ajio]({ajio})")
                st.markdown(f"[Myntra]({myntra})")
            else:
                st.error("No matching category found.")
        elif rating in [1, 2, 3]:
            user_categoryselection = st.text_input("Please specify a category:")
            matching_rows = df.loc[df['items'].str.lower() == user_categoryselection.lower()]

            if not matching_rows.empty:
                random_row = matching_rows.sample(n=1).iloc[0]
                amazon = random_row['amazon']
                ajio = random_row['ajio']
                myntra = random_row['myntra']

                st.markdown(f"[Amazon]({amazon})")
                st.markdown(f"[Ajio]({ajio})")
                st.markdown(f"[Myntra]({myntra})")
            else:
                st.error("No matching category found.")
        else:
            st.write("Rate us higher to get referral links!")
        return '⭐' * int(rating)

    rating = st.slider("Rate us:", 0, 5, value=0)
    st.write(f"You rated us: {star_rating(rating)}")

    # Navigation button to restart sessions and navigate to Home
    if st.button("Go to Home"):
        # Clear session state
        st.session_state.clear()
        # Redirect to Home page
        navigate_to("Home")