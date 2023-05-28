import argparse
import sys

import mediapipe as mp
import streamlit as st
from PIL import Image as PilImage
from utils.helper_functions import yaml_loader, \
    plot_knn_examples_for_uploaded_image, infer

parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int,
                    help='Input size of image', default=128)
parser.add_argument('--batch_size', type=int,
                    help='Batch size', default=128)
parser.add_argument('--num_workers', type=int,
                    help='Number of workers', default=8)
args = parser.parse_args()

general_dict = yaml_loader('configs/general.yaml')

st.set_option('deprecation.showPyplotGlobalUse', False)  # Hide the warning

mp_drawing = mp.solutions.drawing_utils

st.title('Person identification App')

st.sidebar.title('Find your photos')
st.sidebar.subheader('Welcome!')

app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['About App', 'Find Similar Image'])

if app_mode == "About App":
    st.markdown('In this application we are using **SimCLR** for **Person Identification**   \n'
                'Python Midterm Project, ASDS, YSU    \n'
                '   \n'
                'Team Members:   \n'
                'Susanna Sargsyan,  \n'
                'Anush Gevorgyan,    \n'
                'Mariam Gasoyan,   \n'
                'Vahram Tadevosyan')


elif app_mode == 'Find Similar Image':
    uploaded_image = st.file_uploader("Please upload your image", type=["jpg", "png"])
    if uploaded_image is None:
        sys.exit()  # TODO retry

    num_neighbors = st.sidebar.slider('Number of nearest neighbors', min_value=1, max_value=10, value=3)
    use_masks = st.sidebar.checkbox("Use binary masks")

    query_filename = uploaded_image.name
    input_image = PilImage.open(uploaded_image)
    input_image = input_image.resize((args.input_size, args.input_size))

    embeddings, filenames = infer(use_masks, args, **general_dict)

    st.image(input_image, caption="Uploaded Image", use_column_width=False)

    # Plot and display the nearest neighbor images
    fig = plot_knn_examples_for_uploaded_image(embeddings, filenames, general_dict['path_to_test_data'],
                                               query_filename, n_neighbors=num_neighbors,
                                               save_dir=general_dict['plt_figures'])
    st.pyplot(fig)
