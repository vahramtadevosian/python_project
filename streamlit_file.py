import streamlit as st
import splitfolders
import mediapipe as mp
import torch
import argparse
from lightly.data import LightlyDataset
from utils.helper_functions import yaml_loader, generate_embeddings, plot_knn_examples, create_test_transforms, \
    plot_knn_examples_for_uploaded_image
from tools.simple_clr import SimCLRModel
from PIL import Image as PilImage
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)  # Hide the warning

mp_drawing = mp.solutions.drawing_utils

st.title('Face Recognition App')

st.sidebar.title('Find your photos')
st.sidebar.subheader('Welcome!')

app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['About App', 'Run Image'])

if app_mode == "About App":
    st.markdown('In this application we are using **SimCLR** for **Person Identification**   \n'
                'Python Midterm Project, ASDS, YSU    \n'
                '   \n'
                'Team Members:   \n'
                'Susanna Sargsyan,  \n'
                'Anush Gevorgyan,    \n'
                'Mariam Gasoyan,   \n'
                'Vahram Tadevosyan')


elif app_mode == 'Run Image':
    uploaded_image = st.file_uploader("Please upload your image", type=["jpg", "png"])

    num_neighbors = st.sidebar.slider('Number of nearest neighbors', min_value=1, max_value=5, value=3)

    config = yaml_loader('configs/general.yaml')

    # Load the SimCLR model
    model = SimCLRModel.load_from_checkpoint(config['checkpoint_path'])
    model.eval()

    # Load the test dataset
    test_transforms = create_test_transforms(resolution=128)
    dataset_test = LightlyDataset(
        input_dir=config['path_to_test_data'],
        transform=test_transforms
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )

    # Generate embeddings for the test dataset
    embeddings, filenames = generate_embeddings(model.cpu(), dataloader_test)

    general_dict = yaml_loader('configs/general.yaml')

    if uploaded_image is not None:
        query_filename = uploaded_image.name
        input_image = PilImage.open(uploaded_image)
        input_image = input_image.resize((300, 300))

        model = SimCLRModel.load_from_checkpoint(general_dict['checkpoint_path'])
        model.eval()
        # emb_uploaded = model.backbone(input_image).flatten(start_dim=1)
        embeddings, filenames = generate_embeddings(model.cpu(), dataloader_test)
        # plot_knn_examples2(embeddings, filenames, general_dict['path_to_test_data'], query_filename, n_neighbors=3,
        #                    save_path=general_dict['plt_figures'])

        st.image(input_image, caption="Uploaded Image", use_column_width=False)

        # Transform and forward the uploaded image through the model
        uploaded_image_tensor = test_transforms(input_image).unsqueeze(0)
        uploaded_image_embedding = model.backbone(uploaded_image_tensor.to(model.device)).flatten(start_dim=1)

        # Find the nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=num_neighbors).fit(embeddings)
        distances, indices = nbrs.kneighbors(uploaded_image_embedding.cpu().detach().numpy())

        # Plot and display the nearest neighbor images
        fig = plot_knn_examples_for_uploaded_image(embeddings, filenames, general_dict['path_to_test_data'],
                                                   query_filename, n_neighbors=num_neighbors,
                                                   save_path=general_dict['plt_figures'])
        st.pyplot(fig)
