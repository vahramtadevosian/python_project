import argparse
import sys

import mediapipe as mp
import streamlit as st
from PIL import Image as PilImage
from lightly.data import LightlyDataset
from torch.utils.data import DataLoader

from tools.dataset import LightlyDatasetWithMasks
from tools.simple_clr import SimCLRModel
from utils.helper_functions import yaml_loader, generate_embeddings, create_test_transforms, \
    plot_knn_examples_for_uploaded_image


parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int,
                    help='Input size of image', default=128)
parser.add_argument('--batch_size', type=int,
                    help='Batch size', default=128)
parser.add_argument('--num_workers', type=int,
                    help='Number of workers', default=8)
parser.add_argument('--use_masks', type=bool,
                    help='Whether to use segmentation masks', default=True)
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

    num_neighbors = st.sidebar.slider('Number of nearest neighbors', min_value=1, max_value=5, value=3)

    # Load the test dataset
    test_transforms = create_test_transforms(resolution=args.input_size)

    if args.use_masks:
        print('using')
        dataset_test = LightlyDatasetWithMasks(
            input_dir=general_dict['path_to_test_data'],
            mask_dir=general_dict['path_to_mask'],
            transform=test_transforms,
            test_mode=True
        )
    else:
        print('not using')
        dataset_test = LightlyDataset(
            input_dir=general_dict['path_to_test_data'],
            transform=test_transforms
        )

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    query_filename = uploaded_image.name
    input_image = PilImage.open(uploaded_image)
    input_image = input_image.resize((args.input_size, args.input_size))

    model = SimCLRModel.load_from_checkpoint(general_dict['checkpoint_path'])
    model.eval()
    embeddings, filenames = generate_embeddings(model.cpu(), dataloader_test)

    st.image(input_image, caption="Uploaded Image", use_column_width=False)

    # Transform and forward the uploaded image through the model
    uploaded_image_tensor = test_transforms(input_image).unsqueeze(0)
    uploaded_image_embedding = model.backbone(uploaded_image_tensor.to(model.device)).flatten(start_dim=1)

    # Plot and display the nearest neighbor images
    fig = plot_knn_examples_for_uploaded_image(embeddings, filenames, general_dict['path_to_test_data'],
                                               query_filename, n_neighbors=num_neighbors,
                                               save_path=general_dict['plt_figures'])
    st.pyplot(fig)
