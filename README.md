# Person Identification with SimCLR

**Python-2 Midterm Project, ASDS, YSU**

This project solves the task of Person Identification using a semi-supervised deep learning approach called [SimCLR](https://arxiv.org/abs/2002.05709) (A Simple Framework for Contrastive Learning of Visual Representations). This method makes use of contrastive learning by maximizing agreement between positive pairs (augmented views of the same image) and minimizing agreement between negative pairs (augmented views of different images). The architecture consists of a ResNet-18 backbone, and a projection head. The latter is only used for the training, while the former is trained to learn embeddings of input images, which are supposed to be close in case of similar images and different otherwise.

Team Members:
- Susanna Sargsyan
- Anush Gevorgyan
- Mariam Gasoyan
- Vahram Tadevosyan

## Setup

1. Clone this repository and enter:
```shell
git clone https://github.com/vahramtadevosian/python_project.git
cd python_project/
```
2. Install requirements using Python 3.9:
```shell
virtualenv --system-site-packages -p python3.9 venv
source venv/bin/activate
pip install -r requirements.txt
```
3. Download the best checkpoint we have trained from [here]() or train the model yourself. Make sure the checkpoint file you want to use for inference is renamed to `best_checkpoint.ckpt` and put in the `./checkpoints` directory.

## Training

1. Put the training data into `./data/train/` and the testing data into `./data/test/`. For CelebA-HQ, download the original images from [here](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view).
2. Optionally, download the segmentation masks of CelebA-HQ from [here](https://drive.google.com/file/d/1u3e3iDyPgZEP5OEsuks3K5GYWyxxkGXy/view?usp=share_link). Make sure to put the training masks in `./mask/train/` and the testing masks in `./mask/test/`.
3. To start the training, run:
```shell
python train.py
```
4. Optionally, you can use the following flags when training:
```shell
python train.py --input_size RES --save_top_k K --batch_size BATCH_SIZE --num_workers N_WORKERS --max_epochs MAX_EPOCHS --log_steps LOG_STEPS
```
* `RES` is the input resolution (we recommend to set this to be at least 128 and a multiple of 32).
* `K` is the number of best checkpoints to save, e.g. if `K` is 3, the top 3 checkpoints are saved during the training.
* `BATCH_SIZE` is the batch size. SimCLR works better with larger batches, so make sure to use the computational resources fully and increase this as much as possible.
* `MAX_EPOCHS` is the number of maximum epochs the model should be trained. Typically, SimCLR is recommended to train longer to achieve better results.
* `LOG_STEPS` is the number of training steps (not epochs) between each logging event.
5. To use the segmentation masks during the training run:
```shell
python train.py --use_masks
```

## Inference with Streamlit

1. To test the trained model make sure that the best checkpoint `best_checkpoint.ckpt` is put in the `./checkpoints` directory.
2. Using streamlit, run:
```shell
streamlit run streamlit_file.py
```
3. Change the `App Mode` to `Find Similar Image` in the left corner of the page.
4. Select the image and check which its nearest neighbors.

## Related Links
* [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ.git)
* [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
* [Lightly Tutorial to Train SimCLR on Clothing](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simclr_clothing.html)
