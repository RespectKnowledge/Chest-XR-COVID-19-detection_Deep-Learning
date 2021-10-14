# Chest-XR-COVID-19-detection_Deep-Learning
Please download the training, validation and testing dataset from the following link
# https://cxr-covid19.grand-challenge.org/

# Training function

If you want to train the deep convolutional neural networks, please use this code,

# for EfficieNet pretrained model

efficientnet_model.py 

# For pretrained models

DDN_covid_models.py 

If you use vision transformer based deep learning models, please use this code for training and validation

# For vision-based transformers

python3 Transformers_models_covid.py

# Prediction fucntion

python3 covid_prediction.ipynb

# Please cite this paper, if you use this dataset
@misc{CovidGrandChallenge2021,
                author = {Akhloufi, Moulay A. and Chetoui, Mohamed},
                title = {{Chest XR COVID-19 detection}},  
                howpublished = {\url{https://cxr-covid19.grand-challenge.org/}},
                month = {August},
                year = {2021},
                note = {Online; accessed September 2021},
                 }
