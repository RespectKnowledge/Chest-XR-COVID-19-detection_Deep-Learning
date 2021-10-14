# Chest-XR-COVID-19-detection_Deep-Learning

6th Place solution for cxr-covid19 challenge on leaderboard

# https://cxr-covid19.grand-challenge.org/evaluation/challenge/leaderboard/

Please download the training, validation and testing dataset from the following link
# https://cxr-covid19.grand-challenge.org/

# Training function

If you want to train the deep convolutional neural networks, please use this code,

# for EfficieNet pretrained model

> python3 efficientnet_model.py 

# For pretrained models

> python3 DDN_covid_models.py 

If you use vision transformer based deep learning models, please use this code for training and validation

# For vision-based transformers

> python3 Transformers_models_covid.py

please note that transformer did not provide optimal solution in our case as compared to deep neural networks.

The proposed solution block diagram is shown below:



![Covid_detection12](https://user-images.githubusercontent.com/46267777/137321737-cc1f7c3f-9f7f-4c46-9b0f-616bd5a43965.png)


# Prediction fucntion

> python3 covid_prediction.ipynb

# Please cite this paper, if you use this dataset
@misc{CovidGrandChallenge2021,
                author = {Akhloufi, Moulay A. and Chetoui, Mohamed},
                title = {{Chest XR COVID-19 detection}},  
                howpublished = {\url{https://cxr-covid19.grand-challenge.org/}},
                month = {August},
                year = {2021},
                note = {Online; accessed September 2021},
                 }
