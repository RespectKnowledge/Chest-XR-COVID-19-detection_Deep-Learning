# Chest-XR-COVID-19-detection_Deep-Learning

6th Place solution for cxr-covid19 challenge on leaderboard

# https://cxr-covid19.grand-challenge.org/evaluation/challenge/leaderboard/

Please download the training, validation and testing dataset from the following link
# https://cxr-covid19.grand-challenge.org/


# Training function

If you want to train the deep convolutional neural networks, please use this code,

# for EfficieNet pretrained models

> python3 efficientnet_model.py 

# For Deep Learning pretrained models

> python3 DDN_covid_models.py 

If you use vision transformer based deep learning models, please use this code for training and validation

# For Vision-based Transformers

> python3 Transformers_models_covid.py

please note that transformer did not provide optimal solution in our case as compared to deep neural networks.


# The proposed deep learning Model


![Covid_detection122](https://user-images.githubusercontent.com/46267777/137321975-1dec1c7d-4f8e-40c9-a2fe-130b96d4a30f.png)

# Training and optimzation hyperparameters 

![training](https://user-images.githubusercontent.com/46267777/137322899-c4aec187-9953-42a1-9393-6cb17fed3848.png)


# Prediction fucntion

> python3 covid_prediction.ipynb


The training and validation results are shown in these Tables.

![Results](https://user-images.githubusercontent.com/46267777/137322629-ccdf0b28-4189-4564-a163-8dddca3bbfc4.png)


# Please cite this paper, if you use this dataset
@misc{CovidGrandChallenge2021,
                author = {Akhloufi, Moulay A. and Chetoui, Mohamed},
                title = {{Chest XR COVID-19 detection}},  
                howpublished = {\url{https://cxr-covid19.grand-challenge.org/}},
                month = {August},
                year = {2021},
                note = {Online; accessed September 2021},
                 }
