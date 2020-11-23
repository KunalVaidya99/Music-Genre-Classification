# Music-Genre-Classification

Music Genre Classification using GTZAN Dataset.Here is the Link to Dataset [GTZAN Dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

Here is the Link to Deployed App on Amazon EC2 For this Repo  [Music-Genre-Recognition-App](http://ec2-3-134-82-239.us-east-2.compute.amazonaws.com:8501/)

The Dataset contains 10 Genres of music :
## Genres
* Blues
* Classical
* Country
* Disco
* Hip-Hop
* Jazz
* Metal
* Pop
* Reggae
* Rock
## Dataset
* Each Genre contains 100 audio files each of duration 30s.
* The features are extracted by dividing the 30s file in 10 files of 3s as this will create more data.
* The Data is stored in features_3_sec.csv.This Data Augmentation really helps model learn better and it performes better on the test set.


## Results
![Training Loss Plot](https://github.com/KunalVaidya99/Music-Genre-Classification/blob/master/musicgenre.PNG)

So this the training and test accuracy obtained after 250 epochs.
A simple DNN model is also showing quite good test accuracy due to data augmentation
