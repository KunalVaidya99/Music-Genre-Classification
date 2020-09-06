# Music-Genre-Classification

Music Genre Classification using GTZAN Dataset.Here is the Link to Dataset [GTZAN Dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

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

Each Genre contains 100 audio files each of duration 30s.The features are extracted by dividing the 30s file in 10 files 3s as this will create more data.
The Data is stored in features_3_sec.csv.This Data Augmentation really helps model learn better and it performes better on the test set.



![Training Loss Plot](https://github.com/KunalVaidya99/Music-Genre-Classification/blob/master/musicgenre.PNG)

So this the training and test accuracy obtained after 1500 epochs.
A simple DNN model is also showing quite good test accuracy due to data augmentation
