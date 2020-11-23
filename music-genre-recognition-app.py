import streamlit as st
from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, 
                          Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,
                          Dropout)
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img,img_to_array
from bing_image_downloader import downloader

st.write(""" # Music Genre Recognition 
App """)

st.write("Made By Kunal Vaidya")
st.write("**This is a Web App to predict Genre of Music.**")
st.write("On the backend of this Web App a Convolutional Neural Network Model is used.The Model was trained on Mel Spectrogram of Music Files in the GTZAN Dataset.")

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1421217336522-861978fdf33a?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1050&q=80");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

file = st.file_uploader("Please Upload Mp3 Audio File Here",
type=["mp3"])



from PIL import Image
import librosa
import numpy as np
import librosa.display
from pydub import AudioSegment
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class_labels = ['blues',
 'classical',
 'country',
 'disco',
 'hiphop',
 'metal',
 'pop',
 'reggae',
 'rock']

def GenreModel(input_shape = (288,432,4),classes=9):
 
 
  X_input = Input(input_shape)

  X = Conv2D(8,kernel_size=(3,3),strides=(1,1))(X_input)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)
  
  X = Conv2D(16,kernel_size=(3,3),strides = (1,1))(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)
  
  X = Conv2D(32,kernel_size=(3,3),strides = (1,1))(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)

  X = Conv2D(64,kernel_size=(3,3),strides=(1,1))(X)
  X = BatchNormalization(axis=-1)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)

  X = Conv2D(128,kernel_size=(3,3),strides=(1,1))(X)
  X = BatchNormalization(axis=-1)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)

  X = Conv2D(256,kernel_size=(3,3),strides=(1,1))(X)
  X = BatchNormalization(axis=-1)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)

  
  X = Flatten()(X)

  #X = Dropout(rate=0.3)(X)

  #X = Dense(256,activation='relu')(X)

  #X = Dropout(rate=0.4)(X)

  X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=9))(X)

  model = Model(inputs=X_input,outputs=X,name='GenreModel')

  return model

model = GenreModel(input_shape=(288,432,4),classes=9)
model.load_weights("genre.h5")


def convert_mp3_to_wav(music_file):
  sound = AudioSegment.from_mp3(music_file)
  sound.export("music_file.wav",format="wav")

def extract_relevant(wav_file,t1,t2):
  wav = AudioSegment.from_wav(wav_file)
  wav = wav[1000*t1:1000*t2]
  wav.export("extracted.wav",format='wav')

def create_melspectrogram(wav_file):
  y,sr = librosa.load(wav_file,duration=3)
  mels = librosa.feature.melspectrogram(y=y,sr=sr)
  
  fig = plt.Figure()
  canvas = FigureCanvas(fig)
  p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
  plt.savefig('melspectrogram.png')


def download_image():
  filename = file.name
  filename = str.split(filename,".")[0]
  downloader.download(filename + " Spotify", limit=1,  output_dir='/', adult_filter_off=True, force_replace=False, timeout=60)
  return filename


def predict(image_data,model):

  #image = image_data.resize((288,432))
  image = img_to_array(image_data)

  image = np.reshape(image,(1,288,432,4))

  prediction = model.predict(image/255)

  prediction = prediction.reshape((9,)) 


  class_label = np.argmax(prediction)

  
  return class_label,prediction

if file is None:
  st.text("Please upload an mp3 file")
else:
  convert_mp3_to_wav(file)
  extract_relevant("music_file.wav",40,50)
  create_melspectrogram("extracted.wav") 
  image_data = load_img('melspectrogram.png',color_mode='rgba',target_size=(288,432))
  
  filename = download_image()
  st.write("The Song You have Choosen Is " +filename )
  st.image(filename +" Spotify" + "/Image_1.jpg",use_column_width=True)
  st.write("**Play the Song Below if you want!**")
  st.audio(file,"audio/mp3")

  #button = st.button("Predict The Genre of My Music!")
  
  #if(button):
  class_label,prediction = predict(image_data,model)

  st.write("## The Genre of Song is "+class_labels[class_label])
    
  prediction = prediction.reshape((9,)) 
  
  color_data = [1,2,3,4,5,6,7,8,9]
  my_cmap = cm.get_cmap('jet')
  my_norm = Normalize(vmin=0, vmax=9)

  fig,ax= plt.subplots(figsize=(6,4.5))
  ax.bar(x=class_labels,height=prediction,
  color=my_cmap(my_norm(color_data)))
  plt.xticks(rotation=45)
  ax.set_title("Probability Distribution Of The Given Song Over Different Genres")
  
  plt.show()
  st.pyplot(fig)

  #st.text("Probability (0: Blues, 1: Classical, 2: Country,3: Disco,4: Hiphop,5: Metal,6: Pop,7: Reggae,8: Rock")
  #st.write(prediction)
