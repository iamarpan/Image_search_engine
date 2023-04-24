import streamlit as st
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
import os
import sklearn
RESNET_MODEL_PATH = './model/resnet-model';
KnnModel = 'model/KnnModel.pickle'
model = tf.keras.models.load_model('model/resnet-model')
neighbors = pickle.load(open(KnnModel, 'rb'));
filenames = pickle.load(open('./embeddings/file_paths.pickle', 'rb'))
feature_list = pickle.load(open('./embeddings/feature_list.pickle',
                                'rb'))
class_ids = pickle.load(open('./embeddings/class_names.pickle', 'rb'))


def check_similarity(image_features):
  distances,indices = neighbors.kneighbors([image_features])
  print(distances,indices)
  filepaths = []
  for index in indices[0]:
      filepaths.append(filenames[index])
  filepaths=["."+ str(filepath) for filepath in filepaths]
  return filepaths


def extract_features(img_path, model):
    input_shape = (224, 224, 3)
    img = tf.keras.preprocessing.image.load_img(img_path,
                         target_size=(input_shape[0], input_shape[1]))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.resnet50.preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features


def predictions(image_path):
    normalized_vector = extract_features(image_path,model)
    return check_similarity(normalized_vector)


st.set_page_config(
    page_title='UNESCO World Heritage Site Image Search Engine'
)

st.title('UNESCO World Heritage Site Image Search Engine')
st.subheader("Upload a image and see the magic.")
st.markdown(
"""
Welcome to Unesco world heritage site image search engine. Currently we support the following heritages sites. We'll
keep to adding more sites to improve user experience.
- Ajanta Caves
- Alai Darwaza
- Basilica of Bom Jesus
- Charar - E - Sharif
- Char Minar
- Chhota Imambara
- Ellora Caves
- Fatehpur Sikri
- Gateway Of India
- Golden Temple
- Hawa Mahal Palace
- Humayun Tomb
- India Gate
- Iron Pillar
- Jamali kamali Tomb
- Khajuraho
- Lotus Temple
- Mysore Palace
- Qutub Minar
- Sun Temple Konark
- Taj Mahal
- Tanjavur Temple
- Victoria Memorial
"""
)
uploaded_file = st.file_uploader("Choose your heritage site image",type=['jpg','png','JPEG'])
if uploaded_file is not None:
    # To read file as bytes:
    image = plt.imread(uploaded_file)
    st.image(image, caption='Uploaded Image',width=200)
    with open(os.path.join("loaded_image", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    image_path = os.path.join("loaded_image",uploaded_file.name)
    similar_files = predictions(image_path)
    monument_name = similar_files[0].split("/")[-2]
    st.write("Similar Monuments")
    st.write("The monument is {}".format(monument_name))
    st.image(similar_files, width=200)

