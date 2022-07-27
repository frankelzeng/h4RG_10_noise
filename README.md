# Introduction
This is the repository for pixel classification project. The main goal is to classify the light curves of 4096×4096 pixels in Roman Space Telescope and identify three types of anomalous pixels by their time series behaviors using Fully Convolutional Neural Network in Tensorflow. The normal pixels are labeled as "type0". The three types of anomalous pixels are RTN pixels with two  (type1), Comic hit pixels (type2), and dark current pixels (type3). We aim to train the fCNN model in order to classify these four types and help boosting the efficiency of image processing in the pipeline of Roman Space Telescope analysis.
# Contribution
Chenxiao Zeng (The Ohio State University), Weiyao Wang (Johns Hopkins University), Chris Hirata (The Ohio State University)
# Structure
The code is organized as following:
gen_df.py in root directory is the code to generate training and testing set for pixels
/tensorflow)dl/cnn_keras.py and
/tensorflow_dl/cnn_keras.ipynb 
are the two main files to build, train and evaluate the fCNN model. Currently it yields 0.91 f-score (weighted over all four types). This model is powerful enough to distinguish type0 and type3 but requires a better performance on type1 and type2.
