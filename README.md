# IVP_Emotion_detection

The IVP_obj_detection.py contains the code for loading a YouTube video 
and detecting the emotions of the people present in the video. It also assigns 
every human present in the video a unique number and tracks them accross 
multiple frames.

The model.py contains the model for emotion detection.

The xls files contain emotions tracked for various humans using random YouTube videos and this was 
used to train the hidden Markov model .

The create_dataset.py was used to generate all of the xls files .

The read_vals.py was used to read all the generated datasets and create the emission and the state change matrix
that was used to train the hidden Markov model.

