# IVP_Emotion_detection

The IVP_obj_detection.py contains the code for loading a YouTube video 
and detecting the emotions of the people present in the video. It also assigns 
every human present in the video a unique number and tracks them accross 
multiple frames.

The model.py contains the model for emotion detection.

The xls files contain emotions tracked for various humans using random YouTube videos and this was 
used to train the hidden Markov model.

The create_dataset.py was used to generate all of the xls files .

The read_vals.py was used to read all the generated datasets and create the emission and the state change matrix
that was used to train the hidden Markov model.

The pose_detector.py contains the logic for classifying the pose of a human into positive or negative based on the image passed to it.
It uses mediapipe to get the landmarks of a human and use that data to determine the pose.
The model used is present in pose_classifier.py - we used a very simple model that is 8 layers deep with 2 output nodes.

The code for the training is present in the train folder.
Two custom datasets were created for the data loaders, one for the pose classifier and the other for the emotion detection. 
