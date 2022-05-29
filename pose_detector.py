import math
import MediaPipe as mp
import os
import torch
import numpy as np
from pose_classifier import PoseClassification


model = PoseClassification()



# load saved weight into the model
if(os.path.exists("./saved_pose")):
   print("LOADING WEIGHTS INTO THE MODEL")
   model.load_state_dict(torch.load("./saved/saved.pt",map_location=torch.device('cpu')))

model.eval()

# get distance between two coordinates
def findDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist


# Calculate angle between two coorinates.
def findAngle(x1, y1, x2, y2):
    theta = math.acos((y2 - y1)*(-y1) / (math.sqrt(
        (x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180/math.pi)*theta
    return degree


# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def findPose(image):

   keypoints = pose.process(image)


   # Get height and width of the frame.
   h, w = image.shape[:2]


   # Refer to the following link for the different body landmarks :-
   #  https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/pose.py


   # we will use 10 different landmarks

   lm = keypoints.pose_landmarks
   lmPose = mp_pose.PoseLandmark
   # Left shoulder.
   l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
   l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

   # Right shoulder.
   r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
   r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

   # Left hip.
   l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
   l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

   # Right hip.
   r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
   r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)

   # Right elbow.
   r_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
   r_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)

   # Left elbow.
   l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
   l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)

   # Right Heel.
   r_heel_x = int(lm.landmark[lmPose.RIGHT_HEEL].x * w)
   r_heel_y = int(lm.landmark[lmPose.RIGHT_HEEL].y * h)

   # Left Heel.
   l_heel_x = int(lm.landmark[lmPose.LEFT_HEEL].x * w)
   l_heel_y = int(lm.landmark[lmPose.LEFT_HEEL].y * h)

   # Left knee.
   l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
   l_knee_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

   # Right knee.
   r_knee_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
   r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)


   # we are interested in the distances between the various landmarks , for example for an hug
   # we will consider the distance between the sholder to the elbow , it might be different for other actions
   # so we will not take distance between shoulder - elbow , shoulder - hip ,   hip - knee and knee and heel
   # The accuracy of predictions can be improved by considering various other landmark points mentioned
   # in the above link and also calculating the distances between each .


   # distances between shoulder and elbow.
   distance_between_ls_le = findDistance(
      l_shldr_x, l_shldr_y, l_elbow_x, l_elbow_y)
   distance_between_rs_re = findDistance(
      r_shldr_x, r_shldr_y, r_elbow_x, r_elbow_y)

   # distacne between shoulder and hip
   distance_between_ls_lh = findDistance(l_shldr_x, l_shldr_y, l_hip_x, l_hip_y)
   distance_between_rs_rh = findDistance(r_shldr_x, r_shldr_y, r_hip_x, r_hip_x)

   # distance between hip and knee
   distance_between_lh_lk = findDistance(l_hip_x, l_hip_y, l_knee_x, l_knee_y)
   distance_between_rh_rk = findDistance(r_hip_x, r_hip_y, r_knee_x, r_knee_y)

   # distance between knee and heel
   distance_between_lk_lh = findDistance(l_knee_x, l_knee_y, l_heel_x, l_heel_y)
   distance_between_rk_rh = findDistance(r_knee_x, r_knee_y, r_heel_x, r_heel_x)

   # Next we do the same for finding the angles between all the landmarks

   # angle between shoulder and elbow.
   angle_between_ls_le = findAngle(l_shldr_x, l_shldr_y, l_elbow_x, l_elbow_y)
   angle_between_rs_re = findAngle(r_shldr_x, r_shldr_y, r_elbow_x, r_elbow_y)

   # angle between shoulder and hip
   angle_between_ls_lh = findAngle(l_shldr_x, l_shldr_y, l_hip_x, l_hip_y)
   angle_between_rs_rh = findAngle(r_shldr_x, r_shldr_y, r_hip_x, r_hip_x)

   # angle between hip and knee
   angle_between_lh_lk = findAngle(l_hip_x, l_hip_y, l_knee_x, l_knee_y)
   angle_between_rh_rk = findAngle(r_hip_x, r_hip_y, r_knee_x, r_knee_y)

   # angle between knee and heel
   angle_between_lk_lh = findAngle(l_knee_x, l_knee_y, l_heel_x, l_heel_y)
   angle_between_rk_rh = findAngle(r_knee_x, r_knee_y, r_heel_x, r_heel_x)


   # and thus our data points are
   data = [distance_between_ls_le, distance_between_rs_re, distance_between_ls_lh, distance_between_rs_rh,
   distance_between_lh_lk,distance_between_rh_rk, angle_between_ls_le, angle_between_rs_re,angle_between_ls_lh,
   angle_between_rs_rh,angle_between_lh_lk,angle_between_rh_rk,angle_between_lk_lh,angle_between_rk_rh]

   output = model(data)
   result = output.detach().numpy()
   predicted_class = np.argmax(result)
   if(predicted_class == 0):
      return "Positive"
   else:
      "Negative"

   

