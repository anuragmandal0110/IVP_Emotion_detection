from pyexpat import model
import torch
import os



def get_model() :
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

    # We are working with a dataset with 7 classes and thus we need to modify the model layers.
    model.fc = torch.nn.Linear(2048, 7)


    #if weights are saved , use that
    if(os.path.exists("./saved")):
       print("LOADING WEIGHTS INTO THE MODEL")
       model.load_state_dict(torch.load("./saved/saved.pt",map_location=torch.device('cpu')))


    return model


