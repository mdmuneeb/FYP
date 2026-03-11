import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torchvision import transforms

from ..config import *
from PIL import Image


class CNNEnsemble:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])

        self.effnet = self.load_effnet()
        self.mobilenet = self.load_mobilenet()
        self.resnet = self.load_resnet()

    def load_effnet(self):

        model = models.efficientnet_v2_s()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features,3)
        model.load_state_dict(torch.load(EFFNET_MODEL, map_location=self.device))
        model.eval().to(self.device)

        return model

    def load_mobilenet(self):

        model = models.mobilenet_v2()
        model.classifier[1] = nn.Linear(model.last_channel,3)
        model.load_state_dict(torch.load(MOBILENET_MODEL,map_location=self.device))
        model.eval().to(self.device)

        return model

    def load_resnet(self):

        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features,3)
        model.load_state_dict(torch.load(RESNET_MODEL,map_location=self.device))
        model.eval().to(self.device)

        return model

    def predict(self, crop):

        img = Image.fromarray(crop)

        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():

            e = torch.softmax(self.effnet(x),1).cpu().numpy()[0]
            m = torch.softmax(self.mobilenet(x),1).cpu().numpy()[0]
            r = torch.softmax(self.resnet(x),1).cpu().numpy()[0]

        return np.concatenate([e,m,r])