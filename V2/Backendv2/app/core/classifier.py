import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

from ..config import *


class CNNEnsemble:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.CLASS_NAMES = ['1509', 'IRRI-6', 'Super White']

        self.effnet = self.load_effnet()
        self.mobilenet = self.load_mobilenet()
        self.resnet = self.load_resnet()

    # ======================
    # LOAD MODELS
    # ======================

    def load_effnet(self):
        model = models.efficientnet_v2_s()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)

        model.load_state_dict(torch.load(EFFNET_MODEL, map_location=self.device))
        model.eval().to(self.device)
        return model

    def load_mobilenet(self):
        model = models.mobilenet_v2()
        model.classifier[1] = nn.Linear(model.last_channel, 3)

        model.load_state_dict(torch.load(MOBILENET_MODEL, map_location=self.device))
        model.eval().to(self.device)
        return model

    def load_resnet(self):
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 3)

        model.load_state_dict(torch.load(RESNET_MODEL, map_location=self.device))
        model.eval().to(self.device)
        return model

    # ======================
    # FEATURE EXTRACTION
    # ======================

    def extract_effnet_features(self, x):
        x = self.effnet.features(x)
        x = self.effnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def extract_mobilenet_features(self, x):
        x = self.mobilenet.features(x)
        x = x.mean([2, 3])
        return x

    def extract_resnet_features(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    # ======================
    # PREDICTION
    # ======================

    def predict(self, crop):

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(crop)

        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():

            # 🔥 Deep Features
            eff_feat = self.extract_effnet_features(x)
            mob_feat = self.extract_mobilenet_features(x)
            res_feat = self.extract_resnet_features(x)

            deep_feat = torch.cat([eff_feat, mob_feat, res_feat], dim=1)
            deep_feat = deep_feat.cpu().numpy()[0]

            # 🔥 Softmax (for XGBoost)
            e = torch.softmax(self.effnet(x), dim=1).cpu().numpy()[0]
            m = torch.softmax(self.mobilenet(x), dim=1).cpu().numpy()[0]
            r = torch.softmax(self.resnet(x), dim=1).cpu().numpy()[0]

            softmax_feat = np.concatenate([e, m, r])

        return {
            "features": deep_feat,      # OSR
            "softmax": softmax_feat     # XGBoost
        }