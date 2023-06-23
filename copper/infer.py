from typing import Tuple, Dict

import os
import lightning as L
import torch
import hydra
from omegaconf import DictConfig
from copper import utils
import random
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import requests
from io import BytesIO

@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig):
    # Get model config
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Check if test=True is passed
    if cfg.get("test"):
        # log.info("Starting Inference!")
        
        # Loading latest checkpoint from saved history
        latest_checkpoint_path = None
        for root, dirs, files in os.walk('./outputs/'):
            for file in files:
                if file.endswith(".ckpt"):
                    if latest_checkpoint_path is None or file > latest_checkpoint_path:
                        latest_checkpoint_path = os.path.join(root, file)

        print("Latest_checkpoint_path:", latest_checkpoint_path)

        # Loading weights from latest checkpoint
        model = model.load_from_checkpoint(latest_checkpoint_path)

        # Defining transformation steps for new data
        transform=T.Compose([
                T.Resize((32, 32)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        if cfg.get("imagepath"):
            # read web image of cat/dog
            response = requests.get(cfg.get("imagepath"))
            image = Image.open(BytesIO(response.content))
        else:
        # Using random image from test data for inference
            dataset = ImageFolder(root="./data/PetImages_split/test/")

            # Selectig random index from test dataset
            indices = random.sample(range(len(dataset)), 1)[0]

            # image, label from selected index
            image, label = dataset[indices]

        # Preprocessing image
        image = transform(image)
        image = image.unsqueeze(0)

        # Predict the class of the image
        with torch.no_grad():
            # Prediction
            prediction = model(image)

            # Get the top 2 probabilities
            predicted_class = torch.softmax(prediction, dim=1)[0].numpy()

            dic = {'cat':round(predicted_class[0],2), 'dog': round(predicted_class[1],2)}
            print("Probability of cat vs dog:", dic)

if __name__ == "__main__":
    main()