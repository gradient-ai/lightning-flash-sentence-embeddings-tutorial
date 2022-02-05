
import torch

import flash
from flash.text import TextClassificationData, TextClassifier
#using the SPAM text message classification from kaggle: https://www.kaggle.com/team-ai/spam-text-message-classification
datamodule = TextClassificationData.from_csv(
    "Message", #source column
    "Category", #target column : The data does not even need to be integer encoded!
    train_file="SPAM text message 20170820 - Data.csv",
    batch_size=8,
)

# The Model is  loaded with a huggingface backbone
model = TextClassifier(backbone="prajjwal1/bert-small", num_classes=datamodule.num_classes)

# A trainer object is created with the help of pytorch-lightning module and the task is finetuned
trainer = flash.Trainer(max_epochs=2, gpus=1)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# few data for predictions
datamodule = TextClassificationData.from_lists(
    predict_data=[
        "Can you spill the secret?",
        "Camera - You are awarded a SiPix Digital Camera! call 09061221066 fromm landline. Delivery within 28 days.",
        "How are you? We have reached India.",
    ],
    batch_size=8,
)
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)

# Finally, we save the model
trainer.save_checkpoint("spam_classification.pt")