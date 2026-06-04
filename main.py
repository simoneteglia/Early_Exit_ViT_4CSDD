# Import the ViT class from the vit module
from vit import ViT

# Dictionary for hyperparameters
hyperparameters = {
    'in_channels': 3,
    'number_of_classes': 3,
    'image_size': 224,
    'patch_size': 16,
    'number_of_encoder' : 6,
    'embeddings' : 256,
    'd_ff_scale' : 4,
    'heads' : 8,
    'input_dropout_rate' : 0.1,
    'attention_dropout_rate' : 0.1,
    'feed_forward_dropout_rate' : 0.1,
}

# Create an instance of the network with parameters from the dictionary
model = ViT(**hyperparameters)
print("Model created!")