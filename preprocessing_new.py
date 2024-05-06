from datasets import load_dataset
from PIL import Image
import io
from transformers import ViTFeatureExtractor

# Load datasets
dataset_train = load_dataset('Jeneral/fer-2013', split='train')
dataset_test = load_dataset('Jeneral/fer-2013', split='test')

# Function to convert image bytes to PIL Image in RGB format
def convert_to_pil(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image_RGB = Image.merge("RGB", (image, image, image))
    return image_RGB

# Apply conversion to both training and testing datasets
dataset_train = dataset_train.map(lambda example: {'img': convert_to_pil(example['img_bytes'])}, remove_columns=['img_bytes'])
dataset_test = dataset_test.map(lambda example: {'img': convert_to_pil(example['img_bytes'])}, remove_columns=['img_bytes'])

# Rename 'labels' column to 'label'
dataset_train = dataset_train.rename_column("labels", "label")
dataset_test = dataset_test.rename_column("labels", "label")

# Load ViT feature extractor
model_id = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)

# Preprocessing function to prepare inputs for the model
def preprocess(batch):
    inputs = feature_extractor(batch['img'], return_tensors='pt')
    inputs['label'] = batch['label']
    return inputs

# Apply preprocessing to both datasets
prepared_train = dataset_train.with_transform(preprocess)
prepared_test = dataset_test.with_transform(preprocess)