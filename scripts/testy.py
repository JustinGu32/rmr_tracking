import torch
from transformers import AutoModel, AutoImageProcessor
from transformers.image_utils import load_image

# load the model and processor
ckpt = "google/siglip2-base-patch16-384"
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoImageProcessor.from_pretrained(ckpt)

# load the image
image = load_image("https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg")
inputs = processor(images=[image], return_tensors="pt").to(model.device)

# run inference
with torch.no_grad():
    outputs = model.get_image_features(**inputs)

if isinstance(outputs, torch.Tensor):
    image_embeddings = outputs
elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
    image_embeddings = outputs.image_embeds
elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
    image_embeddings = outputs.pooler_output
else:
    raise TypeError(f"Unsupported output type from get_image_features: {type(outputs)}")

print(image_embeddings.shape)
