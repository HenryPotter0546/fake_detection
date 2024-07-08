import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from ssim import image_similarity_vectors_via_numpy

image_path="/home/zzw5373/wh/fake_detection/img/generate/astronaut_rides_horse.png"
device = "cuda:0"
prompt = ""

model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

init_image = Image.open(image_path).convert("RGB")
init_image = init_image.resize((512, 512))
# init_image.save("/home/zzw5373/wh/fake_detection/img/real_image/manually_shot_cat.png")
images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
images[0].save("/home/zzw5373/wh/fake_detection/img/generate/astronaut_rides_horse_two.png")

image_path="/home/zzw5373/wh/fake_detection/img/generate/astronaut_rides_horse_two.png"
init_image = Image.open(image_path).convert("RGB")
init_image = init_image.resize((512, 512))
images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
images[0].save("/home/zzw5373/wh/fake_detection/img/generate/astronaut_rides_horse_three.png")


image_cp1 = Image.open("/home/zzw5373/wh/fake_detection/img/generate/astronaut_rides_horse.png")
image_cp2 = Image.open("/home/zzw5373/wh/fake_detection/img/generate/astronaut_rides_horse_three.png")
cosine_similarity = image_similarity_vectors_via_numpy(image_cp1, image_cp2)
print('cosine_similarity:', cosine_similarity)