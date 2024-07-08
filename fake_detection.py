from diffusers import StableDiffusionPipeline
import torch
import os
from utils import process_image, image2tensor

device = "cuda:0"
img_size=512

t_id=0 # the index of timestep

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt=""

image_path = "/home/zzw5373/wh/fake_detection/img/generate/astronaut_rides_horse.png"
image_tensors = image2tensor(image_path, img_size, device)
img_latent = pipe.vae.encode(image_tensors).latent_dist.sample(generator=None) * pipe.vae.config.scaling_factor
prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False)

num_inference_steps = 50
pipe.scheduler.set_timesteps(num_inference_steps, pipe.device)
timesteps = pipe.scheduler.timesteps
print("timesteps: ",timesteps)



t = timesteps[t_id]
print("t: ",t)

sampled_noise = torch.randn_like(img_latent).to(device)
t = torch.tensor(t, dtype=torch.long, device=device)

# add noise
x_t_latent = pipe.scheduler.add_noise(img_latent, sampled_noise, t)
print(x_t_latent)

#TODO using scale_model_input or add_noise
x_t_latent = pipe.scheduler.scale_model_input(img_latent, t)
print("x_t_latent.shape: ", x_t_latent.shape)
print("prompt_embeds.shape: ", prompt_embeds.shape)
print("-----------------------------------------------")
print("prompt_embeds: ", prompt_embeds)
print("-----------------------------------------------")

# predict the noise residual
noise_pred  = pipe.unet(
    x_t_latent,
    t,
    encoder_hidden_states=prompt_embeds,
    added_cond_kwargs=None,
    return_dict=False
)[0]

print("noise_pred: ", noise_pred)
print("-----------------------------------------------")



 # compute the previous noisy sample x_t -> x_t-1
latents = self.scheduler.step(noise_pred, t, latents, , return_dict=False)[0]
x_t_next_latent, x_0_predicted_latent = pipe.scheduler.step(model_pred, t, x_t_latent, return_dict=False)

latent = x_0_predicted_latent


latent = pipe.vae.decode(latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
# Postprocess image
image = pipe.image_processor.postprocess(latent)
