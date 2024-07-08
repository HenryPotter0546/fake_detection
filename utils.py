from PIL import Image
import torchvision
import torch

def process_image(image_pil, res=None, range=(-1, 1)):
        if res:
            image_pil = image_pil.resize(res, Image.BILINEAR)
        image = torchvision.transforms.ToTensor()(image_pil) # range [0, 1]
        r_min, r_max = range[0], range[1]
        image = image * (r_max - r_min) + r_min # range [r_min, r_max]
        return image[None, ...], image_pil
    
def image2tensor(image_path, img_size, device):
    # load image from file
    imgs = []
    image_size = (img_size, img_size)
    image_pil = Image.open(image_path).convert('RGB')
    img, _ =  process_image(image_pil, res=image_size)
    img = img.to(device)
    imgs.append(img)
    imgs = torch.vstack(imgs)
    images = torch.nn.functional.interpolate(imgs, size=img_size, mode="bilinear")
    image_tensors = images.to(torch.float16)
    return image_tensors