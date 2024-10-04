import gradio as gr
from PIL import Image
import os
import numpy as np
from outpaint import outpainting
from unet_scratch import colorazation, UNETmodel, utils1
from colorization import inference, model
from deeplab import colorazation, deeplabmodel, utils 

# outpaint
def outpaint(input_img):
    # Load the model
    model_path = 'outpaint\G_325.pth'
    gen_model = outpainting.load_model(model_path)

    # Preprocess the input image
    input_img = np.array(input_img)

    # Perform outpainting
    output_img, blended_img = outpainting.perform_outpaint(gen_model, input_img)

    # Convert the output and blended images to PIL images
    output_img_pil = Image.fromarray((output_img * 255).astype(np.uint8))
    blended_img_pil = Image.fromarray((blended_img * 255).astype(np.uint8))

    return output_img_pil, blended_img_pil
    # return blended_img_pil

# scratch model
def unetColorize_image(image):
    file_path = r'unet_scratch\ImageColorizationModel_8666.pth'
    model_2 = utils1.load_model(model_class=UNETmodel.MainModel, file_path=file_path)

    output_img = utils1.predict_color(model_2, image=image)
    return output_img

# pretrained model
def colorize_image(image):
     # Load the model
    # file_path = 'ImageColorizationModel10.pth'
    file_path = r'colorization\model_final.pth'
    model_2 = inference.load_model(model_class=model.MainModel, file_path=file_path)
    output_img = inference.predict_color(model_2, image=image)
    return output_img

# deeplab model
def depColorize_image(image):
     # Load the model
    file_path = r'deeplab\ImageColorizationModel10.pth'
    model_2 = utils.load_model(model_class=deeplabmodel.MainModel, file_path=file_path)

    output_img = utils.predict_color(model_2, image=image)
    return output_img


# Create the Gradio interface
outpaint_interface = gr.Interface(
    outpaint,
    gr.Image(type="pil", label="Input Image"),
    [gr.Image(type="pil", label="Output Image"), gr.Image(type="pil", label="Blended Image")],
    # gr.Image(type="pil", label="Output Image")
    # [gr.Image(type="pil", label="Outpainted Image")],
    title="Image Outpainting",
    description="Upload an image to perform outpainting.",
)

# pretrained model
colorization_interface = gr.Interface(
    colorize_image,
    gr.Image(type="pil", label="Input Image"),
    [gr.Image(type="pil", label="Output Image")],
    title="Image Colorization",
    description="Upload an image to perform colorization.",

)

# deeplab model
depinterface = gr.Interface(
    depColorize_image,
    gr.Image(type="pil", label="Input Image"),
    [gr.Image(type="pil", label="Output Image")],
    title="Image Colorization",
    description="Upload an image to perform colorization.",

)

# scratch model
unet_interface = gr.Interface(
    unetColorize_image,
    gr.Image(type="pil", label="Input Image"),
    [gr.Image(type="pil", label="Output Image")],
    title="Image Colorization",
    description="Upload an image to perform colorization.",

)


# Launch the interface
# interface.launch(share=True)
with gr.TabbedInterface([outpaint_interface, unet_interface, colorization_interface, depinterface ], ["Outpainting", "Colorization_unet_scratch","Colorization_pretrain_unet","colorization_Deeplab"]) as tabs:
    tabs.launch(share=True)