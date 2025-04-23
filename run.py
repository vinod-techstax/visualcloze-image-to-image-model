import torch
# from diffusers import VisualClozePipeline
from src.diffusers.pipelines.visualcloze.pipeline_visualcloze import VisualClozePipeline
from diffusers.utils import load_image


# Load in-context images (make sure the paths are correct and accessible)
image_paths = [
    # in-context examples
    [
        load_image('https://github.com/lzyhha/VisualCloze/raw/main/examples/examples/tryon/00700_00.jpg'),
        load_image('https://github.com/lzyhha/VisualCloze/raw/main/examples/examples/tryon/03673_00.jpg'),
        load_image('https://github.com/lzyhha/VisualCloze/raw/main/examples/examples/tryon/00700_00_tryon_catvton_0.jpg'),
    ],
    # query with the target image
    [
        load_image('https://github.com/lzyhha/VisualCloze/raw/main/examples/examples/tryon/00555_00.jpg'),
        load_image('https://github.com/lzyhha/VisualCloze/raw/main/examples/examples/tryon/12265_00.jpg'),
        None
    ],
]

print("task 1")

# Task and content prompt
task_prompt = "Each row shows a virtual try-on process that aims to put [IMAGE2] the clothing onto [IMAGE1] the person, producing [IMAGE3] the person wearing the new clothing."
content_prompt = None

print("task 2")

# Load the VisualClozePipeline
pipe = VisualClozePipeline.from_pretrained("VisualCloze/VisualClozePipeline-384", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  # Save some VRAM by offloading the model to CPU

print("task 3")

# Run the pipeline
image_result = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    upsampling_height=1632,
    upsampling_width=1232,
    upsampling_strength=0.3,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0][0]


print("image_result")
print(image_result)

# Save the resulting image
image_result.save("visualcloze.png")