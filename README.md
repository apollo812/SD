## 1. Git clone
``git clone https://github.com/apollo812/SD.git SDXL``

## 2. Run the setup.sh file
``bash setup.sh``

## 3. Generate the image from text
Write your desired prompt in the “Prompt” field.

``python scripts/txt2img.py --prompt "prompt"``

Results are saved in the output folder.

## 4. env setting info
MODEL_TYPE=server/local

MODEL_LOAD_TYPE=pretrained/single

MODEL=base/refiner

OUTPUT_PATH=output