import random
import onnxruntime as ort
import torch
from torchvision import transforms
from PIL import Image

IMG_SIZE = 64
IMAGE_DIR = "data/lesson many/fruits/"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

class_names = [
    'Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado', 'Banana',
    'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Cherry', 'Clementine',
    'Corn', 'Cucumber Ripe', 'Grape Blue', 'Kiwi', 'Lemon', 'Limes',
    'Mango', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach',
    'Pear', 'Pepper Green', 'Pepper Red', 'Pineapple', 'Plum',
    'Pomegranate', 'Potato Red', 'Raspberry', 'Strawberry', 'Tomato',
    'Watermelon'
]

codes = [f"{i:02d}" for i in range(len(class_names))]
labels = [f"{n} ({c})" for n, c in zip(class_names, codes)]

session = ort.InferenceSession("fruits+.onnx")
softmax = torch.nn.Softmax(dim=0)

for _ in range(5):
    idx = random.randint(0, 66)
    path = f"{IMAGE_DIR}{idx}.jpg"

    img = Image.open(path)
    tensor = transform(img).unsqueeze(0).numpy()

    output = session.run(None, {"input": tensor})[0][0]
    pred = output.argmax()

    prob = softmax(torch.tensor(output))[pred].item() * 100

    print(f"На малюнку №{idx} з ймовірністю {prob:.2f}% зображено {labels[pred]}.")