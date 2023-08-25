from flask import Flask, render_template, request
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
from PIL import Image
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__)

# import model
model = YOLO('best.pt')

# Memproses gambar
def preprocess_image(image):
    image = Image.open(image)
    image = resize(image, (640, 640))
    
    #Mengubah to Tensor
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    return image


def classify_image(image):

    image_tensor = preprocess_image(image)

    results = model(image_tensor)

    for result in results:
        classes = result.names
        highest_prob = max(result.probs.top5conf)
        clasify = classes[result.probs.top1]
        array1 = result.probs.top5
        array2 = result.probs.top5conf
        class_array = [classes[val] for val in array1]
        combined = [(class_array[i], array2[i]) for i in range(len(array2))]
        combined_clear = [(label, round(score.item(), 3)) for label, score in combined]



    return clasify, highest_prob, combined_clear

@app.route('/', methods=['GET', 'POST'])

def home():
    if request.method == 'POST':
        # mendapatkan gambar dari upload image
        image = request.files['image']

        # Save gambar ke tmp
        temp_dir = os.path.join(app.root_path, 'static', 'temp')
        filename = image.filename
        image.save(os.path.join(temp_dir, filename))

        clasify, highest_prob, combined_clear  = classify_image(image)

        # menuliskan hasil klasifikasi ke gambar
        img = cv2.imread(os.path.join(temp_dir, filename))
        for idx, text in enumerate(combined_clear[:5]):
            y_offset = 30 * (idx + 1)
            cv2.putText(img, f"{text}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,205), 2)

        cv2.imwrite(os.path.join(temp_dir, 'result.jpg'), img)

        # Delete gambar tmp
        os.remove(os.path.join(temp_dir, filename))

        return render_template('result.html', clasify=clasify , highest_prob=highest_prob )
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
