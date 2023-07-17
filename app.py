from flask import Flask, render_template, request
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__)

# Load model
model = YOLO('best.pt')

# Function to preprocess image
def preprocess_image(image):
    image = Image.open(image)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    return image

# Function to perform image classification
def classify_image(image):
    # Preprocess image
    image_tensor = preprocess_image(image)

    # Perform inference
    results = model(image_tensor)

    # Get class labels and probabilities
    for result in results:
        classes = result.names
        highest_prob = max(result.probs.top5conf)
        clasify = classes[result.probs.top1]
        array1 = result.probs.top5
        array2 = result.probs.top5conf
        class_array1 = ["jinak" if val == 1 else "ganas" for val in array1]
        combined = [(class_array1[i], array2[i]) for i in range(len(array2))]
        combined_clear = [(label, round(score.item(), 3)) for label, score in combined]


    return clasify, highest_prob, combined_clear

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get uploaded image
        image = request.files['image']

        # Save temporary image
        temp_dir = os.path.join(app.root_path, 'static', 'temp')
        filename = image.filename
        image.save(os.path.join(temp_dir, filename))

        # Perform image classification
        clasify, highest_prob, combined_clear  = classify_image(image)

        # Write prediction on the image
        img = cv2.imread(os.path.join(temp_dir, filename))
        cv2.putText(img, f"{combined_clear[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
        cv2.putText(img, f"{combined_clear[1]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
        cv2.imwrite(os.path.join(temp_dir, 'result.jpg'), img)

        # Delete temporary file
        os.remove(os.path.join(temp_dir, filename))

        return render_template('result.html', clasify=clasify , highest_prob=highest_prob )
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
