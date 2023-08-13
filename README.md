# Third task

## The use of object recognition technology to identify the objects that the robot can meet by using Teachable Machine

##### I used the Teachable Machine technology from Google, and then the image recognition model, and I used pictures of fruits (bananas, oranges, strawberries) and then taught them to the machine, and then I made a machine test to make sure of that
###### Now I will show the steps and pictures to do that:




##### 1- First, enter the machine learning from Google, then create a new project using the image project:

<img width="500" alt="T31" src="https://github.com/Worod44/Task3/assets/95488818/584e15a3-c347-48f7-851e-a5537cba582d">



##### 2- After that, I renewed how many classes I wanted. I chose three classes, a class for orange, a class for bananas, and a class for strawberries, and uploaded pictures of the three fruits from the image file:

<img width="320" alt="T32" src="https://github.com/Worod44/Task3/assets/95488818/c7ba819e-3c0c-452c-b007-4156e0ee8078">
<img width="320" alt="T33" src="https://github.com/Worod44/Task3/assets/95488818/ee4e934b-cd82-4b91-af49-3f55f108e0f4">
<img width="320" alt="T34" src="https://github.com/Worod44/Task3/assets/95488818/4c5e30c0-8d9f-4398-ae6d-0b3b42a90ecb">





##### 3- Third, I pressed the "Training" button to train the three classes to the machine:

<img width="500" alt="T34" src="https://github.com/Worod44/Task3/assets/95488818/4c5e30c0-8d9f-4398-ae6d-0b3b42a90ecb">





##### 4- Then, I uploaded a picture of the orange fruit, different from the pictures that I uploaded of the orange class, to test the training of the machine. Was it done correctly or not? 

<img width="300" alt="T35" src="https://github.com/Worod44/Task3/assets/95488818/3d78f800-19af-4311-9b04-a76701dd2520">




###### The result is clear in the picture, the training is successful



##### 5- Finally, I uploaded a picture of strawberry and banana fruit to verify whether the machine will be able to distinguish the two fruits together or not ?


<img width="300" alt="T36" src="https://github.com/Worod44/Task3/assets/95488818/dad162b0-b455-4613-aae6-f621baa8738c">


<img width="500" alt="T37" src="https://github.com/Worod44/Task3/assets/95488818/58e905f2-d22e-4f5a-a91c-1ae4e11d21aa">


###### The result is not very good and accurate, but we can say that she was able to distinguish that





### Next, I exported the training code using two programming languages, Java and Python

<img width="605" alt="T38" src="https://github.com/Worod44/Task3/assets/95488818/5761d21a-65a1-4cf5-976d-27259a9e7388">


#### Java language:
```
<div>Teachable Machine Image Model</div>
<button type="button" onclick="init()">Start</button>
<div id="webcam-container"></div>
<div id="label-container"></div>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>
<script type="text/javascript">
    // More API functions here:
    // https://github.com/googlecreativelab/teachablemachine-community/tree/master/libraries/image

    // the link to your model provided by Teachable Machine export panel
    const URL = "./my_model/";

    let model, webcam, labelContainer, maxPredictions;

    // Load the image model and setup the webcam
    async function init() {
        const modelURL = URL + "model.json";
        const metadataURL = URL + "metadata.json";

        // load the model and metadata
        // Refer to tmImage.loadFromFiles() in the API to support files from a file picker
        // or files from your local hard drive
        // Note: the pose library adds "tmImage" object to your window (window.tmImage)
        model = await tmImage.load(modelURL, metadataURL);
        maxPredictions = model.getTotalClasses();

        // Convenience function to setup a webcam
        const flip = true; // whether to flip the webcam
        webcam = new tmImage.Webcam(200, 200, flip); // width, height, flip
        await webcam.setup(); // request access to the webcam
        await webcam.play();
        window.requestAnimationFrame(loop);

        // append elements to the DOM
        document.getElementById("webcam-container").appendChild(webcam.canvas);
        labelContainer = document.getElementById("label-container");
        for (let i = 0; i < maxPredictions; i++) { // and class labels
            labelContainer.appendChild(document.createElement("div"));
        }
    }

    async function loop() {
        webcam.update(); // update the webcam frame
        await predict();
        window.requestAnimationFrame(loop);
    }

    // run the webcam image through the image model
    async function predict() {
        // predict can take in an image, video or canvas html element
        const prediction = await model.predict(webcam.canvas);
        for (let i = 0; i < maxPredictions; i++) {
            const classPrediction =
                prediction[i].className + ": " + prediction[i].probability.toFixed(2);
            labelContainer.childNodes[i].innerHTML = classPrediction;
        }
    }
</script>
```


#### Python language
```
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("<IMAGE_PATH>").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
```



