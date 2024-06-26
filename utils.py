from PIL import Image
import numpy as np

def clean_image(image):
    # Resize the image using PIL's resize method with LANCZOS filter
    image = image.resize((512, 512), Image.LANCZOS)
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def get_prediction(model, image):
    # Assuming this function returns model predictions
    predictions = model.predict(image)
    predictions_arr = predictions[0]
    return predictions, predictions_arr

def make_results(predictions, predictions_arr):
    # Assuming this function processes predictions and returns results
    classes = ['Healthy', 'Disease1', 'Disease2', 'Disease3']
    max_index = np.argmax(predictions_arr)
    status = classes[max_index]
    prediction = predictions_arr[max_index]
    return {'status': status, 'prediction': prediction}
