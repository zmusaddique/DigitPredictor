from flask import Flask, render_template, request, flash
import os, io, random, subprocess, joblib, cv2, base64, numpy as np
from werkzeug.utils import secure_filename
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier



# Load and initialize the models
knn_enhanced = joblib.load('models/knn_model_ehanced.pkl')
knn_extended = joblib.load('models/knn_clf_extended_colab.pkl')

rf_ovo_clf = joblib.load('models/ovo_rf_clf_colab.pkl')
rf_ovr_clf = joblib.load('models/ovr_rf_clf_colab.pkl')

sgd_ovo_clf = joblib.load('models/ovo_sgd_clf_colab.pkl')
sgd_ovr_clf = joblib.load('models/ovr_sgd_clf_colab.pkl')

svc_ovo_clf = joblib.load('models/ovo_svc_clf_colab.pkl')
svc_ovr_clf = joblib.load('models/ovr_svc_clf_colab.pkl')

ALLOWED_EXTENSIONS = {"webp", "png", "jpg", "jpeg", "gif"}
UPLOADS_FOLDER = "uploads"

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 4MB limit
app.config['UPLOAD_FOLDER'] = UPLOADS_FOLDER
# Define the color gradient for confidence scores
color_gradient = [
    (1.0, 0.0, 0.0),  # red
    (1.0, 0.5, 0.0),  # orange
    (0.0, 1.0, 0.0)   # green
]

def get_color_from_gradient(gradient, value):
    colors = [np.array(color) for color in gradient]
    segments = len(colors) - 1
    segment_length = 1.0 / segments
    segment_index = int(value / segment_length)
    segment_start = segment_index * segment_length
    segment_end = (segment_index + 1) * segment_length
    segment_value = (value - segment_start) / segment_length
    color = colors[segment_index] * (1.0 - segment_value) + colors[segment_index + 1] * segment_value
    return tuple(color)

def get_confidence_score_color(score):
    return get_color_from_gradient(color_gradient, score)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST']) # type: ignore
def predict():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return render_template('index.html')

        file = request.files["file"]

        # If the user uploads their own image
        if file and allowed_file(file.filename):
            
            # Check the file size
            if not is_file_size_valid(file):
                flash("File size exceeds the limit (4MB)")
                return render_template('index.html')

            # Save the file
            filename = secure_filename(file.filename)  # type: ignore
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Perform image preprocessing and prediction
            predictions = process_uploaded_image(filename)

        # If the user selects an image from the menu
        else:
            selected_image = request.form.get("image-select")
              # If the user selects an image from the menu
            if selected_image:

                # Perform image preprocessing and prediction
                predictions = process_selected_image(selected_image)


            else:
                flash("No selected image")
                return render_template('index.html')

        # Flash a success message
        flash('Success!', 'success')
        return render_template("index.html", predictions=predictions)
        

def is_file_size_valid(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size <= app.config['MAX_CONTENT_LENGTH']

test_images, test_labels = None, None

def process_selected_image(selected_image):
    global test_images, test_labels
    # Check if the selected image is from the menu
    if selected_image != "NONE":
        # Get the label corresponding to the selected image
        label = int(selected_image)

        # Check if the cached dataset exists
        if os.path.exists('test_images.npy') and os.path.exists('test_labels.npy'):
            # Load the cached dataset
            test_images = np.load('test_images.npy')
            test_labels = np.load('test_labels.npy')
        else:
            # Run your script to download and cache the MNIST dataset
            subprocess.run(['python3', 'preprocess_test_set.py'])

            # Load the cached dataset
            test_images = np.load('test_images.npy')
            test_labels = np.load('test_labels.npy')

        # Filter images with the selected label
        images_with_label = [image for image, image_label in zip(test_images, test_labels) if image_label == label]
        
        # Select a random image with the selected label
        random_image = random.choice(images_with_label)

        # Perform image preprocessing and prediction on the selected image
        predictions = predict_image(random_image)

        # Pass the predictions, predicted label, confusion matrix, and model metrics to the template
        return predictions #type: ignore
    else:
        flash("Please select an image!", 'error')
    
def process_uploaded_image(filename):
    # Read the uploaded image
    image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Perform image preprocessing steps
    preprocessed_image = preprocess_image(image)

    # Perform prediction using preprocessed image
    predictions = predict_image(preprocessed_image)

    return predictions


def preprocess_image(image):
    # Resize the image to a fixed size
    resized_image = cv2.resize(image, (28, 28))

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Normalize the pixel values to [0, 1]
    normalized_image = grayscale_image / 255.0 # type: ignore

    # Invert the colors of the image
    inverted_image = 1 - normalized_image

    # Flatten the image to a 1D array
    flattened_image = inverted_image.flatten()

    # Return the preprocessed image
    return flattened_image

def predict_image(preprocessed_image):
    # Perform prediction using a machine learning model
    predictions = make_predictions(preprocessed_image)

    return predictions

def confusion_matrix(model):
    image_path = f'confusion_matrices/{model}_cm.png'
    if os.path.exists(image_path):
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            img_tag = f'data:image/png;base64,{encoded_image}'
            return img_tag
    else:
        return None

def calculate_confidence_score(model, preprocessed_image):
    """
    This function calculates how confident the model is with the prediction.
    This does not guarantee the correctness of the prediction. 
    """
    confidence_score = 0
    predicted_probabilities = None
    decision_values = None
    if isinstance(model, (OneVsOneClassifier, OneVsRestClassifier)):
        base_estimator = model.estimator  # type: ignore
        if isinstance(base_estimator, (SGDClassifier, RandomForestClassifier, SVC)):
        # For SGD, SVC, and RandomForestClassifier models, use decision_function to get the confidence scores
            if hasattr(model, 'decision_function'):
                decision_values = model.decision_function(preprocessed_image)
                confidence_scores = (decision_values - decision_values.min()) / (decision_values.max() - decision_values.min()) * 100
                confidence_score = max(max(confidence_scores)) #type: ignore
            elif hasattr(model, 'predict_proba'):
                # For RandomForestClassifier, use predict_proba to get the predicted probabilities
                predicted_probabilities = model.predict_proba(preprocessed_image)  #type: ignore
                confidence_score = predicted_probabilities[0].max(axis=0) * 100
            else:
                raise ValueError("Unsupported model type")
            
    elif isinstance(model, SVC):
        # For SVC model, use decision_function to get the confidence scores
        decision_values = model.decision_function(preprocessed_image)
        confidence_score = max(max(model.decision_function(preprocessed_image)))  # type: ignore
        
    elif isinstance(model, KNeighborsClassifier):
        # For KNeighborsClassifier model, use predict_proba to get the predicted probabilities
        predicted_probabilities = model.predict_proba(preprocessed_image) #type: ignore
        confidence_score = max(predicted_probabilities[0]) * 100
    else:
        raise ValueError("Unsupported model type")

    return confidence_score, predicted_probabilities if predicted_probabilities is not None else decision_values #type: ignore

import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_digit(data):
    image = data.reshape(28, 28)
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(image, cmap=mpl.cm.binary, interpolation="nearest") #type: ignore
    ax.axis("off")

    # Render the figure on the canvas
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Save the rendered figure to a buffer
    buffer = io.BytesIO()
    canvas.print_png(buffer)

    # Encode the image buffer to base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    plt.close(fig)  # Close the figure

    return image_base64


def make_predictions(preprocessed_image):

    # Assuming your 1D array is named 'input_array'
    preprocessed_image = preprocessed_image.reshape(1, -1)
    # preprocessed_image_index = np.where(np.all(test_images == preprocessed_image, axis=1))[0][0] #type: ignore
    
    # Use the preprocessed image and loaded models to make predictions
    knn_enhance_pred = knn_enhanced.predict(preprocessed_image)[0]
    knn_extended_pred = knn_extended.predict(preprocessed_image)[0]

    rf_ovo_clf_pred = rf_ovo_clf.predict(preprocessed_image)[0]
    rf_ovr_clf_pred = rf_ovr_clf.predict(preprocessed_image)[0]

    sgd_ovo_clf_pred = sgd_ovo_clf.predict(preprocessed_image)[0]
    sgd_ovr_clf_pred = sgd_ovr_clf.predict(preprocessed_image)[0]

    svc_ovo_clf_pred = svc_ovo_clf.predict(preprocessed_image)[0]
    svc_ovr_clf_pred = svc_ovr_clf.predict(preprocessed_image)[0]

    # Generate base64-encoded image data


    # Create a dictionary to store the model predictions and confidence scores
    predictions = {
        'image': plot_digit(preprocessed_image),
        'knn_enhanced': {
            'name': 'K-Nearest Neighbors (KNN)',
            'training':'This model was trained on handwirtten images in the mnist dataset with the some good parameters using a function called gridsearchCV()',
            'description':'The KNN classifier is a simple yet powerful supervised machine learning algorithm used for classification tasks. It belongs to the family of instance-based learning, where the algorithm learns patterns from the training data and makes predictions based on the similarity of new instances to the labeled instances in the training set.',
            'video':'https://youtu.be/otolSnbanQk',
            'sklearn':'http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html',
            'prediction':knn_enhance_pred,
            'confidence': calculate_confidence_score(knn_enhanced, preprocessed_image),
            'accuracy': 0.9714, 
            'precision': 0.9715597201945959, 
            'recall': 0.9714,
            'f1_score':0.9713597782738583,
            'confusion_matrix': confusion_matrix('knn_enhanced') #type:ignore
        },
        'knn_extended': {
            'name': 'K-Nearest Neighbors (KNN)',
            'training':'This model was trained on 4 DIFFERENT VARIANTS of handwirtten images in the mnist dataset with the some good parameters using a function called gridsearchCV()',
            'description':'The KNN classifier is a simple yet powerful supervised machine learning algorithm used for classification tasks. It belongs to the family of instance-based learning, where the algorithm learns patterns from the training data and makes predictions based on the similarity of new instances to the labeled instances in the training set.',
            'video':'https://youtu.be/otolSnbanQk',
            'sklearn':'http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html',
            'prediction': knn_extended_pred,
            'confidence': calculate_confidence_score(knn_extended, preprocessed_image),
            'accuracy': 0.9763, 
            'precision': 0.9766352308968693, 
            'recall': 0.9763,
            'f1_score':0.9762833512584528,
            'confusion_matrix': confusion_matrix("knn_extended") #type:ignore
        },
        'rf_ovo_clf': {
            'name': 'Random Forest Classifier',
            'training': 'This Random Forest Classifier is trained using the One-vs-One (OvO) strategy. In this strategy, multiple binary classifiers are trained, each distinguishing between one pair of classes. The final prediction is made by combining the predictions of all binary classifiers.',
            'description': 'The Random Forest Classifier is an ensemble learning method that combines multiple decision trees to make predictions. It constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) or the mean prediction (regression) of the individual trees.',
            'video': 'https://youtu.be/nxFG5xdpDto',
            'sklearn': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html',
            'prediction': rf_ovo_clf_pred,
            'confidence': calculate_confidence_score(rf_ovo_clf, preprocessed_image),
            'accuracy': 0.9656, 
            'precision': 0.9654536968771158, 
            'recall': 0.9656,
            'f1_score':0.9656443811404773,
            'confusion_matrix': confusion_matrix("rf_ovo_clf") #type:ignore
        },
        'rf_ovr_clf': {
            'name': 'Random Forest Classifier',
            'training': 'This Random Forest Classifier is trained using the One-vs-Rest (OvR) strategy. In this strategy, multiple binary classifiers are trained, each distinguishing between one class and the rest of the classes. The final prediction is made by selecting the class with the highest probability or by combining the predictions of all binary classifiers.',
            'description': 'The Random Forest Classifier is an ensemble learning method that combines multiple decision trees to make predictions. It constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) or the mean prediction (regression) of the individual trees.',
            'video': 'https://youtu.be/nxFG5xdpDto',
            'sklearn': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html',
            'prediction': rf_ovr_clf_pred,
            'confidence': calculate_confidence_score(rf_ovr_clf, preprocessed_image),
            'accuracy': 0.9665, 
            'precision': 0.9664331517825872, 
            'recall': 0.9665,
            'f1_score':0.9664931379194104,
            'confusion_matrix': confusion_matrix("rf_ovr_clf") #type:ignore
        },
        'sgd_ovo_clf': {
            'name': 'SGD Classifier',
            'training': 'This SGD Classifier is trained using the One-vs-One (OvO) strategy. In this strategy, a binary classifier is trained for every pair of classes. Each binary classifier is trained to distinguish between one pair of classes while ignoring the others. The final prediction is made by selecting the class with the most number of votes from all binary classifiers.',
            'description': 'The SGD Classifier is a linear classification model that applies Stochastic Gradient Descent optimization to train the model. It is capable of handling large-scale and sparse datasets efficiently. The classifier finds an optimal separating hyperplane that maximizes the margin between classes.',
            'video': 'https://www.example.com/sgd-classifier-tutorial',
            'sklearn': 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html',
            'prediction': sgd_ovo_clf_pred,
            'confidence': calculate_confidence_score(sgd_ovo_clf, preprocessed_image),
            'accuracy': 0.813, 
            'precision': 0.8591031022514455, 
            'recall': 0.813,
            'f1_score':0.800547282398953,
            'confusion_matrix': confusion_matrix("sgd_ovo_clf") #type:ignore
        },
        'sgd_ovr_clf': {
            'name': 'SGD Classifier',
            'training': 'This SGD Classifier is trained using the One-vs-One (OvO) strategy. In this strategy, a binary classifier is trained for every pair of classes. Each binary classifier is trained to distinguish between one pair of classes while ignoring the others. The final prediction is made by selecting the class with the most number of votes from all binary classifiers.',
            'description': 'The SGD Classifier is a linear classification model that applies Stochastic Gradient Descent optimization to train the model. It is capable of handling large-scale and sparse datasets efficiently. The classifier finds an optimal separating hyperplane that maximizes the margin between classes.',
            'video': 'https://www.example.com/sgd-classifier-tutorial',
            'sklearn': 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html',
            'prediction': sgd_ovr_clf_pred,
            'confidence': calculate_confidence_score(sgd_ovr_clf, preprocessed_image),
            'accuracy': 0.8253, 
            'precision': 0.8492765055766178, 
            'recall': 0.8253,
            'f1_score':0.8208239418579742,
            'confusion_matrix': confusion_matrix("sgd_ovr_clf") #type:ignore
        },
        'svc_ovo_clf': {
             'name': 'SVC (Support Vector Classifier)',
            'training': 'This SVC is trained using the One-vs-One (OvO) strategy. In this strategy, a binary classifier is trained for every pair of classes. Each binary classifier is trained to distinguish between one pair of classes while ignoring the others. The final prediction is made by selecting the class with the most number of votes from all binary classifiers. This model is puposefully trained bad to show the extremes of classification',
            'description': 'The Support Vector Classifier (SVC) is a powerful machine learning model that uses support vectors to perform classification. It finds a hyperplane that maximizes the margin between classes, allowing for effective separation. The SVC can handle both linearly separable and non-linearly separable data through the use of different kernel functions.',
            'video': 'https://www.example.com/svc-classifier-tutorial',
            'sklearn': 'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html',
            'prediction': svc_ovo_clf_pred,
            'confidence': calculate_confidence_score(svc_ovo_clf, preprocessed_image),
            'accuracy': 0.1028, 
            'precision': 0.010280000000000001, 
            'recall': 0.1028,
            'f1_score': 0.019165469713456656,
            'confusion_matrix': confusion_matrix("svc_ovo_clf") #type:ignore
        },
        'svc_ovr_clf': {
            'name': 'SVC (Support Vector Classifier)',
            'training': 'This SVC is trained using the One-vs-Rest (OvR) strategy. In this strategy, a binary classifier is trained for each class, treating the samples of that class as positive and the samples of all other classes as negative. The final prediction is made by selecting the class with the highest decision score. This model is puposefully trained bad to show the extremes of classification',
            'description': 'The Support Vector Classifier (SVC) is a powerful machine learning model that uses support vectors to perform classification. It finds a hyperplane that maximizes the margin between classes, allowing for effective separation. The SVC can handle both linearly separable and non-linearly separable data through the use of different kernel functions.',
            'video': 'https://www.example.com/svc-classifier-tutorial',
            'sklearn': 'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html',
            'prediction': svc_ovr_clf_pred,
            'confidence': calculate_confidence_score(svc_ovr_clf, preprocessed_image),
            'accuracy': 0.1028, 
            'precision': 0.01056784, 
            'recall': 0.1028,
            'f1_score': 0.019165469713456656,
            'confusion_matrix': confusion_matrix("svc_ovr_clf") #type:ignore
        }
    }

    # Perform any post-processing on the predictions

    return predictions

@app.route("/working")
def advanced_view():
    return render_template("working.html")

@app.route("/na")
def na():
    return render_template("na.html")

if __name__ == '__main__':
    app.run(debug=True, port=5001)
