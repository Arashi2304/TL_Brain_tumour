import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.io import imread
from tensorflow import keras
from keras.applications import VGG16
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

TARGET_SIZE = (224, 224)

#Section 1: Data Preprocessing
def prep_data(str):
    '''
    Imports the dataset from the file and segregates it into training and validation sets.
    
    Parameters
    ----------
    str : string
        The path to the dataset.
    
    Returns
    -------
    X_train : numpy array
        The training set.
    y_train : numpy array
        The labels for the training set.
    X_val : numpy array
        The validation set.
    y_val : numpy array
        The labels for the validation set.
    '''
    image_array = []
    label_array = []
    
    for label in os.listdir(str):
        label_dir = os.path.join(str, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                image_path = os.path.join(label_dir, filename)
                
                try:
                    if ':' in filename:
                        continue

                    image = imread(image_path, as_gray=True)
                    resized_image = resize(image, TARGET_SIZE, anti_aliasing=True)
                    resized_image = np.stack([resized_image] * 3, axis=-1)
                    image_array.append(resized_image)

                    if label == 'yes':
                        label_array.append(1)
                    elif label == 'no':
                        label_array.append(0)
                except Exception as e:
                    continue
                
    # Convert lists to numpy arrays
    image_array = np.array(image_array)
    label_array = np.array(label_array)

    combined = list(zip(image_array, label_array))
    np.random.shuffle(combined)

    split_index = int(0.9 * len(combined))
    train_data = combined[:split_index]
    val_data = combined[split_index:]

    X_train, y_train = zip(*train_data)
    X_val, y_val = zip(*val_data)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    return X_train, y_train, X_val, y_val

#Section 2: Model initialization and training
def create_model():
    '''
    Initializes the model and compiles it.
    
    Returns
    -------
    detection_model : keras model
        The compiled model.
    base_model : keras model
        The base model, VGG16.
    '''
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    detection_model = keras.Sequential([
        base_model,
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    detection_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return detection_model, base_model

def train_model(detection_model, base_model, X_train, y_train, X_val, y_val):
    '''
    Trains the model.
    
    Parameters
    ----------
    detection_model : keras model
        The compiled model.
    base_model : keras model
        The base model, VGG16.
    X_train : numpy array
        The training set.
    y_train : numpy array
        The labels for the training set.
    X_val : numpy array
        The validation set.
    y_val : numpy array
        The labels for the validation set.
    
    Returns
    -------
    detection_model : keras model
        The trained model.
    history_pre : keras history
        The history of the model before fine-tuning.
    history_post : keras history
        The history of the model after fine-tuning.
    '''
    history_pre = detection_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    
    for layer in base_model.layers[:10]:
        layer.trainable = True

    detection_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history_post = detection_model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))
    
    return detection_model, history_pre, history_post

def plot_statistics(history_pre, history_post, detection_model, X_val, y_val, save_path_statistics='Figures/statistics.png', save_path_cm='Figures/confusion_matrix.png'):
    '''
    Function to plot the accuracy and loss of the model before and after fine-tuning, and the confusion matrix.
    
    Parameters
    ----------
    history_pre : keras history
        The history of the model before fine-tuning.
    history_post : keras history
        The history of the model after fine-tuning.
    detection_model : keras model
        The trained model.
    X_val : numpy array
        The validation set.
    y_val : numpy array
        The labels for the validation set.
    '''
    y_val_pred = detection_model.predict(X_val)
    y_val_pred_binary = (y_val_pred > 0.5).astype(int)
    cm = confusion_matrix(y_val, y_val_pred_binary)
    
    # Plotting accuracy and loss
    plt.figure(figsize=(12, 6))
    
    #Plot accuracy from the first training session
    plt.subplot(2, 2, 1)
    plt.plot(history_pre.history['accuracy'], label='Train (Before Fine-tuning)')
    plt.plot(history_pre.history['val_accuracy'], label='Validation (Before Fine-tuning)')
    plt.title('Model Accuracy (Before Fine-tuning)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot accuracy from the second training session (fine-tuning)
    plt.subplot(2, 2, 2)
    plt.plot(history_post.history['accuracy'], label='Train (After Fine-tuning)')
    plt.plot(history_post.history['val_accuracy'], label='Validation (After Fine-tuning)')
    plt.title('Model Accuracy (After Fine-tuning)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss from the first training session
    plt.subplot(2, 2, 3)
    plt.plot(history_pre.history['loss'], label='Train (Before Fine-tuning)')
    plt.plot(history_pre.history['val_loss'], label='Validation (Before Fine-tuning)')
    plt.title('Model Loss (Before Fine-tuning)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot loss from the second training session (fine-tuning)
    plt.subplot(2, 2, 4)
    plt.plot(history_post.history['loss'], label='Train (After Fine-tuning)')
    plt.plot(history_post.history['val_loss'], label='Validation (After Fine-tuning)')
    plt.title('Model Loss (After Fine-tuning)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path_statistics)
    plt.show()
    
    # Plotting confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Tumor Absent', 'Tumour Present'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(save_path_cm)
    plt.show()
    
    return None

if __name__ == '__main__':
    # Importing the dataset
    X_train, y_train, X_val, y_val = prep_data('Data/brain_tumor_data')
    
    # Initializing and training the model
    detection_model, base_model = create_model()
    detection_model, history_pre, history_post = train_model(detection_model, base_model, X_train, y_train, X_val, y_val)
    
    # Plotting statistics
    plot_statistics(history_pre, history_post, detection_model, X_val, y_val)