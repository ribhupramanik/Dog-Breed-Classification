# utils.py
import tensorflow as tf

# Set to the image size your model expects (change if needed)
IMG_SIZE = 224
BATCH_SIZE = 32

def process_image(image_path):
    """Reads image_path (string tensor) and returns a preprocessed image tensor."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0  # normalize to [0,1]
    return image

def get_image_label(image_path, label):
    image = process_image(image_path)
    return image, label

def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    if test_data:
        print('Creating test data batches..')
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch
    #Data is a validation dataset, if don't need to shuffle it
    elif valid_data:
        print("Creating validation data batches..")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch

    else:
        print('Creating taining data batches...')
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y)))
        #Shuffling pathnames and labels before mapping preprocessor function is faster than suffling images
        data = data.shuffle(buffer_size=len(x))

        #Create (image, label) tuples (this also turns the image path into a preprocessed image)
        data = data.map(get_image_label)

        #Turn the training data into batches
        data_batch = data.batch(BATCH_SIZE)
        
    return data_batch