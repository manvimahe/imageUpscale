import streamlit as st
import streamlit as st
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import keras.backend as K
import keras
plt.gray()

# Function to add a background and adjust the layout width
def setup_page():
    # Background
    with open("C:/Users/LENOVO/Desktop/imageUpscale/images/gola3.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded_string});
            background-size: cover;
        }}
        
        /* Increase the max width of the page */
        .stApp > .main .block-container{{
            max-width: 90%;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

selected_box = st.sidebar.selectbox(
'Choose one of the following',
('Home', 'Upscale your Image', 'How it Works?')
)

def home():
    st.title('Image Upscaling')
    st.subheader("Motivation for the Project")
    st.write("This is a project done under IET NITK where the goal is to enhance the brightness of low light image and then pass it to the super resolution model to enhance the visibility and clarity of the image and removing the noise such that the output image is much more clear and visibily pleasing and aesthetically appealing.")
    st.subheader("Project Description")
    st.write("Our proposed model begins by implementing a mechanism to determine whether the input image exhibits characteristics of low-light conditions.Causes of low-light conditions can be due to insufficient or absent light source or uneven illumination caused by back-light and shadows. Subsequently, we will develop a function capable of discerning whether the given image meets the criteria for low-light classification. Upon identification of a low-light image, we will employ the Zero DCE model to enhance its brightness.")
    st.write("Following the enhancement process through the Zero DCE model, the image will undergo further refinement using the Super Resolution model. This subsequent step aims to produce a substantially clearer and denoised version of the image, leveraging the sophisticated capabilities inherent to the Super Resolution model.")
    st.write("In essence, our model is designed to automatically detect and address low-light scenarios in images, enhance their brightness using Zero DCE, and further refine them to achieve superior clarity and noise reduction through the Super Resolution model. This holistic approach ensures that images exhibiting low-light conditions are effectively processed to yield optimal visual outcomes.")


def analyze_image(img):

    gray_img = img.convert('L')

    img_array = np.array(gray_img)


    brightness = np.mean(img_array)

    contrast = np.std(img_array)

    return brightness<60 and contrast<50


def build_dce_net(): 
    input_img = keras.Input(shape=[None, None, 3])
    conv1 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(input_img)
    conv2 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv1)
    conv3 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv2)
    conv4 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv3)
    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
    conv5 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(int_con1)
    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(int_con2)
    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    x_r = layers.Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same")(
        int_con3
    )
    return keras.Model(inputs=input_img, outputs=x_r)


class SpatialConsistencyLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(reduction="none")

        self.left_kernel = tf.constant(
            [[[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.right_kernel = tf.constant(
            [[[[0, 0, 0]], [[0, 1, -1]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.up_kernel = tf.constant(
            [[[[0, -1, 0]], [[0, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.down_kernel = tf.constant(
            [[[[0, 0, 0]], [[0, 1, 0]], [[0, -1, 0]]]], dtype=tf.float32
        )

    def call(self, y_true, y_pred):
        original_mean = tf.reduce_mean(y_true, 3, keepdims=True)
        enhanced_mean = tf.reduce_mean(y_pred, 3, keepdims=True)
        original_pool = tf.nn.avg_pool2d(
            original_mean, ksize=4, strides=4, padding="VALID"
        )
        enhanced_pool = tf.nn.avg_pool2d(
            enhanced_mean, ksize=4, strides=4, padding="VALID"
        )

        d_original_left = tf.nn.conv2d(
            original_pool,
            self.left_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        d_original_right = tf.nn.conv2d(
            original_pool,
            self.right_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        d_original_up = tf.nn.conv2d(
            original_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_original_down = tf.nn.conv2d(
            original_pool,
            self.down_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        d_enhanced_left = tf.nn.conv2d(
            enhanced_pool,
            self.left_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        d_enhanced_right = tf.nn.conv2d(
            enhanced_pool,
            self.right_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        d_enhanced_up = tf.nn.conv2d(
            enhanced_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_enhanced_down = tf.nn.conv2d(
            enhanced_pool,
            self.down_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        d_left = tf.square(d_original_left - d_enhanced_left)
        d_right = tf.square(d_original_right - d_enhanced_right)
        d_up = tf.square(d_original_up - d_enhanced_up)
        d_down = tf.square(d_original_down - d_enhanced_down)
        return d_left + d_right + d_up + d_down
    




class ZeroDCE(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dce_model = build_dce_net()

    def compile(self, learning_rate, **kwargs):
        super().compile(**kwargs)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.spatial_constancy_loss = SpatialConsistencyLoss(reduction="none")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.illumination_smoothness_loss_tracker = keras.metrics.Mean(
            name="illumination_smoothness_loss"
        ) 
        self.spatial_constancy_loss_tracker = keras.metrics.Mean(
            name="spatial_constancy_loss"
        )
        self.color_constancy_loss_tracker = keras.metrics.Mean(
            name="color_constancy_loss"
        )
        self.exposure_loss_tracker = keras.metrics.Mean(name="exposure_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.illumination_smoothness_loss_tracker,
            self.spatial_constancy_loss_tracker,
            self.color_constancy_loss_tracker,
            self.exposure_loss_tracker,
        ]

    def get_enhanced_image(self, data, output):
        r1 = output[:, :, :, :3]
        r2 = output[:, :, :, 3:6]
        r3 = output[:, :, :, 6:9]
        r4 = output[:, :, :, 9:12]
        r5 = output[:, :, :, 12:15]
        r6 = output[:, :, :, 15:18]
        r7 = output[:, :, :, 18:21]
        r8 = output[:, :, :, 21:24]
        x = data + r1 * (tf.square(data) - data)
        x = x + r2 * (tf.square(x) - x)
        x = x + r3 * (tf.square(x) - x)
        enhanced_image = x + r4 * (tf.square(x) - x)
        x = enhanced_image + r5 * (tf.square(enhanced_image) - enhanced_image)
        x = x + r6 * (tf.square(x) - x)
        x = x + r7 * (tf.square(x) - x)
        enhanced_image = x + r8 * (tf.square(x) - x)
        return enhanced_image

    def call(self, data):
        dce_net_output = self.dce_model(data)
        return self.get_enhanced_image(data, dce_net_output)

    

    def train_step(self, data):
        with tf.GradientTape() as tape:
            output = self.dce_model(data)
            losses = self.compute_losses(data, output)

        gradients = tape.gradient(
            losses["total_loss"], self.dce_model.trainable_weights
        )
        self.optimizer.apply_gradients(zip(gradients, self.dce_model.trainable_weights))

        self.total_loss_tracker.update_state(losses["total_loss"])
        self.illumination_smoothness_loss_tracker.update_state(
            losses["illumination_smoothness_loss"]
        )
        self.spatial_constancy_loss_tracker.update_state(
            losses["spatial_constancy_loss"]
        )
        self.color_constancy_loss_tracker.update_state(losses["color_constancy_loss"])
        self.exposure_loss_tracker.update_state(losses["exposure_loss"])

        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        output = self.dce_model(data)
        losses = self.compute_losses(data, output)

        self.total_loss_tracker.update_state(losses["total_loss"])
        self.illumination_smoothness_loss_tracker.update_state(
            losses["illumination_smoothness_loss"]
        )
        self.spatial_constancy_loss_tracker.update_state(
            losses["spatial_constancy_loss"]
        )
        self.color_constancy_loss_tracker.update_state(losses["color_constancy_loss"])
        self.exposure_loss_tracker.update_state(losses["exposure_loss"])

        return {metric.name: metric.result() for metric in self.metrics}

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        """While saving the weights, we simply save the weights of the DCE-Net"""
        self.dce_model.save_weights(
            filepath,
            overwrite=overwrite,
            save_format=save_format,
           
        )

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        """While loading the weights, we simply load the weights of the DCE-Net"""
        self.dce_model.load_weights(
            filepath=filepath,
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            
        )



def model1(img):
    model = ZeroDCE()  
    model.compile(learning_rate=1e-4)
    model.load_weights("C:/Users/LENOVO/Desktop/imageUpscale/my_model_weights.h5")
    
    
    if img.mode != 'RGB':
        img = img.convert('RGB')

    
    
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    
    enhanced_img_array = model.predict(img_array)[0]  
    enhanced_img_array = np.clip(enhanced_img_array, 0, 1)  
    enhanced_img = (enhanced_img_array * 255).astype(np.uint8)  
    median_filtered_image1=cv2.bilateralFilter(enhanced_img,9,75,75)
    median_filtered_image=cv2.fastNlMeansDenoisingColored(enhanced_img, None,20, 10, 7,21)




    median_filtered_image = cv2.medianBlur(enhanced_img, 5)
    enhanced_img_pil = Image.fromarray(enhanced_img)


    return enhanced_img_pil

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * tf.math.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) / tf.math.log(10.0))

def compute_ssim(original_image, generated_image):

    original_image = tf.convert_to_tensor(original_image, dtype = tf.float32)
    generated_image = tf.convert_to_tensor(generated_image, dtype = tf.float32)

    ssim = tf.image.ssim(original_image, generated_image, max_val = 1.0, filter_size = 11, filter_sigma = 1.5, k1 = 0.01, )

    return tf.math.reduce_mean(ssim, axis = None, keepdims = False, name = None)


def model2(img):
    model_path = "C:/Users/LENOVO/Desktop/imageUpscale/mymodelsuper.h5"
    model = load_model(model_path, custom_objects={'PSNR': PSNR, 'compute_ssim': compute_ssim})

    if img.mode != 'RGB':
        img = img.convert('RGB')


    original_size = img.size  

    img = img.resize((256, 256))

    img_array = img_to_array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    enhanced_img_array = model.predict(img_array)[0]

    enhanced_img_array = np.clip(enhanced_img_array, 0, 1)
    enhanced_img = (enhanced_img_array * 255).astype(np.uint8)

    resized_img = Image.fromarray(enhanced_img).resize(original_size, Image.BILINEAR)

    return resized_img


def photo():
    st.header("Image UpScaling using ML Techniques")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        if analyze_image(img):
         im1=model1(img)
         #im2=model2(im1)
         
         st.image(img, caption='Original Image')
         st.image(im1, caption='Bright Image')
        #  st.image(im2, caption='Upscaled Image')
        else:
         im1=model2(img)
         st.image(img, caption='Original Image')
         st.image(im1, caption='Upscaled Image')
        

        

        if st.button("Download Upscaled Image"):
                download_img(im1)


def download_img(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    st.download_button(
        label="Click to download",
        data=img_bytes,
        file_name='upscaled_image.png',
        mime='image/png',
    )

def info():
    st.header("How  it Works?")
    st.subheader("ZERO-DCE MODEL")
    st.write("The Zero-Reference Deep Curve Estimation (Zero DCE) model is a state-of-the-art method in the field of image enhancement, particularly in addressing low-light conditions. Unlike traditional methods that rely on reference images or prior knowledge, the Zero DCE model operates without any reference input, hence the term 'zero-reference.'")
    st.write("At its core, the Zero DCE model utilizes deep neural networks to predict a transformation curve that can effectively enhance the brightness and visibility of low-light images. This transformation curve is learned directly from the input low-light image itself, without the need for additional reference images or external information.")
    st.write("The key innovation of the Zero DCE model lies in its ability to capture and exploit the inherent characteristics of low-light images to generate accurate and effective enhancement curves. By leveraging deep learning techniques, the model can adaptively adjust the brightness levels of pixels in the input image, effectively amplifying details and enhancing visibility without introducing excessive noise or artifacts.")
    image1=Image.open("C:/Users/LENOVO/Desktop/imageUpscale/images/img6.jpg")
    image2=Image.open("C:/Users/LENOVO/Desktop/imageUpscale/images/img6_res.jpg")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1, caption='Image with low brightness', use_column_width=True)
    with col2:
        st.image(image2, caption='Image after being passed through the model', use_column_width=True)

    st.subheader("SUPER RESOLUTION MODEL")
    st.write("ResNet architecture, initially developed for image classification, has been repurposed for the task of super-resolution. By embracing residual learning, where the model learns to predict the difference between low and high-resolution images, ResNet-based models excel at preserving critical visual details while enhancing image resolution. This strategy enables the model to focus on capturing fine-grained features crucial for improving image quality.")
    st.write("Integral to ResNet-based super-resolution models are skip connections, facilitating smoother gradient flow through the network and mitigating the vanishing gradient problem. These connections retain valuable information from earlier layers and help the model capture intricate image structures and long-range dependencies more effectively. During training, loss functions like mean squared error or perceptual loss functions measure the discrepancy between model predictions and ground truth images, guiding the optimization process.")
    st.write("Furthermore, ResNet-based super-resolution models employ upsampling techniques like bicubic interpolation or transposed convolutions to upscale low-resolution feature maps, enhancing the visual quality of generated images. This approach finds applications across diverse domains, including medical imaging, satellite imaging, surveillance, and the enhancement of low-quality videos or images. Leveraging ResNet's robust architecture, these models contribute significantly to advancing image processing and computer vision tasks, enabling the generation of high-quality images from low-quality inputs.")

    image3=Image.open("C:/Users/LENOVO/Desktop/imageUpscale/images/car1.jpg")
    image4=Image.open("C:/Users/LENOVO/Desktop/imageUpscale/images/car2.jpg")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image3, caption='Low-Resolution Image', use_column_width=True)
    with col2:
        st.image(image4, caption='Image after being passed through the model', use_column_width=True)

# Setup page layout and background
setup_page()

if selected_box == 'Home':
    home()
elif selected_box == 'Upscale your Image':
    photo()
elif selected_box == 'How it Works?':
    info()
