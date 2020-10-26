import tensorflow as tf
import cv2
import numpy as np
import time

def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = img.copy()[:,:,::-1]
    for i in range(3):
        #mean = preprocessed_img[:,:,i].mean()
        preprocessed_img[:,:,i] = preprocessed_img[:,:,i]-means[i]
        #std = preprocessed_img[:,:,i].std()
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    #preprocessed_img = np.ascontiguousarray(preprocess_image, dtype=np.float32)
    preprocessed_img_tensor = tf.convert_to_tensor(preprocessed_img)
    preprocessed_img_tensor = tf.expand_dims(preprocessed_img_tensor, 0)
    return preprocessed_img_tensor

def np_to_tensor(img):
    if len(img.shape)<3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2,0,1)) # stimmt nicht
    output = tf.convert_to_tensor(output)
    output = tf.expand_dims(output,0)
    return output

def tv_norm(input, tv_beta):
    img = input[0,:,:,0]
    row_grad = tf.pow(tf.reduce_mean(tf.abs((img[:-1,:] - img[1:,:]))), tv_beta)
    col_grad = tf.pow(tf.reduce_mean(tf.abs((img[:,:-1] - img[:,1:]))), tv_beta)
    return row_grad + col_grad

def save(mask, img, blurred):
    mask = np.array(mask[0])
    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1-mask

    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap)/255
    cam = 1.0*heatmap + np.float32(img)/255
    cam = cam/np.max(cam)

    img = np.float32(img)/255
    perturbated = np.multiply(1-mask, img) + np.multiply(mask, blurred)

    cv2.imwrite('perturbated.png', np.uint8(255 * perturbated))
    cv2.imwrite('heatmap.png', np.uint8(255 * heatmap))
    cv2.imwrite('mask.png', np.uint8(255 * mask))
    cv2.imwrite('cam.png', np.uint8(255 * cam))


learning_rate = .1

tv_beta = 3
l1_coeff = 0.01
tv_coeff = 0.2

max_iterations = 500

model = tf.keras.applications.VGG19(weights='imagenet')
model.trainable = False

path =  '/home/student/PycharmProjects/tf2_meaningful_perturbation/examples/YellowLabradorLooking_new.jpg'
# 'examples/flute.jpg'
original_img = cv2.imread(path, 1)#('/home/student/PycharmProjects/pytorch_perturbation/pytorch-explain-black-box/examples/flute.jpg', 1)
original_img = cv2.resize(original_img, (224, 224))

img = np.float32(original_img) / 255

blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
blurred_img2 = np.float32(cv2.medianBlur(original_img, 11))/255
blurred_img_numpy = (blurred_img1 + blurred_img2) / 2

mask_init = np.ones((28, 28), dtype = np.float32)

img = preprocess_image(img)
blurred_img = preprocess_image(blurred_img2)

mask = np.float32([mask_init])
mask = tf.convert_to_tensor(mask)
mask = tf.expand_dims(mask,3)
mask = tf.Variable(mask, trainable=True)

upsample = tf.keras.layers.UpSampling2D(size=(8,8), interpolation='bilinear')
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

target = model(img)
#target = tf.nn.softmax(model(img))
category = np.argmax(target)
print('Category: ', category)
start = time.time()

# start with for loop
for i in range(50):
    elapsed_time = time.time() - start
    print('Iter ', (i + 1), (time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    upsampled_mask = upsample(mask)
    upsampled_mask = tf.tile(upsampled_mask, multiples=(1,1,1,3))

    perturbated_input = tf.math.multiply(img, upsampled_mask) + tf.math.multiply(blurred_img, 1-upsampled_mask)

    noise = np.zeros((224,224,3), dtype=np.float32)
    cv2.randn(noise, 0, 0.2)
    noise = np.float32([noise])
    noise = tf.convert_to_tensor(noise)

    perturbated_input = perturbated_input + noise

    outputs = model(perturbated_input)
    #outputs = tf.nn.softmax(model(perturbated_input))
    with tf.GradientTape() as tape:
        loss = l1_coeff*tf.reduce_mean(tf.abs(1-mask)) + tv_coeff*tv_norm(mask, tv_beta) + outputs[0, category]
    print('Loss ', i, loss)

    grads = tape.gradient(loss, [mask])
    optimizer.apply_gradients(zip(grads, [mask]))
    tf.clip_by_value(mask, clip_value_min=0, clip_value_max=1)

upsamled_mask = upsample(mask)
save(upsamled_mask, original_img, blurred_img_numpy)
