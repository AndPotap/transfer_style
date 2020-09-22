import time
import tensorflow as tf
from matplotlib import pyplot as plt
from utils import load_img
from utils import imshow
from utils import vgg_layers
from utils import StyleContentModel
from utils import tensor_to_image
from utils import train_step
from utils import print_stats_of_layers


# content_path = './imgs/paola_face.jpeg'
content_path = './imgs/lupillo_cobijas.jpg'
# style_path = './imgs/girl.jpg'
# style_path = './imgs/mandolin.jpg'
# style_path = './imgs/starry.jpg'
style_path = './imgs/dali1.jpg'
output_file = './imgs/styled.png'
save_progress = True
# epochs = 1
# steps_per_epoch = 1
epochs = 10
steps_per_epoch = 100
style_weight = 1.e-2
content_weight = 1.e4

content_image = load_img(content_path)
content_image = tf.image.rot90(content_image, k=3)
style_image = load_img(style_path)

if save_progress:
    plt.subplot(1, 2, 1)
    imshow(content_image, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style_image, 'Style Image')
    plt.savefig(output_file[:-4] + 'style.png')

x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
prediction_probabilities.shape

content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image * 255)

print_stats_of_layers(zip(style_layers, style_outputs))
extractor = StyleContentModel(style_layers, content_layers)
results = extractor(tf.constant(content_image))

print('Styles:')
print_stats_of_layers(results['style'].items())

print("Contents:")
print_stats_of_layers(results['content'].items())

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']
image = tf.Variable(content_image)
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

targs_style = style_targets, style_weight, num_style_layers
targs_content = content_targets, content_weight, num_content_layers

start = time.time()
step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image, extractor, opt,
                   targs_style, targs_content)
        print(".", end='')
    if save_progress:
        tensor_to_image(image).save(output_file[:-4] + str(n) + '.png')
    print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end - start))
tensor_to_image(image).save(output_file)
