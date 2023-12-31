import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

strategy = tf.distribute.MirroredStrategy()

print("TensorFlow version:", tf.__version__)
print(tf.config.list_physical_devices('GPU'))

@tf.keras.saving.register_keras_serializable()
class MyModel(tf.keras.Model):
		def __init__(self):
			super(MyModel, self).__init__()
			self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
			self.flatten = tf.keras.layers.Flatten()
			self.d1 = tf.keras.layers.Dense(128, activation='relu')
			self.d2 = tf.keras.layers.Dense(10)

		def call(self, x):
			x = self.conv1(x)
			x = self.flatten(x)
			x = self.d1(x)
			return self.d2(x)
		
# Create an instance of the model
model = MyModel()

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices(
		(x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


if(input("train model (y/n): ") == "y"):
	mnist = tf.keras.datasets.mnist

	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0

	# Add a channels dimension
	x_train = x_train[..., tf.newaxis].astype("float32")
	x_test = x_test[..., tf.newaxis].astype("float32")

	train_ds = tf.data.Dataset.from_tensor_slices(
			(x_train, y_train)).shuffle(10000).batch(32)

	test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

	

	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	optimizer = tf.keras.optimizers.Adam()

	train_loss = tf.keras.metrics.Mean(name='train_loss')
	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

	test_loss = tf.keras.metrics.Mean(name='test_loss')
	test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

	@tf.function
	def train_step(images, labels):
		with tf.GradientTape() as tape:
			# training=True is only needed if there are layers with different
			# behavior during training versus inference (e.g. Dropout).
			predictions = model(images, training=True)
			loss = loss_object(labels, predictions)
		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		train_loss(loss)
		train_accuracy(labels, predictions)

	@tf.function
	def test_step(images, labels):
		# training=False is only needed if there are layers with different
		# behavior during training versus inference (e.g. Dropout).
		predictions = model(images, training=False)
		t_loss = loss_object(labels, predictions)

		test_loss(t_loss)
		test_accuracy(labels, predictions)

	EPOCHS = 3

	for epoch in range(EPOCHS):
		# Reset the metrics at the start of the next epoch
		train_loss.reset_states()
		train_accuracy.reset_states()
		test_loss.reset_states()
		test_accuracy.reset_states()

		for images, labels in train_ds:
			train_step(images, labels)

		for test_images, test_labels in test_ds:
			test_step(test_images, test_labels)

		print(
			f'Epoch {epoch + 1}, '
			f'Loss: {train_loss.result()}, '
			f'Accuracy: {train_accuracy.result() * 100}, '
			f'Test Loss: {test_loss.result()}, '
			f'Test Accuracy: {test_accuracy.result() * 100}'
		)
	model.save("./test.keras")
im = Image.open('test.png', 'r').convert('L')
im = tf.convert_to_tensor(list(im.getdata())) / 255
im = tf.reshape(im,(1,28,28,1))
model = tf.keras.models.load_model('./test.keras')
for test_images, test_labels in test_ds:
	max = -float("inf")
	max_i = 0
	for i, value in enumerate(tf.squeeze(model.call(im))):
		if value > max:
			max_i = i
			max = value
	print(max_i)
	print(model.call(im))
	break


first_image = im
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

