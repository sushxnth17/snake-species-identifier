import os

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.applications import MobileNetV2


def main():
	data_dir = "dataset"
	image_size = (224, 224)
	batch_size = 32
	num_classes = 2

	# Load images from class subfolders (cobra, krait) with integer labels.
	train_ds = tf.keras.utils.image_dataset_from_directory(
		data_dir,
		labels="inferred",
		label_mode="int",
		image_size=image_size,
		batch_size=batch_size,
		shuffle=True,
	)

	# Normalize image pixels to match MobileNetV2 expected input range.
	train_ds = train_ds.map(
		lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
		num_parallel_calls=tf.data.AUTOTUNE,
	).prefetch(tf.data.AUTOTUNE)

	base_model = MobileNetV2(
		input_shape=(224, 224, 3),
		include_top=False,
		weights="imagenet",
	)
	base_model.trainable = False

	model = models.Sequential(
		[
			base_model,
			layers.GlobalAveragePooling2D(),
			layers.Dense(64, activation="relu"),
			layers.Dense(num_classes, activation="softmax"),
		]
	)

	model.compile(
		optimizer="adam",
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)

	model.fit(train_ds, epochs=3)
	model.save("snake_model.h5")


if __name__ == "__main__":
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
	main()
