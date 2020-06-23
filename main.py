"""Compositional pattern producing networks with Fourier features.
Based on https://arxiv.org/abs/2006.10739
"""

import yaml
import math
import os

import tensorflow as tf
import tensorflow_datasets as tfds
import imageio


CFG = yaml.safe_load(open("config.yaml", "r").read())


# ============================== DATA ==========================================
def get_coord_ints(y_dim, x_dim):
  """Return a 2x2 matrix where the values at each location are equal to the
  indices of that location
  """
  ys = tf.range(y_dim)[tf.newaxis]
  xs = tf.range(x_dim)[:, tf.newaxis]
  coord_ints = tf.stack([ys+xs-ys, xs+ys-xs], axis=2)
  return coord_ints


def generate_scaled_coordinate_hints(batch_size, img_dim=256):
  """Generally used as the input to a CPPN, but can also augment each layer
  of a ConvNet with location hints
  """
  spatial_scale = 1. / img_dim
  coord_ints = get_coord_ints(img_dim, img_dim)
  coords = tf.cast(coord_ints, tf.float32)
  coords = tf.stack([coords[:, :, 0] * spatial_scale,
                     coords[:, :, 1] * spatial_scale], axis=-1)
  coords = tf.tile(coords[tf.newaxis], [batch_size, 1, 1, 1])
  return coords


def get_imgs():
  """Read the six sample images from disk
  """
  imgs = []
  for img_name in os.listdir("images"):
    img = imageio.imread(os.path.join("images", img_name))
    imgs.append(img)
  imgs = tf.stack(imgs, axis=0)
  imgs = tf.cast(imgs, tf.float32) / 255.
  labels = tf.range(6)[:, tf.newaxis]
  return imgs, labels


# ============================== FOURIER MAPPING ===============================
def initialize_fourier_mapping_vector(m, sigma):
  d = 2
  B = tf.random.normal((m, d)) * sigma
  return B


def fourier_mapping(coords, B):
  """Preprocess each coordinate — scaled from [0, 1) — by converting each
  coordinate to a random fourier feature, as determined by a matrix with values
  samples from a Gaussian distribution.
  """
  sin_features = tf.math.sin((2 * math.pi) * (tf.matmul(coords, B, transpose_b=True)))
  cos_features = tf.math.cos((2 * math.pi) * (tf.matmul(coords, B, transpose_b=True)))
  features = tf.concat([sin_features, cos_features], axis=-1)
  return features


# ============================== NETWORK =======================================
class Generator(tf.keras.Model):
  """Compositional pattern-producing network, optionally using Fourier features.
  """
  def __init__(self, use_fourier_features):
    super(Generator, self).__init__()
    self.condition_embed = tf.keras.layers.Dense(CFG['units'])
    self.mlps = [tf.keras.layers.Dense(CFG['units']) for _ in range(CFG['layers'])]
    self.mlp_out = tf.keras.layers.Dense(3)
    self.B = initialize_fourier_mapping_vector(m=CFG['m'], sigma=CFG['sigma'])
    self.use_fourier_features = use_fourier_features


  def call(self, x):
    batch_size = x.shape[0]
    coords = generate_scaled_coordinate_hints(batch_size)
    if self.use_fourier_features:
      features = fourier_mapping(coords, B=self.B)
    else:
      features = coords
    for i, mlp in enumerate(self.mlps):
      if i == 0:
        cond_embed = self.condition_embed(x)
        cond_embed = tf.tile(cond_embed[:, tf.newaxis, tf.newaxis], [1, 256, 256, 1])
        x = mlp(features) + cond_embed
      else:
        x = mlp(x)
      x = tf.nn.relu(x)
    x = self.mlp_out(x)
    x = tf.nn.sigmoid(x)
    return x


# ============================== VISUALIZATION =================================
def draw_mosaic(rows:list):
  """Given a list of rows of images, return a mosaic image composing all of
  the images together.
  """
  rowcs = []
  for row in rows:
    rowc = tf.concat(row, axis=1)
    rowcs.append(rowc)
  mosaic = rowcs[0]
  if len(rowcs) > 1:
    for rowc in rowcs[1:]:
      mosaic = tf.concat([mosaic, rowc], axis=0)
  return mosaic


def preview_dataset():
  import plotly.express as px
  imgs = get_imgs()
  mosaic = draw_mosaic([
    [imgs[0], imgs[1], imgs[2]],
    [imgs[3], imgs[4], imgs[5]],
  ])
  fig = px.imshow(mosaic)
  fig.show()


def view_predictions(imgs, labels, model):
  import plotly.express as px
  pred_imgs = model(labels)
  mosaic = draw_mosaic([
    tf.unstack(imgs, axis=0),
    tf.unstack(pred_imgs, axis=0)
  ])
  fig = px.imshow(mosaic)
  fig.show()


def make_video(path, normal_samples, fourier_samples, ground_truth):
  writer = imageio.get_writer(path)
  for normal_sample, fourier_sample in zip(normal_samples, fourier_samples):
    mosaic = draw_mosaic([
      tf.unstack(normal_sample, axis=0),
      tf.unstack(fourier_sample, axis=0),
      tf.unstack(ground_truth, axis=0)
    ])
    writer.append_data(mosaic.numpy())
  writer.close()


if __name__ == "__main__":
  imgs, labels = get_imgs()
  model = Generator(use_fourier_features=CFG['use_fourier_features'])
  model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(lr=1e-3)
  )
  model.fit(x=labels, y=imgs, batch_size=6, epochs=1000)
  view_predictions(imgs, labels, model)
