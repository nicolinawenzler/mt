import torch
import cv2
import numpy as np


class vis:
	def to_1d_index(self, xs, ys, height, width):
		"""
		Take regular indices into images and turn them into 1d-ones
		:param xs: indices into the first dimension of the images [n, steps]
		:param ys: indices into the second dimension of the images [n, steps]
		:param height: height of the images
		:param width: width of the images
		:return: a list of 1d indices [n, steps]
		"""
		xs = xs.long().cuda()
		ys = ys.long().cuda()
		batch_size = xs.shape[0]
		base = (torch.arange(batch_size).long().cuda() * height * width).view((-1, 1))
		result = base + (xs * width + ys)
		return result

	def draw_points_bilinear(self, image, xs, ys, density=torch.ones(1)):
		"""
		draw a lists of points into given images
		:param image: batch of images [n, h, w]
		:param xs: batch of x coordinates [n, steps]
		:param ys: batch of y coordinates [n, steps]
		:param density: density of points. Used to adjust intensity
		:return: rendered images [n, h, w]
		"""
		density = density.view((-1, 1))
		height = image.shape[1]
		width = image.shape[2]
		x1 = torch.floor(xs)
		x2 = x1 + 1
		y1 = torch.floor(ys)
		y2 = y1 + 1

		w22 = ((xs - x1) * (ys - y1)) / density
		w21 = ((xs - x1) * (y2 - ys)) / density
		w12 = ((x2 - xs) * (ys - y1)) / density
		w11 = ((x2 - xs) * (y2 - ys)) / density

		image = image.put_(self.to_1d_index(x1, y1, height, width), w11, accumulate=True)
		image = image.put_(self.to_1d_index(x1, y2, height, width), w12, accumulate=True)
		image = image.put_(self.to_1d_index(x2, y1, height, width), w21, accumulate=True)
		image = image.put_(self.to_1d_index(x2, y2, height, width), w22, accumulate=True)

		return torch.clamp(image, 0.0, 1.0)

	def render_line(self, x1, y1, x2, y2, image_h, image_w, image, density=0.5):
		"""
		Render a batch of lines
		:param x1: vector of x coordinates [n]
		:param y1: vector of y coordinates [n]
		:param x2: vector of x coordinates [n]
		:param y2: vector of y coordinates [n]
		:param image_h: height of the images to be rendered
		:param image_w: width of the images to be rendered
		:return: batch of images [n, h, w]
		"""
		steps = 2 * image_w
		dist = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2).cuda()
		lin = torch.linspace(0.0, 1.0, steps=steps).view(1, steps).cuda()
		xs = x1.view((-1, 1)) + lin * (x2 - x1).view((-1, 1))
		ys = y1.view((-1, 1)) + lin * (y2 - y1).view((-1, 1))

		image = self.draw_points_bilinear(image, xs, ys, density=density*steps / dist)
		return image

	def color_image(self, image, value):
		image[:] = value
		return image
