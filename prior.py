import cv2
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch.autograd import Variable
import torch.nn as nn
from visualize import vis

# generates a character type
class GenerativeModel():
	"""
	the generative model
	"""

	def generate_token_for_input(self, batch_size, given_coordinates=None):
		"""
		generates the control points for the input image
		:param batch_size: the batch size
		:param given_coordinates: if the coordinates are specified (to make the produced character readable)
		:return: the control points
		"""
		sigma = torch.Tensor([0.5]).repeat(batch_size).cuda()

		# sample control point coordinates if not given
		if given_coordinates is None:
			low = Variable(torch.zeros(10)).cuda()
			high = (Variable(torch.ones(10))*31).cuda()
			coordinates = pyro.sample('points', dist.Uniform(low, high))

		# use given coordinates
		else:
			coordinates = given_coordinates
		p1x = pyro.sample('p1x', dist.Normal(coordinates[0].cuda(), sigma))
		p1y = pyro.sample('p1y', dist.Normal(coordinates[1].cuda(), sigma))
		p2x = pyro.sample('p2x', dist.Normal(coordinates[2].cuda(), sigma))
		p2y = pyro.sample('p2y', dist.Normal(coordinates[3].cuda(), sigma))
		p3x = pyro.sample('p3x', dist.Normal(coordinates[4].cuda(), sigma))
		p3y = pyro.sample('p3y', dist.Normal(coordinates[5].cuda(), sigma))
		p4x = pyro.sample('p4x', dist.Normal(coordinates[6].cuda(), sigma))
		p4y = pyro.sample('p4y', dist.Normal(coordinates[7].cuda(), sigma))
		p5x = pyro.sample('p5x', dist.Normal(coordinates[8].cuda(), sigma))
		p5y = pyro.sample('p5y', dist.Normal(coordinates[9].cuda(), sigma))
		return p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y

	def generate_values(self, batch_size):
		"""
		samples values for the control points of a generated image
		:param batch_size: the size of the batch
		:return: the points
		"""

		sigma = torch.Tensor([0.5]).repeat(batch_size).cuda()

		low = Variable(torch.zeros(batch_size)).cuda()
		high = (Variable(torch.ones(batch_size))*31).cuda()

		points = pyro.sample('points', dist.Uniform(low, high))
		points = points.repeat(batch_size, 1)
		p1x = pyro.sample('p1x', dist.Normal(points[:, 0], sigma))
		p1y = pyro.sample('p1y', dist.Normal(points[:, 1], sigma))
		p2x = pyro.sample('p2x', dist.Normal(points[:, 2], sigma))
		p2y = pyro.sample('p2y', dist.Normal(points[:, 3], sigma))
		p3x = pyro.sample('p3x', dist.Normal(points[:, 4], sigma))
		p3y = pyro.sample('p3y', dist.Normal(points[:, 5], sigma))
		p4x = pyro.sample('p4x', dist.Normal(points[:, 6], sigma))
		p4y = pyro.sample('p4y', dist.Normal(points[:, 7], sigma))
		p5x = pyro.sample('p5x', dist.Normal(points[:, 8], sigma))
		p5y = pyro.sample('p5y', dist.Normal(points[:, 9], sigma))
		p6x = pyro.sample('p6x', dist.Normal(points[:, 10], sigma))
		p6y = pyro.sample('p6y', dist.Normal(points[:, 11], sigma))
		p7x = pyro.sample('p7x', dist.Normal(points[:, 12], sigma))
		p7y = pyro.sample('p7y', dist.Normal(points[:, 13], sigma))
		p8x = pyro.sample('p8x', dist.Normal(points[:, 14], sigma))
		p8y = pyro.sample('p8y', dist.Normal(points[:, 15], sigma))
		p9x = pyro.sample('p9x', dist.Normal(points[:, 16], sigma))
		p9y = pyro.sample('p9y', dist.Normal(points[:, 17], sigma))
		densities = Variable(torch.zeros(batch_size, 8)).cuda()
		for i in range(8):
			densities[:, i] = pyro.sample('dens{}'.format(i), dist.Uniform(torch.zeros(100).cuda(), torch.ones(100).cuda()))

		return p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y, p6x, p6y, p7x, p7y, p8x, p8y, p9x, p9y, densities

	def ink_model(self, img_batch):
		"""
		blurs the given image
		:param img:
		:return: the blurred image
		"""
		gaussian_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1).cuda()
		img_batch = img_batch.unsqueeze(1)
		gaussian_weights = torch.nn.Parameter(torch.Tensor([[[[0.024879, 0.107973, 0.024879],
		                                                      [0.107973, 0.468592, 0.107973],
		                                                      [0.024879, 0.107973, 0.024879]]]]).cuda(), requires_grad=False)
		gaussian_filter.weight = gaussian_weights
		img_blurred = gaussian_filter(img_batch)
		img_blurred = img_blurred.squeeze()
		return img_blurred

	def compute_and_draw_line(self, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y, p6x=None, p6y=None, p7x=None, p7y=None, p8x=None, p8y=None, p9x=None, p9y=None, densities=None, image_size=32):
		"""
		computes and draws lines into images, batchwise
		:param batch_of_substroke: vector of control points of substroke [batch_size, 5, 2]
		:return: batch of images
		"""
		image_combined = torch.zeros(100, image_size, image_size).cuda()
		vis_instance = vis()

		if densities is not None:
			points = [p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y, p6x, p6y, p7x, p7y, p8x, p8y, p9x, p9y]
			#points = torch.clamp()
			for i in range(8):
				#points[i] = torch.clamp(points[i], 0.0, 30.9)
				image_combined = vis_instance.render_line(x1=points[2*i], y1=points[i+1], x2=points[2*i+2], y2=points[2*i+3],
				                                          image_h=image_size, image_w=image_size, image=image_combined,
				                                          density=densities[:, i])
		else:
			points = [p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y]

			for i in range(4):
				image_combined = vis_instance.render_line(x1=points[2*i], y1=points[2*i+1], x2=points[2*i+2], y2=points[2*i+3], image_h=image_size,
				                                          image_w=image_size,
				                                          image=image_combined)
		return image_combined
