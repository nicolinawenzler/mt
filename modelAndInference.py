from collections import namedtuple

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
import torch.nn as nn
import cv2
from torch.autograd import Variable

from prior import GenerativeModel

ModelState = namedtuple('ModelState', ['image'])
GuideState = namedtuple('GuideState', ['control_points'])


class VariationalInferenceComponents(nn.Module):
	"""
	contains the optimization components
	"""
	def __init__(self, image_size, batch_size):
		super(VariationalInferenceComponents, self).__init__()
		self.image_size = image_size
		self.batch_size = batch_size
		self.predict = Predict().cuda()
		print("PREDICT", self.predict)

	def prior(self):
		"""
		calls the generative model to generate an image
		:return: the generated image
		"""
		model_instance = GenerativeModel()
		p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y, p6x, p6y, p7x, p7y, p8x, p8y, p9x, p9y, densities = model_instance.generate_values(self.batch_size)
		image_combined = model_instance.compute_and_draw_line(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y, p6x, p6y, p7x, p7y, p8x, p8y, p9x, p9y, densities, 32)
		#img_blurred = model_instance.ink_model(image_combined).cuda()
		img_blurred = image_combined.cuda()

		return img_blurred

	def model(self, data):
		"""
		compares the generated image to the given the observed data
		:param data: the observed data
		"""
		sigma = torch.Tensor([1]).cuda()
		image = self.prior()
		data_t = torch.Tensor(data).cuda()
		obs = pyro.sample('obs', dist.Normal(image, sigma), obs=data_t)

	def guide(self, data):
		"""
		the inference part
		:param data: the training data
		:return: the inferred points (only needed for visualization purpose, not for the crucial inference part)
		"""
		pyro.module('predict', self.predict)
		data_t = torch.from_numpy(data).cuda()

		data_from_nn = self.predict(data_t, self.batch_size)
		p_mu = data_from_nn[:, 0:18]
		p_mu = (p_mu*29+1).cuda()
		d_mu = data_from_nn[:, 18:]*0.8

		p1x = pyro.sample('p1x', dist.Uniform(p_mu[:, 0]-0.5, p_mu[:, 0]+0.5))
		p1y = pyro.sample('p1y', dist.Uniform(p_mu[:, 1]-0.5, p_mu[:, 1]+0.5))
		p2x = pyro.sample('p2x', dist.Uniform(p_mu[:, 2]-0.5, p_mu[:, 2]+0.5))
		p2y = pyro.sample('p2y', dist.Uniform(p_mu[:, 3]-0.5, p_mu[:, 3]+0.5))
		p3x = pyro.sample('p3x', dist.Uniform(p_mu[:, 4]-0.5, p_mu[:, 4]+0.5))
		p3y = pyro.sample('p3y', dist.Uniform(p_mu[:, 5]-0.5, p_mu[:, 5]+0.5))
		p4x = pyro.sample('p4x', dist.Uniform(p_mu[:, 6]-0.5, p_mu[:, 6]+0.5))
		p4y = pyro.sample('p4y', dist.Uniform(p_mu[:, 7]-0.5, p_mu[:, 7]+0.5))
		p5x = pyro.sample('p5x', dist.Uniform(p_mu[:, 8]-0.5, p_mu[:, 8]+0.5))
		p5y = pyro.sample('p5y', dist.Uniform(p_mu[:, 9]-0.5, p_mu[:, 9]+0.5))
		p6x = pyro.sample('p6x', dist.Uniform(p_mu[:, 10]-0.5, p_mu[:, 10]+0.5))
		p6y = pyro.sample('p6y', dist.Uniform(p_mu[:, 11]-0.5, p_mu[:, 11]+0.5))
		p7x = pyro.sample('p7x', dist.Uniform(p_mu[:, 12]-0.5, p_mu[:, 12]+0.5))
		p7y = pyro.sample('p7y', dist.Uniform(p_mu[:, 13]-0.5, p_mu[:, 13]+0.5))
		p8x = pyro.sample('p8x', dist.Uniform(p_mu[:, 14]-0.5, p_mu[:, 14]+0.5))
		p8y = pyro.sample('p8y', dist.Uniform(p_mu[:, 15]-0.5, p_mu[:, 15]+0.5))
		p9x = pyro.sample('p9x', dist.Uniform(p_mu[:, 16]-0.5, p_mu[:, 16]+0.5))
		p9y = pyro.sample('p9y', dist.Uniform(p_mu[:, 17]-0.5, p_mu[:, 17]+0.5))
		densities = torch.zeros(100, 8).cuda()
		for i in range(8):
			densities[:, i] = pyro.sample('dens{}'.format(i), dist.Uniform(d_mu[:, i], d_mu[:, i]+0.1))

		return p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y, p6x, p6y, p7x, p7y, p8x, p8y, p9x, p9y, densities

	def flatten(self, a_variable):
		"""
		reshapes a given variable
		:param a_variable:
		:return: the reshaped variable
		"""
		return a_variable.view(self.batch_size, int(a_variable.shape[1] ** 2))


class Predict(nn.Module):
	"""
	the neural network used to predict the parameters for the control points to create the character in a given image
	"""
	def __init__(self):
		"""
		initializes the neural network
		"""
		super(Predict, self).__init__()
		self.l0 = nn.Conv2d(1, 10, kernel_size=5, padding=2)
		self.l1 = nn.ReLU()
		self.l2 = nn.MaxPool2d(kernel_size=2)
		self.l3 = nn.Conv2d(10, 10, kernel_size=3, padding=1)
		self.l4 = nn.ReLU()
		self.l5 = nn.MaxPool2d(kernel_size=2)
		self.l6 = nn.Linear(640, 100)
		self.l6.weight.data.normal_(0.0, 0.3)
		self.l7 = nn.Sigmoid()
		self.l8 = nn.Linear(100, 26)
		self.l8.weight.data.normal_(0.0, 0.3)
		self.l9 = nn.Sigmoid()

	def forward(self, x, batch_size):
		"""
		runs the neural network
		:param x: the input (batch of images)
		:param batch_size: the batch size
		:return: the estimated control points
		"""
		x = self.l0(x)
		x = self.l1(x)
		x = self.l2(x)
		x = self.l3(x)
		x = self.l4(x)
		x = self.l5(x)
		x = x.view(batch_size, -1)
		x = self.l6(x)
		x = self.l7(x)
		x = self.l8(x)
		x = x * 2
		x = self.l9(x)
		control_points_mu = x
		return control_points_mu
