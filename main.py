import cv2
import numpy as np
import pyro.optim as optim
import torch
from pyro.infer import SVI, Trace_ELBO
from torch.autograd import Variable

from modelAndInference import VariationalInferenceComponents
from prior import GenerativeModel
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def main(**kwargs):

	def visualize_control_points(image, step):
		"""
		visualizes the control points to see different iterations steps. Does not take part in optimization process.
		:param image: the input image
		:param step: the current optimization step
		:return:
		"""
		# predicts the control points
		p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y, p6x, p6y, p7x, p7y, p8x, p8y, p9x, p9y, densities = varicomp.guide(image)
		# generates an image with given control points
		img = generative_model.compute_and_draw_line(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y, p6x, p6y, p7x, p7y, p8x, p8y, p9x, p9y, densities=densities, image_size=32)
		#img_blurred = model_instance.ink_model(img)
		img_blurred = img
		img_to_save = img_blurred.data.cpu().numpy()
		cv2.imwrite("step{}".format(step) +"00.png", img_to_save[0]*255)
		cv2.imwrite("step{}".format(step) +"42.png", img_to_save[42]*255)

	def visualize_loss(loss, step):
		"""
		visualizes the loss
		:param loss: current loss
		:param step: current optimization step
		:return:
		"""
		loss_arr = np.asarray(loss)
		plt.plot(loss_arr, linewidth=2)
		plt.text(2, 2, "loss")
		plt.savefig("loss{}".format(step)+".png")

	# set global parameters and create instances
	image_size = 32
	batch_size = 100
	num_optimization_steps = 10000
	adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
	varicomp = VariationalInferenceComponents(image_size=image_size, batch_size=batch_size)
	generative_model = GenerativeModel()
	if not torch.cuda.is_available():
		print("Error: Cuda need to be available in this configuration.")

	# generate control points for input image
	coordinates_N=torch.Tensor([26, 4, 6, 4, 26, 20, 6, 20, 6, 22])
	p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y = generative_model.generate_token_for_input(batch_size, given_coordinates=coordinates_N)

	# generate input image with sampled control points
	images = generative_model.compute_and_draw_line(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y)
	images_blurred = generative_model.ink_model(images)

	# create svi instance
	svi = SVI(varicomp.model, varicomp.guide, optim.Adam(adam_params), loss=Trace_ELBO(num_particles=100))

	# (prepare and) save input images
	images_prepared = images_blurred[:, :, :].detach().cpu().numpy()
	cv2.imwrite("inputimage00.png", (images_prepared[0]*255))
	cv2.imwrite("inputimage42.png", (images_prepared[42]*255))

	images_ready = np.expand_dims(images_prepared, 1)
	lossgraph = []

	# do actual optimization
	for step in range(num_optimization_steps):
		perm = np.random.permutation(images_ready)
		mini_batch = perm[0:batch_size, :, :]
		loss = svi.step(mini_batch)

		if step > 0 and step % 1 == 0:
			if not np.isinf(loss):
				lossgraph.append(loss)
			else:
				print("step:", step, "is inf", loss)
			if step < 10:
				print("step:", step, "loss:", loss)
				visualize_control_points(perm[0:batch_size], step)
			elif step % 25 == 0:
				print("step:", step, "loss:", loss)
				visualize_control_points(perm[0:batch_size], step)
				visualize_loss(lossgraph, step)


main()
