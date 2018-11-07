import cv2
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from scipy import interpolate
from torch.autograd import Variable


# generates a character type
def generateType():
	"""
	determines the characteristics of the type
	- number of strokes
	- number of substrokes
	- relations
	and passes those to generateToken, which creates the concret token
	"""
	# sample number of strokes
	kappa = pyro.sample('kappa', dist.Categorical(Variable(torch.Tensor([0, 0.4, 0.3, 0.3]))))
	number_of_strokes = kappa.item()
	#is_first_stroke = True
	print("#strokes:", number_of_strokes)

	all_strokes = []
	scales = []

	# sample number of sub-strokes
	for i in range(number_of_strokes):
		number_of_substrokes = pyro.sample('ni', dist.Categorical(Variable(torch.Tensor([0, 0.6, 0.3, 0.15, 0.05]))))
		number_of_substrokes = number_of_substrokes.item()
		print("#substrokes:", number_of_substrokes)

		# generate stroke
		control_points, scales_substroke = generateStroke(number_of_substrokes)
		all_strokes.append(control_points)
		scales.append(scales_substroke)

		# sample relation to previous stroke
		relation = pyro.sample('rel_helper', dist.Categorical(Variable(torch.Tensor([0.25, 0.25, 0.25, 0.25]))))
		rel_helper = relation.item()

		# sample relation details
		# if it is the first stroke, it only can be independent (of the previous stroke)
		if i == 0:
			# J_i = pyro.sample('J_i', dist.multinomial, ps=ps_for_independent_relation_grid)
			# L_i = pyro.sample('L_i', dist.uniform)
			J_i, L_i = 0, 0
			relations_previous_strokes = np.array([('independent', J_i, L_i, 0)], dtype=[('relation', 'U11'), ('1_prevstro', 'i4'), ('substroke', 'i4'), ('3_tau', 'f')])
		else:
			relations_previous_strokes = sampleRelation(rel_helper, all_strokes, relations_previous_strokes)

	generateToken(number_of_strokes, all_strokes, relations_previous_strokes, scales)


def generateStroke(number_of_substrokes):
	"""
	generates a stroke with a given number of substrokes
	:param number_of_substrokes:
	:return:
	"""

	scales = Variable(torch.zeros(int(number_of_substrokes)))
	xi = Variable(torch.zeros(int(number_of_substrokes), 5, 2))

	# sample control points and scale
	alpha = Variable(torch.Tensor([1]))
	beta = Variable(torch.Tensor([1]))

	for j in range(number_of_substrokes):

		for point in range(5):
			xi[j, point, 0] = pyro.sample('point0', dist.Uniform(Variable(torch.Tensor([0])), Variable(torch.Tensor([31]))))
			xi[j, point, 1] = pyro.sample('point1', dist.Uniform(Variable(torch.Tensor([0])), Variable(torch.Tensor([31]))))

		scales[j] = pyro.sample('scale', dist.Gamma(alpha, beta))

		# translates the stroke to the coordinates (0, 0) to make it easier to translate whole stroke
		#subtrahent0 = np.repeat(xi[j, 0, 0].data.numpy(), 5, axis=0)
		#subtrahent1 = np.repeat(xi[j, 0, 1].data.numpy(), 5, axis=0)
		#subtrahent = Variable(torch.Tensor(5, 2))
		#subtrahent[:, 0] = torch.from_numpy(subtrahent0)
		#subtrahent[:, 1] = torch.from_numpy(subtrahent1)

	#xi[j, :, :] = xi[j, :, :] - subtrahent
	print("xi", xi)

	return xi, scales


# samples the relation to the previous stroke(s)
def sampleRelation(xi, previous_strokes, relation):
	"""
	determines details of the sampled relation to the previous stroke(s)
	:param xi: the sampled relation id
	:param previous_strokes: list of previous strokes
	:param relation: the previous relations
	:return: relation, the relation details
	"""
	length = len(previous_strokes)

	# independent
	if xi == 0:  # Todo set J_i und L_i
		# J_i = pyro.sample('J_i', dist.multinomial, ps=ps_for_independent_relation_grid)
		# L_i = pyro.sample('L_i', dist.uniform)
		J_i = 0
		L_i = 0
		new = np.array([('independent', J_i, L_i, 0)], dtype=[('relation', 'U11'), ('1_prevstro', 'i4'), ('substroke', 'i4'), ('3_tau', 'f')])
		relation = np.append(relation, new)

	# start
	elif xi == 1:
		if length == 2:  # there is only one stroke where the current one can be appended to
			u_i = Variable(torch.zeros(1))
		else:
			ps_for_start = Variable(torch.zeros(length - 1))
			ps_for_start[:] = 1 / (length - 1)
			u_i = pyro.sample('u_i', dist.Categorical(ps_for_start))
			u_i = u_i.item()
		new = np.array([('start', u_i, 0, 0)], dtype=[('relation', 'U11'), ('1_prevstro', 'i4'), ('substroke', 'i4'), ('3_tau', 'f')])
		relation = np.append(relation, new)

	# end
	elif xi == 2:
		if length == 2:
			u_i = Variable(torch.zeros(1))
		else:
			ps_for_end = Variable(torch.zeros(length - 1))
			ps_for_end[:] = 1 / (length - 1)
			u_i = pyro.sample('u_i', dist.Categorical(ps_for_end))
			u_i = u_i.item()

		new = np.array([('end', u_i, 0, 0)], dtype=[('relation', 'U11'), ('1_prevstro', 'i4'), ('substroke', 'i4'), ('3_tau', 'f')])
		relation = np.append(relation, new)

	# along
	elif xi == 3:
		if length == 2:
			u_i = v_i = Variable(torch.zeros(1))
		else:
			ps_for_along = Variable(torch.zeros(length - 1))
			ps_for_along[:] = 1 / (length - 1)
			u_i = pyro.sample('u_i', dist.Categorical(ps_for_along))
			u_i = u_i.item()-1
			ps_for_along2 = Variable(torch.zeros(previous_strokes[u_i].size(0)))
			ps_for_along2[:] = 1 / len(previous_strokes[u_i])
			v_i = pyro.sample('v_i', dist.Categorical(ps_for_along2))
			v_i = v_i.item()-1

		tau = pyro.sample('tau', dist.Uniform(Variable(torch.Tensor([0.01])), Variable(torch.Tensor([0.99]))))
		new = np.array([('along', u_i, v_i, tau)], dtype=[('relation', 'U11'), ('1_prevstro', 'i4'), ('substroke', 'i4'), ('3_tau', 'f')])
		relation = np.append(relation, new)
	return relation


def generateToken(kappa, control_points, relations, scales):
	"""
	draws the actual token
	:param kappa: number of strokes
	:param control_points: the sampled control points
	:param relations: the relations between the strokes
	:param scales: the scales of the single substrokes
	"""

	image = initialize_image(32)
	title = "#strokes: {} ".format(kappa)
	splines = []

	# iteration over strokes
	for i in range(kappa):
		stroke = control_points[i]
		start_position = get_start_position(relations, i, control_points, splines)
		splines_stroke = []
		title = title + "#substrokes: {} ".format(len(stroke))

		# iteration over substrokes
		for j in range(len(stroke)):
			substroke_scale = scales[i][j]
			scale_noise = pyro.sample("scale_noise", dist.Normal(Variable(torch.zeros(1)), Variable(torch.ones(1))))
			substroke_scale = substroke_scale + scale_noise
			scaled_substroke = stroke[j] * substroke_scale
			scaled_substroke += start_position
			pos_noise = pyro.sample("pos_noise", dist.Normal(Variable(torch.zeros(1)), Variable(torch.ones(1))))
			scaled_substroke[:, 1:] = scaled_substroke[:, 1:] + pos_noise

			# global translation
			transformed = transform_stroke(scaled_substroke)

			# compute splines
			spline_substroke, unew = compute_spline(transformed)
			splines_stroke.append(spline_substroke)
			start_position = scaled_substroke[-1, :]

			# draw substroke
			draw_substroke(spline_substroke, unew, image)
			control_points[i][j] = scaled_substroke

		splines.append(splines_stroke)

	# create, show and save the image
	img2 = image.astype(np.uint8)
	img_blurred = ink_model(img2)
	cv2.imwrite("figure.png", img_blurred)
	cv2.imshow(title, img_blurred)
	cv2.waitKey(0)



def initialize_image(size):
	"""
	initializes an image with the given size
	:param size: the size of the image in pixel
	:return: the image
	"""
	img = np.zeros((size, size, 3))
	return img



def get_start_position(relations, current_stroke, control_points, splines):
	"""
	computes the start position of a stroke
	:param relations: the relations between the strokes
	:param current_stroke: the current stroke
	:param control_points: the control points of all strokes
	:param splines: the splines of the substrokes
	:return: the start position as (x, y)-coordinates
	"""
	start_position = Variable(torch.zeros(1, 2))
	mean = Variable(torch.zeros(2))

	if relations['relation'][current_stroke] == 'independent':
		mean[:] = control_points[current_stroke][0, :][0]
		print("independent", mean)

	elif relations['relation'][current_stroke] == 'along':
		position_stroke = relations['1_prevstro'][current_stroke]
		position_substroke = relations['substroke'][current_stroke]
		tau = int(relations['3_tau'][current_stroke]*100)
		mean[:] = splines[0][position_stroke][position_substroke][tau]
		print("along", mean)

	elif relations['relation'][current_stroke] == 'start':
		mean[:] = control_points[current_stroke-1][0, :][0]
		print("start", mean)

	elif relations['relation'][current_stroke] == 'end':
		mean[:] = control_points[current_stroke-1][-1, :][-1]
		print("end", mean)

	else:
		raise NameError('You did not specify a correct relation between the strokes')

	var = Variable(torch.Tensor([0.1]))
	start_position[0, 0] = pyro.sample("get_start_position_first_dim", dist.Normal(mean[0], var))
	start_position[0, 1] = pyro.sample("get_start_position_second_dim", dist.Normal(mean[1], var))

	print("start position", start_position)
	return start_position


def transform_stroke(points):
	"""
	re-scales and translates the given substroke
	:param points: the control points of all strokes
	:return: the control points after translation and rescaling
	"""
	x_scale = pyro.sample("x_scale", dist.Normal(Variable(torch.Tensor([1])), Variable(torch.Tensor([0.01]))))
	y_scale = pyro.sample("y_scale", dist.Normal(Variable(torch.Tensor([1])), Variable(torch.Tensor([0.01]))))
	x_trans = pyro.sample("x_trans", dist.Normal(Variable(torch.Tensor([0])), Variable(torch.Tensor([0.01]))))
	y_trans = pyro.sample("y_trans", dist.Normal(Variable(torch.Tensor([0])), Variable(torch.Tensor([0.01]))))

	for p in range(5):
		points[p, 0] = (points[p, 0] * x_scale) + x_trans
		points[p, 1] = (points[p, 1] * y_scale) + y_trans
		#points[p, 0] = torch.clamp((points[p, 0] * x_scale) + x_trans, 0.5, 31.5)
		#points[p, 1] = torch.clamp((points[p, 1] * y_scale) + y_trans, 0.5, 31.5)
	return points


def compute_spline(substroke):
	"""
	computes the spline representation for a substroke
	:param substroke: the substroke to compute the spline for
	:return: the spline parameters
	"""

	listx = substroke.data.numpy()[:, 0]
	listy = substroke.data.numpy()[:, 1]
	tck, u = interpolate.splprep([listx, listy], s=0)
	unew = np.arange(0, 1.01, 0.01)
	out = interpolate.splev(unew, tck)

	return out, unew


def draw_substroke(out, unew, image):
	"""
	visualizes a substroked, represented by a spline
	:param out: the coordinates of the spline
	:param unew: the steps of the distance between the coordinates
	:param image: the image to draw the splines into
	"""
	outx = [int(out[0][j]) for j in range(unew.size)]
	outy = [int(out[1][j]) for j in range(unew.size)]

	for line in range(1, unew.size):
		cv2.line(image, (outx[line - 1], outy[line - 1]), (outx[line], outy[line]), (255, 255, 255), 1)


def ink_model(img):
	"""
	blurs the given image
	:param img: the image
	:return: the blurred image
	"""
	standard_dev = 0.7
	img_blur = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=standard_dev, sigmaY=standard_dev, borderType=0)

	return img_blur


generateType()
