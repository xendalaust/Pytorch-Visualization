import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import os
import torch.nn as nn
import time

resnet = models.resnet152()
resnet.load_state_dict(torch.load("ckpt_collect/resnet152/resnet152.pt"), strict = False)

def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--use-cuda', action='store_true', default=True, 
                     help='Use NVIDIA GPU Acceleration')
	parser.add_argument('--image-path', type=str, default='./Missing_False/152/', 
                     help='Input Image Path')
	parser.add_argument('--save-path', type=str, default='./visual_result/152/',
                     help='Save Image Path')
	parser.add_argument('--ckpt-path', type=str, default='./ckpt_collect/resnet152/resnet152.pt',
	                    help='Input Image Path')
	parser.add_argument('--model',  type=str, default = models.resnet152(),
	                    help='Choose the Model')
	
	args = parser.parse_args()
	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
	    print("Using GPU for Acceleration")
	else:
	    print("Using CPU for Computation")

	return args

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = preprocessed_img.requires_grad_(True)
	return input

def show_cam_on_image(img, mask, file_name, save_path):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite(save_path + str(file_name) + "_GradCam.jpg", np.uint8(255 * cam))

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """

	def __init__(self, model, target_layers,use_cuda):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model, target_layers)
		self.cuda = use_cuda

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		output = output.view(output.size(0), -1)
		if self.cuda:
			output = output.cpu()
			output = resnet.fc(output).cuda()
		else:
			output = resnet.fc(output)
		return target_activations, output

class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index = None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.Tensor(torch.from_numpy(one_hot)).requires_grad_(True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.zero_grad()
		one_hot.backward(retain_graph=True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

		target = features[-1]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam

class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()
		for module in self.model.named_modules():
			module[1].register_backward_hook(self.backpg_relu)

	def backpg_relu(self, module, grad_in, grad_out):
		if isinstance(module, nn.ReLU):
			return (torch.clamp(grad_in[0], min=0.0),)

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			output = self.forward(input.cuda())
		else:
			output = self.forward(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.from_numpy(one_hot).requires_grad_(True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)
		#self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)
		output = input.grad.cpu().data.numpy()
		output = output[0,:,:,:]

		return output

if __name__ == '__main__':
	""" python grad_cam.py <path_to_image>
	1. Loads an image with opencv.
	2. Preprocesses it for VGG19 and converts to a pytorch variable.
	3. Makes a forward pass to find the category index with the highest score,
	and computes intermediate activations.
	Makes the visualization. """

	image   = []
	args    = get_args()

	model = models.resnet152()
	model.load_state_dict(torch.load(args.ckpt_path), strict = False)
	model_fc = models.resnet152()
	model_fc.load_state_dict(torch.load(args.ckpt_path), strict = False)
	
	del model.fc
	grad_cam = GradCam(model , \
					target_layer_names = ["layer4"], use_cuda = args.use_cuda)
	i = 0
	x = os.walk(args.image_path)
	for root, dirs, filename in x:
		print("root", root)
		print("dirs", dirs)
		print("filename", filename)
	for s in filename:
		image.append(cv2.imread(args.image_path + s,1))
	for img in image:
		img = np.float32(cv2.resize(img, (224, 224))) / 255
		input = preprocess_image(img)
		input.required_grad = True
		print('input.size() = ',input.size())
		file_name = filename[i][:-4]
		i += 1

		# If None, returns the map for the highest scoring category.
		# Otherwise, targets the requested index.
		target_index = None
		mask = grad_cam(input, target_index)
		
		show_cam_on_image(img, mask, file_name, args.save_path)
		
		gb_model = GuidedBackpropReLUModel(model_fc, use_cuda = args.use_cuda)
		gb = gb_model(input, index=target_index)
		gb = gb.transpose((1, 2, 0))
		cam_mask = cv2.merge([mask, mask, mask])
		cam_gb = deprocess_image(cam_mask * gb)
		gb = deprocess_image(gb)
		
		cv2.imwrite(args.save_path + str(file_name) + '_GuidedBackpropReLUModel.jpg', gb)
		cv2.imwrite(args.save_path + str(file_name) + '_DeprocessImage.jpg', cam_gb)
