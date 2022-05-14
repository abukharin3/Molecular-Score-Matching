import numpy as np

def kl_divergence(real_data, sampled_data):
	real_data = real_data.numpy()
	x_min = min(min(sampled_data[:, 0]), min(real_data[:, 0]))
	x_max = max(max(sampled_data[:, 0]), max(real_data[:, 0]))
	y_min = min(min(sampled_data[:, 1]), min(sampled_data[:, 1]))
	y_max = max(max(sampled_data[:, 1]), max(sampled_data[:, 1]))
	real_hist, xbins, ybins = np.histogram2d(real_data[:, 0], real_data[:, 1], density=True)
	gen_hist, xbins1, ybins1 = np.histogram2d(sampled_data[:, 0], sampled_data[:, 1], bins=[xbins, ybins], density=True)

	kl = np.sum(np.where(real_hist != 0, real_hist * np.log(real_hist / gen_hist), 0))
	return kl
