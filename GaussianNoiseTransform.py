import torch


# Noise is Gaussian with 10 percent cross-section
class GaussianNoiseTransform(object):
    def __init__(self, mean=0., std=1., k=25):
        self.std = std
        self.mean = mean
        self.k = k

    def __call__(self, tensor):

        for i, wnd in enumerate(tensor):
            # reshape and flatten
            x_transf = torch.flatten(wnd, start_dim=0)

            n = x_transf.size(0)
            perm = torch.randperm(n)
            idx = perm[:(n - self.k)]

            noise = torch.randn(x_transf.size())
            # only 10% is noise
            noise[idx] = 0.

            corrupted_image = x_transf + noise * self.std + self.mean

            tensor[i, :] = torch.unsqueeze(corrupted_image,1)

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
