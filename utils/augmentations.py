import torch
import math

class GaussianNoise(object):
    def __init__(self, factor = 1):
        """
        DESC
        ---
        Creates random gaussian noise that is added to 
        the tensors based on the standard deviations of columns
        scaled by a factor.
        ---
        INPUTS
        ---
        factor (float) : hyperparameter that scales the standard
        deviation of the tensor's columns before constructing the
        random gaussian noise tensor.
        ---
        RETURN
        ---
        torch.Tensor containing augmented data
        """
        self.factor = factor

    def __call__(self, data):
        column_stddev = torch.std(data, dim = 0)
        noise = torch.normal(0, self.factor, data.shape)
        noise = noise * column_stddev 
        return data + noise

class RotateColsRandomly(object):

    def __init__(self, col_1, col_2):
        """
        DESC
        ---
        An augmentation that accepts a tensor (representing a data table)
        and it rotates two columns in the dataset in a linear transformation.

        For example, if we plot col1 and col2 and x, and y axis and then rotate and plot
        again, the plot will be rotated by a random degree (0-360)
        ---
        INPUTS
        ---
        col_1 (int, required): the index [zero based] for the x column
        col_2 (int, required): the index [zero based] for the y column
        ---
        RETURN
        ---
        RotateColsRandomly instance

        """
        self.col_1 = col_1 # x-column
        self.col_2 = col_2 # y-column

    def __call__(self, x):

        # create an identity matrix of shape dim
        dims = x.shape[1]
        A = torch.eye(dims)

        # calculate random radians to rotate
        rad = torch.rand(1)[0] * 2 * math.pi

        # set the relevant cells in the transformation
        A[self.col_1, self.col_1] =  torch.cos(rad)
        A[self.col_1, self.col_2] = -torch.sin(rad)
        A[self.col_2, self.col_1] =  torch.sin(rad)
        A[self.col_2, self.col_2] =  torch.cos(rad)

        # transform
        return (A @ x.T).T