import torch
import numpy as np
from torchvision.transforms import functional as F


ECG_CHANNEL_MEAN = [-0.0015939727891236544, -0.0013031433336436749, 0.00029035151237621903, 0.0014302238123491406, -0.0008918124949559569, -0.00048305661766789854, 0.00017116559320129454, -0.0009138658060692251, -0.001489124959334731, -0.0017458670772612095, -0.0007674962398596108, -0.0020748174283653498]
ECG_CHANNEL_STD = [0.1553376317024231, 0.15989091992378235, 0.16283635795116425, 0.1345466524362564, 0.13730694353580475, 0.14131711423397064, 0.22873035073280334, 0.33088698983192444, 0.32615572214126587, 0.29820892214775085, 0.26580968499183655, 0.27013346552848816]


class RandomCrop1D:
    def __init__(self, target_size):
        """
        Args:
            target_size (int): Desired number of timesteps after cropping.
        """
        self.target_size = target_size

    def __call__(self, sample):
        """
        Args:
            sample (torch.Tensor): Time series data of shape (time_steps, num_channels).
        
        Returns:
            torch.Tensor: Randomly cropped time series data of shape (target_size, num_channels).
        """
        timesteps, num_channels = sample.shape
        
        if timesteps < self.target_size:
            raise ValueError("Input timesteps must be greater than or equal to the target size.")
        
        # Randomly select the starting index for the crop
        start_idx = torch.randint(0, timesteps - self.target_size + 1, (1,)).item()
        
        # Crop the sample
        cropped_sample = sample[start_idx:start_idx + self.target_size, :]
        
        return cropped_sample


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.01):
        """
        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        noise = torch.randn_like(sample) * self.std + self.mean
        return sample + noise


class AmplitudeScale1D:
    def __init__(self, scale_range=(0.8, 1.2), per_channel=False):
        """
        Args:
            scale_range (tuple): Range for the random scaling factor.
            per_channel (bool): If True, apply different scaling factors for each channel.
        """
        self.scale_range = scale_range
        self.per_channel = per_channel

    def __call__(self, sample):
        timesteps, num_channels = sample.shape
        if self.per_channel:
            # Different scaling factor for each channel
            scales = torch.FloatTensor(num_channels).uniform_(*self.scale_range)
            return sample * scales.unsqueeze(0)  # Scale each channel independently
        else:
            # Same scaling factor for all channels
            scale = torch.FloatTensor(1).uniform_(*self.scale_range).item()
            return sample * scale


class RandomRescale1D:
    def __init__(self, scale_range=(0.95, 1.05)):
        """
        Args:
            scale_range (tuple): Range for rescaling the time dimension.
        """
        self.scale_range = scale_range

    def __call__(self, sample):
        timesteps, num_channels = sample.shape
        scale_factor = torch.FloatTensor(1).uniform_(*self.scale_range).item()
        scaled_length = int(timesteps * scale_factor)
        
        # Use interpolate for 1D rescaling
        resized_sample = torch.nn.functional.interpolate(
            sample.T.unsqueeze(0),  # (num_channels, timesteps) -> (1, num_channels, timesteps)
            size=scaled_length,
            mode="linear",
            align_corners=False
        ).squeeze(0).T  # Back to (scaled_length, num_channels)

        return resized_sample


class RandomResizedCrop1D:
    def __init__(self, target_size, scale=(0.08, 1.0)):
        """
        Args:
            target_size (int): The number of time steps after cropping and resizing.
            scale (tuple): Range of proportion of the original time series to select.
        """
        self.target_size = target_size
        self.scale = scale

    def __call__(self, sample):
        """
        Args:
            sample (np.array): Time series sample of shape (time_steps, num_channels).
        
        Returns:
            torch.Tensor: Cropped and resized time series data of shape (target_size, num_channels).
        """
        time_steps, num_channels = sample.shape

        # Determine the crop size within the defined scale
        crop_size = int(np.random.uniform(*self.scale) * time_steps)
        start = np.random.randint(0, time_steps - crop_size + 1)
        
        # Crop the sample
        cropped_sample = sample[start:start + crop_size, :]
        
        # Resize to the target size
        # F.resize: input is expected to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
        resized_sample = F.resize(cropped_sample.T.unsqueeze(-1), size=(self.target_size, 1))
        
        # Convert back to (scaled_size, num_channels)
        resized_sample = resized_sample.squeeze(-1).T
        
        return resized_sample
    

class RandomRescaleCrop1D:
    def __init__(self, target_size, scale_range=(0.5, 1.5)):
        """
        Args:
            target_size (int): Desired number of time steps after cropping.
            scale_range (tuple): Range of scaling factors to resize the time series.
        """
        self.target_size = target_size
        self.scale_range = scale_range

    def __call__(self, sample):
        """
        Args:
            sample (np.array): Time series sample of shape (time_steps, num_channels).
        
        Returns:
            torch.Tensor: Rescaled and cropped time series data of shape (target_size, num_channels).
        """
        time_steps, num_channels = sample.shape

        # Randomly choose a scale factor
        scale_factor = np.random.uniform(*self.scale_range)
        scaled_size = int(time_steps * scale_factor)
        
        # Resize the sample to the scaled size
        # F.resize: input is expected to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
        resized_sample = F.resize(sample.T.unsqueeze(-1), size=(scaled_size, 1))
        
        # Convert back to (scaled_size, num_channels)
        resized_sample = resized_sample.squeeze(-1).T
        
        # If the resized sample is larger than the target size, crop it
        if scaled_size > self.target_size:
            start = np.random.randint(0, scaled_size - self.target_size + 1)
            cropped_sample = resized_sample[start:start + self.target_size, :]
        else:
            # If smaller, pad to reach the target size
            pad_size = self.target_size - scaled_size
            cropped_sample = torch.nn.functional.pad(resized_sample, (0, 0, 0, pad_size), mode='constant', value=0)

        return cropped_sample


class CenterCrop1D:
    def __init__(self, target_size):
        """
        Args:
            target_size (int): The desired number of timesteps after cropping.
        """
        self.target_size = target_size

    def __call__(self, sample):
        """
        Args:
            sample (torch.Tensor): Time series data of shape (time_steps, num_channels).
        
        Returns:
            torch.Tensor: Center-cropped time series data of shape (target_size, num_channels).
        """
        timesteps, num_channels = sample.shape
        
        if timesteps < self.target_size:
            raise ValueError("Input timesteps must be greater than or equal to the target size.")
        
        # Calculate the starting index for center cropping
        start = (timesteps - self.target_size) // 2
        end = start + self.target_size
        
        # Perform center crop
        cropped_sample = sample[start:end, :]
        
        return cropped_sample


class RandomZeroOut1D:
    def __init__(self, zero_ratio_range=(0.1, 0.2)):
        """
        Args:
            zero_ratio_range (tuple): A tuple specifying the minimum and maximum proportion
                                      of timesteps to be zeroed out, e.g., (0.1, 0.2).
        """
        self.zero_ratio_range = zero_ratio_range

    def __call__(self, sample):
        """
        Args:
            sample (torch.Tensor): Time series data of shape (time_steps, num_channels).
        
        Returns:
            torch.Tensor: Time series data with a random period zeroed out.
        """
        timesteps, num_channels = sample.shape

        # Determine a random zero ratio within the specified range
        zero_ratio = torch.FloatTensor(1).uniform_(*self.zero_ratio_range).item()
        
        # Calculate the length of the zeroed segment
        zero_length = int(timesteps * zero_ratio)
        
        # Randomly select the start index for the zeroed segment
        start_idx = torch.randint(0, timesteps - zero_length + 1, (1,)).item()

        # Zero out the selected segment
        sample[start_idx:start_idx + zero_length, :] = 0
        
        return sample
    
class Transpose:
    def __init__(self, dims):
        """
        Args:
            dims (tuple): The dimensions to swap, e.g., (0, 1) to switch between
                          (time_steps, num_channels) and (num_channels, time_steps).
        """
        self.dims = dims

    def __call__(self, sample):
        """
        Args:
            sample (torch.Tensor): Input tensor to be transposed.
        
        Returns:
            torch.Tensor: Transposed tensor.
        """
        return sample.transpose(*self.dims)


class Normalize1D:
    def __init__(self, mean, std, transpose_back=False):
        """
        Args:
            mean (list): List of mean values for each channel.
            std (list): List of standard deviation values for each channel.
        """
        self.mean = mean
        self.std = std
        # self.normalize = Normalize(mean=mean, std=std)
        self.transpose_back = transpose_back

    def __call__(self, sample):
        """
        Args:
            sample (torch.Tensor): Time series data of shape (time_steps, num_channels).
        
        Returns:
            torch.Tensor: Normalized data with shape (time_steps, num_channels).
        """
        # Transpose to (num_channels, time_steps)
        sample = sample.T

        # Nomalize: Expected tensor to be a tensor image of size (..., C, H, W)
        sample = sample.unsqueeze(-1)
        
        # Apply normalization
        sample = F.normalize(sample, mean=self.mean, std=self.std)
        sample = sample.squeeze(-1)
        if self.transpose_back:
            sample = sample.T
        
        return sample