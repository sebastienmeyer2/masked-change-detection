"""OmniMAE for image reconstruction and change detection."""


from torch import nn
from torch.nn import Module, Sequential
from torch.nn import Conv2d, BatchNorm2d, LogSoftmax, ReLU, Conv3d, Linear

import einops


from omnimae.omni_mae_model import vit_base_mae_pretraining


from models.images_to_video import create_video, create_mask
from models.fresunet import FresUNet
from models.omnimae_loss import unpatchify


class OmniMAEPair(Module):
    """Take a pair of images as input and pass them through OmniMAE as a sequence."""
    def __init__(
        self, omnimae, video_ordering="1221", patch_shape=[2, 16, 16], nb_channels=3,
        masking_method="none", masking_proportion=0.5, img_height=224, img_width=224
    ):

        super().__init__()

        # Parameters
        self.video_ordering = video_ordering
        self.patch_shape = patch_shape
        self.masking_method = masking_method
        self.masking_proportion = masking_proportion

        self.nb_channels = nb_channels
        self.img_height = img_height
        self.img_width = img_width

        # Use OmniMAE as it is (freeze its parameters)
        self.omnimae = omnimae
        for param in self.omnimae.parameters():
            param.requires_grad = False

        # Finetune more layers if needed
        if self.nb_channels != 3:

            # Extend first layer
            layer = self.omnimae.trunk.patch_embed.proj._modules["1"]

            new_layer = Conv3d(
                in_channels=self.nb_channels,
                kernel_size=[2, 16, 16],
                out_channels=768,
                stride=[2, 16, 16],
            )

            new_layer.weight.requires_grad = False

            copy_weights = 0

            new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()

            for i in range(self.nb_channels - layer.in_channels):
                channel = layer.in_channels + i
                new_layer.weight[:, channel:channel+1, :, :] = \
                    layer.weight[:, copy_weights:copy_weights+1, :, :].clone()
            new_layer.weight = nn.Parameter(new_layer.weight)

            self.omnimae.trunk.patch_embed.proj._modules["1"] = new_layer

            # Extend last layer
            layer = self.omnimae.head

            new_layer = Linear(
                in_features=384,
                out_features=self.nb_channels * 512,
                bias=True
            )

            new_layer.weight.requires_grad = False

            copy_weights = 0

            new_layer.weight[:layer.out_features, :] = layer.weight.clone()

            for i in range(self.nb_channels - 3):
                new_layer.weight[(3+i)*512:(3+i+1)*512, :] = \
                    layer.weight[:512, :].clone()
            new_layer.weight = nn.Parameter(new_layer.weight)

            self.omnimae.head = new_layer

    def finetune_first_last_layers(self):

        self.omnimae.trunk.patch_embed.proj._modules["1"].weight.requires_grad = True
        self.omnimae.head.weight.requires_grad = True

    def forward(self, img_1, img_2):

        # Repeat the input if needed
        if img_1.ndim == 3:
            img_1 = einops.repeat(img_1, "c h w -> b c h w", b=1)
            img_2 = einops.repeat(img_2, "c h w -> b c h w", b=1)

        # Create the video
        video, true_video = create_video(img_1, img_2, video_ordering=self.video_ordering)

        # Create the associated mask
        mask = create_mask(
            batch_size=video.shape[0], video_ordering=self.video_ordering,
            patch_shape=self.patch_shape, img_height=self.img_height, img_width=self.img_width,
            masking_method=self.masking_method, masking_proportion=self.masking_proportion
        )

        # Forward the OmniMAE part
        _, decoder_patch_features = self.omnimae.trunk(video, mask=mask)
        output = self.omnimae.head(decoder_patch_features)

        return output, true_video, mask


class CNNHead(Module):

    def __init__(
        self, kernel_size=5, nb_channels=3, img_height=224, img_width=224
    ):

        super().__init__()

        # Parameters
        self.kernel_size = kernel_size
        self.nb_channels = nb_channels
        self.img_height = img_height
        self.img_width = img_width

        # Create a convolutional part which will output the change map
        self.cnn = Sequential(
            Conv2d(
                self.nb_channels, self.nb_channels, kernel_size=self.kernel_size,
                padding=self.kernel_size//2, stride=1, bias=False
            ),
            BatchNorm2d(self.nb_channels),
            ReLU(),
            Conv2d(
                self.nb_channels, 2, kernel_size=self.kernel_size, padding=self.kernel_size//2,
                stride=1
            ),
            LogSoftmax(dim=1)
        )

    def forward(self, img):

        output = self.cnn(img)

        return output


class OmniMAECNN(Module):
    """Take a pair of images as input and uses OmniMAE to perform change detection."""
    def __init__(
        self, omnimae_pair, kernel_size=5, patch_shape=[2, 16, 16], nb_channels=3, img_height=224,
        img_width=224
    ):

        super().__init__()

        # Parameters
        self.kernel_size = kernel_size

        self.patch_shape = patch_shape
        self.nb_channels = nb_channels
        self.img_height = 224
        self.img_width = 224

        # Use OmniMAE as it is (freeze its parameters)
        self.omnimae_pair = omnimae_pair

        # Create a convolutional part which will output the change map
        self.cnn = CNNHead(
            kernel_size=kernel_size, nb_channels=nb_channels, img_height=img_height,
            img_width=img_width
        )

    def forward(self, img_1, img_2):

        # Perform pass with OmniMAE
        output, _, _ = self.omnimae_pair(img_1, img_2)

        # Reshape the decoded patches to channels
        output = output.reshape(
            (*output.shape[:-1], output.shape[-1] // self.nb_channels, self.nb_channels)
        )

        # Unpatchify the predicted patches
        output = unpatchify(
            output, patch_shape=self.patch_shape, nb_channels=self.nb_channels,
            img_height=self.img_height, img_width=self.img_width
        )

        # Compute the difference of images
        first_output = output[:, :, 0, ...]
        last_output = output[:, :, 2, ...]

        output = (last_output - first_output).abs()

        # Output a change map
        output = self.cnn(output)

        return output


class OmniMAEFresUNet(Module):
    """Put together OmniMAE and FresUNet for change detection."""
    def __init__(
        self, input_nbr, label_nbr, nb_channels=3, patch_shape=[2, 16, 16], img_height=224,
        img_width=224, omnimae_pair=None
    ):

        super().__init__()

        # Parameters
        self.nb_channels = nb_channels
        self.patch_shape = patch_shape
        self.img_height = img_height
        self.img_width = img_width

        # Models
        self.omnimae_pair = omnimae_pair

        self.fresunet = FresUNet(input_nbr, label_nbr)

    def forward(self, x1, x2):

        if self.omnimae_pair is not None:

            # Pass OmniMAE
            output, _, _ = self.omnimae_pair(x1, x2)

            # Reshape the decoded patches to channels
            output = output.reshape(
                (*output.shape[:-1], output.shape[-1] // self.nb_channels, self.nb_channels)
            )

            # Unpatchify the predicted patches
            output = unpatchify(
                output, patch_shape=self.patch_shape, nb_channels=self.nb_channels,
                img_height=self.img_height, img_width=self.img_width
            )

            # Retrieve images
            img_1 = output[:, :, 0, :, ...]
            img_2 = output[:, :, 2, :, ...]

        x = self.fresunet(img_1, img_2)

        return x


def load_pretrained_model(device="cuda"):
    """Load checkpoint of OmniMAE."""
    model = vit_base_mae_pretraining()

    model = model.to(device)
    model = model.eval()

    return model
