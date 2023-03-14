import torch.nn as nn

from tactile_dexterity.utils import vicreg_loss

# Taken from dexVR github repo
class VICReg(nn.Module):
    def __init__(
        self,
        backbone,
        projector,
        augment_fn,
        sim_coef,
        std_coef,
        cov_coef
    ):
        super(VICReg, self).__init__()

        # Networks
        self.backbone = backbone
        self.projector = projector
        self.augment_fn = augment_fn

        # Loss parameters
        self.sim_coef = sim_coef
        self.std_coef = std_coef
        self.cov_coef = cov_coef

    def get_image_representation(self, image):
        augmented_image = self.augment_fn(image)
        representation = self.projector(self.backbone(augmented_image))
        return representation

    def forward(self, image):
        first_projection = self.get_image_representation(image)
        second_projection = self.get_image_representation(image)

        loss, loss_info = vicreg_loss(
            input_rep = first_projection,
            output_rep = second_projection,
            feature_size = first_projection.shape[-1],
            sim_coef = self.sim_coef,
            std_coef = self.std_coef,
            cov_coef = self.cov_coef
        )

        return loss, loss_info

    def get_encoder_weights(self):
        return self.backbone.state_dict()

    def to(self, device):
        self.projector.to(device)
        self.backbone.to(device)

    def train(self):
        self.projector.train()
        self.backbone.train()

    def eval(self):
        self.projector.eval() 
        self.backbone.eval()

    def parameters(self):
        return list(list(self.projector.parameters()) + list(self.backbone.parameters()))