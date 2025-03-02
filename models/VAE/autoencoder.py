import torch
import torch.nn as nn
from models.VAE.encoder import Encoder
from models.VAE.decoder import Decoder
from models.VAE.distribution import DiagonalGaussianDistribution

class AutoencoderKL(nn.Module):
    def __init__ (self,
                  ddconfig,
                  embed_dim,
                  image_key='image',
                  colorize_nlabels=None,
                  monitor=None,
                  enable_decoder_cond_lora=False,
                  ):
        super(AutoencoderKL, self).__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig) # Decoder needs to be written
        # self.loss = instantiate_from_config(lossconfig)
        assert ddconfig['double_z']
        self.quant_conv = nn.Conv2d(2*ddconfig['z_channels'], 2*embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig['z_channels'], 1) # 
        self.embed_dim = embed_dim 
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        # if enable_decoder_cond_lora:
        #     replace_layers(self.decoder.up, ResnetBlock, ResnetBlockLoRA)

    # def init_from_checkpoint

    def encode(self, x):
        h, _ = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def decode(self, z, hs=None):
        z = self.post_quant_conv(z.type(self.post_quant_conv.weight.dtype))
        return self.decoder(z, hs)
    
    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        return self.decode(z), posterior

    def get_input(self, batch, image_key):
        x = batch[image_key]
        if len(x.shape)==3:
            x = x[...,None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        if optimizer_idx==0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx,
                                            self.global_step, last_layer=self.get_last_layer(), split='train')
            return aeloss

        if optimizer_idx==1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            return discloss
        
    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        pass
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    # def log_images

    # def to_rgb

    