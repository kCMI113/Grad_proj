from torch.optim import Adam, lr_scheduler


class MultiOptimizer():
    def __init__(self, model, lr, weight_decay):

        self.lr1 = lr
        self.lr2 = lr/(2**6)
        
        self.encoder_params = model.Encoder.parameters()
        self.decoder_params = model.Decoder.parameters()

        self.encoder_optimizer = Adam(params=self.encoder_params, lr=self.lr1, weight_decay=weight_decay)
        self.decoder_optimizer = Adam(params=self.decoder_params, lr=self.lr2, weight_decay=weight_decay)

    def zero_grad(self):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

    def step(self):
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

class MultiScheduler():
    def __init__(self, enc_optim, dec_optim, milestones=[25,50], gamma1 = 0.5, gamma2=4):
        
        self.enc_scheduler = lr_scheduler.MultiStepLR(optimizer=enc_optim, milestones=milestones, gamma=gamma1)
        self.dec_scheduler = lr_scheduler.MultiStepLR(optimizer=dec_optim, milestones=milestones, gamma=gamma2)

    def step(self):
        self.enc_scheduler.step()
        self.dec_scheduler.step()
