import torch
import torch.nn as nn
from .base_trainer import BaseTrainer


class WGAN_Trainer(BaseTrainer):
    # default frameword: WGAN-GP
    def __init__(
        self,
        g_net,
        d_net,
        g_optimer,
        d_optimer,
        g_scheduler=None,
        d_scheduler=None,
        clip_value=0.1,
        use_gp=True,
        lambda_gp=10,
        critic_iter=5,
        save_step_func=None,
        fp16=False,
        **kwargs
    ):
        self.clip_value = clip_value
        self.use_gp = use_gp
        self.critic_iter = critic_iter
        self.lambda_gp = lambda_gp
        self.g_loss = torch.zeros(1)
        self.d_loss = torch.zeros(1)
        self.save_step_func = save_step_func
        self.fp16=fp16
        self.g_scaler = torch.cuda.amp.GradScaler(enabled=fp16)
        self.d_scaler = torch.cuda.amp.GradScaler(enabled=fp16)

        if g_scheduler is None or d_scheduler is None:
            w_scheduler = None
        else:
            w_scheduler = {"g_scheduler": g_scheduler, "d_scheduler": d_scheduler}

        super(WGAN_Trainer, self).__init__(
            model={"g_net": g_net, "d_net": d_net},
            optimer={"g_optimer": g_optimer, "d_optimer": d_optimer},
            scheduler=w_scheduler,
            **kwargs
        )

    def sent2device(self):
        for k, v in self.model.items():
            self.model[k] = v.to(self.device)

    # W_distance ≈ E_{x~P_r}[D(x)] - E_{z~P_z}[D(G(z))]
    def train_step(self, packs):
        
        if len(packs) == 2:
            real, _ = packs
        elif len(packs) == 1:
            real = packs
        else:
            raise ValueError("The length of packs must be 1 or 2.")
        # x, y = real.min(), real.max()
        
        g_net, d_net = self.model["g_net"], self.model["d_net"]
        g_optimer, d_optimer = self.optimer["g_optimer"], self.optimer["d_optimer"]
        g_optimer.zero_grad()
        d_optimer.zero_grad()
        z = torch.randn(real.shape[0], g_net.z_dim, 1, 1, device=self.device)
        if self.iter_cnt % self.critic_iter == 0:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                loss = -d_net(g_net(z)).mean()
            self.g_scaler.scale(loss).backward()
            self.g_scaler.step(g_optimer)
            self.g_scaler.update()
            self.g_loss = loss
        else:
            if self.use_gp:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    epsilon = (
                        torch.empty(real.shape[0], 1, 1, 1).uniform_().to(self.device)
                    )
                    x_gen = g_net(z).detach()
                    x_interpolate = epsilon * real + (1 - epsilon) * x_gen
                    x_interpolate.requires_grad_(True)
                    critic = d_net(x_interpolate)
                    gradient = torch.autograd.grad(
                        critic,
                        x_interpolate,
                        grad_outputs=torch.ones_like(critic),
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    gp = (gradient.view(gradient.shape[0], -1).norm(2, dim=1) - 1) ** 2
                    gp.unsqueeze_(1)
                    loss = (d_net(x_gen) - d_net(real)) + self.lambda_gp * gp
                    loss = loss.mean()
                self.d_scaler.scale(loss).backward()
                self.d_scaler.step(d_optimer)
                self.d_scaler.update()
            else:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    loss = -(d_net(real) - d_net(g_net(z).detach())).mean()
                self.d_scaler.scale(loss).backward()
                self.d_scaler.unscale_(d_optimer)
                nn.utils.clip_grad.clip_grad_value_(
                    d_net.parameters(), self.clip_value
                ),
                self.d_scaler.step(d_optimer)
                self.d_scaler.update()
            self.d_loss = loss
        return {
            "G_loss": "{:.2e}".format(self.g_loss.item()),
            "D_loss": "{:.2e}".format(self.d_loss.item()),
        }

    def save_model(self, save_dir, save_scheduler=False):
        torch.save(
            {k: v.state_dict() for k, v in self.model.items()},
            save_dir + "/model.pth",
        )
        torch.save(
            {k: v.state_dict() for k, v in self.optimer.items()},
            save_dir + "/optimer.pth",
        )
        if self.scheduler is not None and save_scheduler:
            torch.save(
                {k: v.state_dict() for k, v in self.scheduler.items()},
                save_dir + "/scheduler.pth",
            )
            
    def save_step(self):
        if self.save_step_func is not None:
            self.save_step_func()

    def load_model(self, save_dir, load_scheduler=False):
        model = torch.load(save_dir + "/model.pth")
        for k, v in self.model.items():
            v.load_state_dict(model[k])
        optimer = torch.load(save_dir + "/optimer.pth")
        for k, v in self.optimer.items():
            v.load_state_dict(optimer[k])
        if self.scheduler is not None and load_scheduler:
            scheduler = torch.load(save_dir + "/scheduler.pth")
            for k, v in self.scheduler.items():
                v.load_state_dict(scheduler[k])
