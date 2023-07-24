import torch
import torch.nn as nn
from tqdm import tqdm
from time import perf_counter


class BaseTrainer:
    def __init__(
        self,
        model,
        optimer,
        device=None,
        train_loader=None,
        test_loader=None,
        scheduler=None,
        loss_func=None,
        eval_func=None,
    ):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        elif not isinstance(device, torch.device):
            raise TypeError(
                "device must be str or torch.device or none, but got {}".format(
                    type(device)
                )
            )
        self.device = device
        self.model = self.model
        self.sent2device()
        self.optimer = optimer
        self.loss_func = loss_func
        self.scheduler = scheduler
        self.test_loader = test_loader
        self.eval_func = eval_func
        self.iter_cnt = 1
        self.last_show_result = None

    def set_dataset(
        self,
        train_loader=None,
        test_loader=None,
    ):
        self.train_loader = train_loader if train_loader != None else self.train_loader
        self.test_loader = test_loader if test_loader != None else self.test_loader

    def sent2device(self):
        self.model = self.model.to(self.device)

    def train(
        self,
        epochs=10,
        save_dir=None,
        save_iter=1000,
        schedule_step_size=None,
        skip_iter=None,
        show_bar=False,
        show_hz=3,
        leave_bar=True,
    ):
        last_time = 0.0
        show_dt = 1 / show_hz
        if schedule_step_size is not None:
            assert self.scheduler is not None
        show_scheduler = schedule_step_size is not None
        for epoch in range(epochs):
            if show_bar:
                t_bar = tqdm(
                    len(self.train_loader),
                    ncols=80,
                    leave=leave_bar,
                    desc="Epoch {}".format(epoch),
                )
            for packs in self.train_loader:
                packs = self.data_preprocess(packs)
                result = self.train_step(packs)
                if show_bar:
                    if perf_counter() - last_time > show_dt:
                        self.show_result(t_bar, result, show_scheduler=show_scheduler)
                        last_time = perf_counter()
                    t_bar.update()
                if save_dir is not None and self.iter_cnt % save_iter == 0:
                    self.save_model(save_dir, save_scheduler=show_scheduler)
                    self.save_step()
                if show_scheduler and self.iter_cnt % schedule_step_size == 0:
                    self.scheduler_step()
                if skip_iter is not None and self.iter_cnt % skip_iter == 0:
                    break
                self.iter_cnt += 1
            self.epoch_step()
        self.iter_cnt = 1

    def train_step(self, packs):
        data, label = packs
        self.optimer.zero_grad()
        output = self.model(data)
        loss = self.loss_func(output, label)
        loss.backward()
        self.optimer.step()
        if self.eval_func is not None:
            acc = self.eval_func(output, label)
        return {"Loss": "{:.2e}".format(loss.item()), "Accuracy": "{:.6}".format(acc)}
    

    def show_result(self, t_bar, result, show_scheduler=False):
        if result is None:
           result = self.last_show_result
        else:
            self.last_show_result = result 
        if show_scheduler:
            result['lr'] = "{:.2e}".format(self.scheduler.get_last_lr()[0])
        if result is not None:
            t_bar.set_postfix(**result)

    def scheduler_step(self):
        self.scheduler.step()

    def epoch_step(self):
        pass
    
    def save_step(self):
        pass

    def data_preprocess(self, packs):
        data, label = packs
        if isinstance(data, (tuple, list)):
            data = [d.to(self.device) for d in data]
        else:
            data = data.to(self.device)
        if isinstance(label, (tuple, list)):
            label = [d.to(self.device) for d in label]
        else:
            label = label.to(self.device)
        return (data, label)

    def save_model(self, save_dir, save_scheduler=False):
        torch.save(self.model.state_dict(), save_dir + "/model.pth")
        torch.save(self.optimer.state_dict(), save_dir + "/optimer.pth")
        if self.scheduler is not None and save_scheduler:
            torch.save(self.scheduler.state_dict(), save_dir + "/scheduler.pth")

    def load_model(self, save_dir, load_scheduler=False):
        self.model.load_state_dict(torch.load(save_dir + "/model.pth"))
        self.optimer.load_state_dict(torch.load(save_dir + "/optimer.pth"))
        if self.scheduler is not None and load_scheduler:
            self.scheduler.load_state_dict(torch.load(save_dir + "/scheduler.pth"))

    # def validate(self, max_iter=None, show_bar=False, show_hz=3, leave_bar=False):
    #     cnt = 1
    #     last_time = 0.0
    #     show_dt = 1 / show_hz
    #     if show_bar:
    #         t_bar = tqdm(
    #             len(self.test_loader),
    #             ncols=80,
    #             leave=leave_bar,
    #         )
    #     correct, sample_num = 0, 0
    #     for data, label in self.test_loader:
    #         if isinstance(data, (tuple, list)):
    #             data = [d.to(self.device) for d in data]
    #         else:
    #             data = data.to(self.device)
    #         if isinstance(label, (tuple, list)):
    #             label = [d.to(self.device) for d in label]
    #         else:
    #             label = label.to(self.device)
    #         output = self.model(data)
    #         acc = self.acc_func(output, label)
    #         sample_num += label.shape[0]
    #         correct += acc * label.shape[0]
    #         if show_bar:
    #             if perf_counter() - last_time > show_dt:
    #                 t_bar.set_postfix(accuracy=acc)
    #                 last_time = perf_counter()
    #             t_bar.update()
    #         if max_iter != None and cnt >= max_iter:
    #             break
    #         cnt += 1
    #     tqdm.write("Accuracy: {}, Sample_num: {}".format(acc, sample_num))
