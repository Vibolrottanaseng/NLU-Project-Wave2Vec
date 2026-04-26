import os
from abc import abstractmethod
import time
import torch
import pandas as pd
from numpy import inf
import json


# ======================================================
# BASE TRAINER
# ======================================================
class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # device setup
        self.device, device_ids = self._prepare_device(args.n_gpu)

        self.model = model.to(self.device)

        # FIXED: wrap correct model
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric

        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        # FIXED: clean init (no inf inside dict)
        self.best_recorder = {
            'val': {},
            'test': {}
        }

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    # ======================================================
    # TRAIN LOOP
    # ======================================================
    def train(self):
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {'epoch': epoch}
            log.update(result)

            self._record_best(log)

            for key, value in log.items():
                print(f'\t{key:15s}: {value}')

            best = False

            if self.mnt_mode != 'off':
                try:
                    improved = (
                        (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or
                        (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    )
                except KeyError:
                    print(f"Warning: Metric '{self.mnt_metric}' not found.")
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Early stopping triggered.")
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

        self._print_best()
        self._print_best_to_file()

    # ======================================================
    # BEST RECORDING
    # ======================================================
    def _record_best(self, log):
        if self.mnt_metric in log:
            improved_val = (
                (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or
                (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
            )
            if improved_val:
                self.best_recorder['val'].update(log)

        if self.mnt_metric_test in log:
            improved_test = (
                (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.mnt_best) or
                (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.mnt_best)
            )
            if improved_test:
                self.best_recorder['test'].update(log)

    # ======================================================
    # SAVE CHECKPOINT
    # ======================================================
    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }

        path = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, path)

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)

    # ======================================================
    # LOAD CHECKPOINT
    # ======================================================
    def load_checkpoint(self, best=False):
        path = os.path.join(
            self.checkpoint_dir,
            "model_best.pth" if best else "current_checkpoint.pth"
        )

        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

    # ======================================================
    # DEVICE
    # ======================================================
    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()

        if n_gpu_use > 0 and n_gpu == 0:
            n_gpu_use = 0

        if n_gpu_use > n_gpu:
            n_gpu_use = n_gpu

        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        return device, list(range(n_gpu_use))

    # ======================================================
    # PRINT BEST
    # ======================================================
    def _print_best(self):
        print(f"\nBest validation results ({self.args.monitor_metric}):")
        for k, v in self.best_recorder['val'].items():
            print(f'\t{k:15s}: {v}')

        print(f"\nBest test results ({self.args.monitor_metric}):")
        for k, v in self.best_recorder['test'].items():
            print(f'\t{k:15s}: {v}')

    # ======================================================
    # SAVE BEST TO FILE
    # ======================================================
    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))

        self.best_recorder['val'].update({
            'time': crt_time,
            'seed': self.args.seed,
            'best_model_from': 'val'
        })

        self.best_recorder['test'].update({
            'time': crt_time,
            'seed': self.args.seed,
            'best_model_from': 'test'
        })

        os.makedirs(self.args.record_dir, exist_ok=True)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.csv')

        if os.path.exists(record_path):
            record_table = pd.read_csv(record_path)
        else:
            record_table = pd.DataFrame()

        record_table = pd.concat([
            record_table,
            pd.DataFrame([self.best_recorder['val']]),
            pd.DataFrame([self.best_recorder['test']])
        ], ignore_index=True)

        record_table.to_csv(record_path, index=False)


# ======================================================
# TRAINER
# ======================================================
class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args,
                 lr_scheduler, train_dataloader, val_dataloader, test_dataloader):

        super().__init__(model, criterion, metric_ftns, optimizer, args)

        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    # ======================================================
    # EPOCH
    # ======================================================
    def _train_epoch(self, epoch):

        # ================= TRAIN =================
        self.model.train()
        train_loss = 0

        for batch in self.train_dataloader:
            images_id, images, problem_vec, reports_ids, reports_masks, seq_length = batch

            images = images.to(self.device)
            problem_vec = problem_vec.to(self.device)
            reports_ids = reports_ids.to(self.device)
            reports_masks = reports_masks.to(self.device)

            output = self.model(images, problem_vec, reports_ids, mode='train')

            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        # ================= VALIDATION =================
        self.model.eval()
        with torch.no_grad():
            val_res, val_gts = [], []

            for batch in self.val_dataloader:
                images_id, images, problem_vec, reports_ids, reports_masks, seq_length = batch

                images = images.to(self.device)
                problem_vec = problem_vec.to(self.device)

                output = self.model(images, problem_vec, mode='sample')

                val_res.extend(self.model.tokenizer.decode_batch(output.cpu().numpy()))
                val_gts.extend(self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy()))

            val_met = self.metric_ftns(
                {i: [gt] for i, gt in enumerate(val_gts)},
                {i: [re] for i, re in enumerate(val_res)}
            )

            log.update({f'val_{k}': v for k, v in val_met.items()})

        # ================= TEST =================
        test_results = []
        with torch.no_grad():
            test_res, test_gts = [], []

            for batch in self.test_dataloader:
                images_id, images, problem_vec, reports_ids, reports_masks, seq_length = batch

                images = images.to(self.device)
                problem_vec = problem_vec.to(self.device)

                output = self.model(images, problem_vec, mode='sample')

                gen = self.model.tokenizer.decode_batch(output.cpu().numpy())
                gt = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                test_res.extend(gen)
                test_gts.extend(gt)

                for i in range(len(images_id)):
                    test_results.append({
                        "id": str(images_id[i]),
                        "ground_truth": gt[i],
                        "generated": gen[i]
                    })

            test_met = self.metric_ftns(
                {i: [gt] for i, gt in enumerate(test_gts)},
                {i: [re] for i, re in enumerate(test_res)}
            )

            log.update({f'test_{k}': v for k, v in test_met.items()})

        # ================= SAVE JSON =================
        save_path = os.path.join(self.args.save_dir, "test_predictions.json")

        with open(save_path, "w") as f:
            json.dump(test_results, f, indent=4, ensure_ascii=False)

        print(f"\nSaved test predictions to: {save_path}")

        self.lr_scheduler.step()

        return log