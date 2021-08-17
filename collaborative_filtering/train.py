import torch
import torch.nn as nn
import numpy as np
import os
import time
import datetime
import enlighten
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from .parameters import parse_args
from .early_stop import EarlyStopping
from .utils import time_since, create_logger, dict2table, deep_apply, get_dataset_name
from .evaluate import evaluate
from .model import MF, MLP
from .loss import BPRLoss, GBPRLoss
from .dataset import TrainingDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = parse_args()


def train():
    if args.model_name == 'MF':
        model = MF(args)
    elif args.model_name == 'MLP':
        model = MLP(args)
    else:
        raise NotImplementedError

    model = model.to(device)
    logger.info(model)

    model.eval()
    metrics, _ = evaluate(model, 'valid', logger)
    model.train()
    logger.info(f'Initial metrics on validation set {deep_apply(metrics)}')
    best_checkpoint = copy.deepcopy(model.state_dict())
    best_val_metrics = copy.deepcopy(metrics)

    if args.loss_type == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_type == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == 'BPR':
        criterion = BPRLoss()
    elif args.loss_type == 'GBPR':
        criterion = GBPRLoss(args.group_coefficient_rho)
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_full = []
    early_stopping = EarlyStopping(args.early_stop_patience)
    start_time = time.time()
    writer = SummaryWriter(log_dir=os.path.join(
        args.tensorboard_runs_path,
        f'{args.model_name}-{args.loss_type}-{get_dataset_name(args.dataset_path)}',
        f"{str(datetime.datetime.now().replace(microsecond=0)).replace(' ', '_').replace(':', '-')}{'-remark-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}",
    ))

    enlighten_manager = enlighten.get_manager()
    batch = 0

    try:
        with enlighten_manager.counter(total=args.num_epochs,
                                       desc='Training epochs',
                                       unit='epochs') as epoch_pbar:
            for epoch in epoch_pbar(range(1, args.num_epochs + 1)):
                dataset = TrainingDataset(args)
                dataloader = DataLoader(dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        pin_memory=True)
                with enlighten_manager.counter(total=len(dataloader),
                                               desc='Training batches',
                                               unit='batches',
                                               leave=False) as batch_pbar:
                    # batch_size
                    # batch_size
                    # batch_size, negative_sampling_ratio
                    # batch_size, group_size
                    for user_indexs, positive_item_indexs, negative_item_indexs, group_user_indexs in batch_pbar(
                            dataloader):
                        batch += 1
                        batch_size = user_indexs.size(0)
                        # 0 1 2 ...
                        # batch_size
                        positive_score = model(user_indexs,
                                               positive_item_indexs)
                        # 0 0 0 0 1 1 1 1 2 2 2 2 ...
                        # batch_size * negative_sampling_ratio
                        negative_score = model(
                            user_indexs.repeat_interleave(
                                args.negative_sampling_ratio),
                            negative_item_indexs.flatten())

                        if args.loss_type == 'BCE':
                            y_pred = torch.cat(
                                (positive_score, negative_score))
                            y_true = torch.cat(
                                (torch.ones(positive_score.size(0)),
                                 torch.zeros(
                                     negative_score.size(0)))).to(device)
                            loss = criterion(y_pred, y_true)
                        elif args.loss_type == 'CE':
                            # batch_size, 1
                            positive_score = positive_score.unsqueeze(dim=-1)
                            # batch_size, negative_sampling_ratio
                            negative_score = negative_score.view(
                                batch_size, args.negative_sampling_ratio)
                            # batch_size, 1 + negative_sampling_ratio
                            y_pred = torch.cat(
                                (positive_score, negative_score), dim=1)
                            # batch_size
                            y_true = torch.zeros(batch_size).long().to(device)
                            loss = criterion(y_pred, y_true)
                        elif args.loss_type == 'BPR':
                            loss = criterion(positive_score, negative_score)
                        elif args.loss_type == 'GBPR':
                            # batch_size, group_size
                            group_positive_score = model(
                                group_user_indexs.flatten(),
                                positive_item_indexs.repeat_interleave(
                                    args.group_size),
                            ).view(batch_size, args.group_size)
                            loss = criterion(positive_score, negative_score,
                                             group_positive_score)
                        else:
                            raise NotImplementedError

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loss_full.append(loss.item())
                        writer.add_scalar('Train/Loss', loss.item(), batch)
                        if batch % args.num_batches_show_loss == 0:
                            logger.info(
                                f"Time {time_since(start_time)}, epoch {epoch}, batch {batch}, current loss {loss.item():.4f}, average loss {np.mean(loss_full):.4f}, latest average loss {np.mean(loss_full[-10:]):.4f}"
                            )
                if epoch % args.num_epochs_validate == 0:
                    model.eval()
                    metrics, overall = evaluate(model, 'valid', logger)
                    model.train()

                    for metric, value in metrics.items():
                        writer.add_scalar(f'Validation/{metric}', value, epoch)
                    logger.info(
                        f"Time {time_since(start_time)}, epoch {epoch}, metrics {deep_apply(metrics)}"
                    )

                    early_stop, get_better = early_stopping(-overall)
                    if early_stop:
                        logger.info('Early stopped')
                        break
                    elif get_better:
                        best_checkpoint = copy.deepcopy(model.state_dict())
                        best_val_metrics = copy.deepcopy(metrics)

    except KeyboardInterrupt:
        logger.info('Stop in advance')

    logger.info(
        f'Best metrics on validation set\n{dict2table(best_val_metrics)}')

    model.load_state_dict(best_checkpoint)
    model.eval()
    metrics, _ = evaluate(model, 'test', logger)
    logger.info(f'Metrics on test set\n{dict2table(metrics)}')


if __name__ == '__main__':
    logger = create_logger()
    logger.info(args)
    logger.info(f'Using device: {device}')
    logger.info(f'Training with dataset {get_dataset_name(args.dataset_path)}')
    train()
