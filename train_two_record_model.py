import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import os
import random
import numpy as np
from models import RFND, calculate_probability
import argparse
from utils import seed_everything
from utils import Twitter_Set, Weibo_Set, PadCollate, CompleteLogger, all_metrics
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import shutil
import torch.nn.functional as F
import time
from tqdm import tqdm
# from comet_ml import Experiment


# 设置随机种子
torch.manual_seed(1023)
np.random.seed(1023)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.multiprocessing.set_sharing_strategy('file_system')
# device 3 for 1e-3

def get_parser():
    parser = argparse.ArgumentParser(description='Multimodal TextCnn')
    # dataset parameters and log parameter
    parser.add_argument('-d', '--data', metavar='DATA', default='twitter', choices=['weibo', 'twitter'],
                        help='dataset: weibo, twitter')
    parser.add_argument('--maxlength', type=int, default=150)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--tag", type=str, default="LogicDM",
                        help="the tags for comet")
    parser.add_argument('--seed', default=0
                        , type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--type', default='fix'
                        , type=str,
                        help='seed for initializing training. ')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--patience', default=5, type=int, metavar='M',
                        help='patience')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')


    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument("--log", type=str, default='LogicDM',
                        help="Where to save logs, checkpoints and debugging images.")

    # model parameter
    parser.add_argument("--graphtype", type=str, default='cross',
                        help="the type of cross modal graph")
    parser.add_argument('--outsize', default=200, type=int)
    parser.add_argument('--sizeclues', default='10#10#10#10#10', type=str)
    parser.add_argument('--rnntype', default='LSTM', type=str, choices=['LSTM', 'GRU', 'RNN'])
    parser.add_argument('--instancedim', default=200, type=int)
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--finetune', default='false', type=str)
    parser.add_argument('--relationdim', default=200, type=int)
    parser.add_argument('--ansnumber', default=2, type=int)
    parser.add_argument('--answerdim', default=200, type=int)
    parser.add_argument('--threshold', default=0.4, type=float)
    # parser.add_argument('--rate', default=0.1, type=float)s
    parser.add_argument('--rate', default=0.15, type=float)
    parser.add_argument('--normtype', default='product', type=str, choices=['minimum',  'product', 'lukasiewicz'])
    parser.add_argument('--rnn', default='true', type=str)
    parser.add_argument('--hop', default='two', type=str, choices=['one', 'two', 'three', 'twotothree', 'onetothree', 'onetotwo'])
    parser.add_argument('--loss-type', default='false', type=str)
    parser.add_argument('--guidehead', default=5, type=int)
    #  graph layer + 1
    parser.add_argument('--gcnumber', default=3, type=int)

    parser.add_argument('--max_iter', default=30, type=int,
                        help='the maximum number of iteration in each epoch ')
    parser.add_argument('--d_rop', default=0.3, type=float,
                        help='d_rop rate of neural network model')

    args = parser.parse_args()
    if args.finetune in ["True", "true"]:
        args.finetune = True
    else:
        args.finetune = False
    if args.rnn  in ['true', 'True']:
        args.rnn = True
    else:
        args.rnn = False

    args.sizeclues = [int(i) for i in str(args.sizeclues).strip().split('#')]
    return args


def main(args, experiment=None):
    logger = CompleteLogger(args.log, args.phase)
    # logger = None

    if args.data == 'twitter':
        if args.type == 'random':
            dataset = Twitter_Set(max_length=args.maxlength)
            train_size = int(len(dataset) * args.split)
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        else:
            train_dataset = Twitter_Set(max_length=args.maxlength, mode='fix', phase='train')
            test_dataset = Twitter_Set(max_length=args.maxlength, mode='fix', phase='test')

        ch = False
    elif args.data == 'weibo':
        if args.type == 'random':
            dataset = Weibo_Set(max_length=args.maxlength)
            train_size = int(len(dataset) * args.split)
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        else:
            train_dataset = Weibo_Set(max_length=args.maxlength, mode='fix', phase='train')
            test_dataset = Weibo_Set(max_length=args.maxlength, mode='fix', phase='test')
        ch = True

    print(len(train_dataset))
    print(len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, collate_fn=PadCollate(ch=ch, graph_type=args.graphtype), num_workers=args.workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, collate_fn=PadCollate(ch=ch, graph_type=args.graphtype), num_workers=args.workers,
                                     shuffle=False)

    a, b = 0, 0
    for i, (imgs, encoded_texts, mask_batch_text,
            adj_matrix, word_len, mask_T_T, mask_T_V, word_spans, labels) in tqdm(enumerate(train_loader)):
        for label in labels:
            if label == 0:
                b += 1
            else:
                a += 1
    print('Train:', a, b)
    a, b = 0, 0
    for i, (imgs, encoded_texts, mask_batch_text,
            adj_matrix, word_len, mask_T_T, mask_T_V, word_spans, labels) in tqdm(enumerate(test_loader)):
        for label in labels:
            if label == 0:
                b += 1
            else:
                a += 1
    print('test:', a, b)

    RFND_Model = RFND(input_size=768, out_size=args.outsize, rnn=args.rnn, rnn_type=args.rnntype, ch=ch, finetune=args.finetune,
                 instance_dim=args.instancedim, top_K=args.topk, size_clues=args.sizeclues, relation_dim=args.relationdim, ans_number=args.ansnumber, answer_dim=args.answerdim,
                 guide_head=args.guidehead, norm_type=args.normtype, threshold=args.threshold, rate=args.rate, gcnumber=args.gcnumber)

    RFND_Model.to(device=device)
    parameters = RFND_Model.parameters()
    optimizer = optim.Adam(params=parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=args.wd,
                           amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=args.patience, verbose=True)

    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        RFND_Model.load_state_dict(checkpoint)

    """
    acc: the number of samples that is accurately predicted / the number of samples
    precision: The percentage of examples classified as positive examples that are actually positive examples.
    recall:  the percentage of positive cases that are correct,
    F: 2*P*R/(P+R)
    """

    if args.phase == 'test':
        test_loss, metrics_test = validate(test_loader, RFND_Model, args, device)
        return test_loss, metrics_test

    # start training
    best_acc1 = 0.
    acc1_store = []
    best_metric = None
    prefix = "seed-" + str(args.seed)
    for epoch in range(args.epochs):
        # train for one epoch
        # evaluate on validation set
        train_loss, metrics_train = train_model(dataloader=train_loader, model=RFND_Model, optimizer=optimizer,
                                                epoch=epoch, args=args)
        # [acc, r_sarcasm, p_sarcasm, f1_sarcasm, r_non_sarcasm, p_non_sarcasm, f1_non_sarcasm]
        test_loss, metrics_test = validate(test_loader, RFND_Model, args=args, device=device)
        # test_loss, metrics_test = validate(train_loader, RFND_Model, args=args, device=device)
        train_metircs = {prefix + "-train_acc": metrics_train[0], prefix + "-train_loss": float(train_loss),
                          prefix + "-train_r_romor": metrics_train[1], prefix + "-train_p_rumor": metrics_train[2],
                          prefix + "-train_f1_romor": metrics_train[3], prefix + "-train_r_non_rumor": metrics_train[4],
                          prefix + "-train_p_non_romor": metrics_train[5], prefix + "-train_f1_non_rumor": metrics_train[6]
                          }
        test_metircs = {prefix + "-test_acc": metrics_test[0], prefix + "-test_loss": float(test_loss),
                          prefix + "-test_r_romor": metrics_test[1], prefix + "-test_p_rumor": metrics_test[2],
                          prefix + "-test_f1_romor": metrics_test[3], prefix + "-test_r_non_rumor": metrics_test[4],
                          prefix + "-test_p_non_romor": metrics_test[5], prefix + "-test_f1_non_rumor": metrics_test[6]
                          }
        # experiment.log_metrics(train_metircs, epoch=epoch)
        # experiment.log_metrics(test_metircs, epoch=epoch)

        # f1_store.append(float(f1_target))
        # acc1_store.append(metrics_test[0])
        lr_scheduler.step(float(metrics_test[0]))
        # remember best acc@1 and save checkpoint
        torch.save(RFND_Model.state_dict(), logger.get_checkpoint_path('latest'))
        # torch.save(RFND_Model.state_dict(), 'result')
        if metrics_test[0] > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_metric = metrics_test
        best_acc1 = max(metrics_test[0], best_acc1)
    print("best_acc1 = {:.4f}".format(best_acc1))

    final_best_metircs = {prefix + "-best_acc": best_metric[0],
                    prefix + "-best_r_romor": best_metric[1], prefix + "-best_p_rumor": best_metric[2],
                    prefix + "-best_f1_romor": best_metric[3], prefix + "-best_r_non_rumor": best_metric[4],
                    prefix + "-best_p_non_romor": best_metric[5], prefix + "-best_f1_non_rumor": best_metric[6]
                    }
    # experiment.log_metrics(final_best_metircs)
    return best_metric


def validate(dataloader, model, args, device):
    model.eval()
    # model.train()
    end = time.time()
    real_labels = []
    predicted_labels = []
    # final_test_loss = 0
    final_test_loss = 0
    for i, (imgs, encoded_texts,  mask_batch_text,
               adj_matrix, word_len, mask_T_T, mask_T_V, word_spans, labels) in tqdm(enumerate(dataloader)):
        imgs = imgs.to(device)
        encoded_texts = {k: v.to(device) for k, v in encoded_texts.items()}

        mask_batch_text = mask_batch_text.to(device)

        adj_matrix = adj_matrix.to(device)
        mask_T_T = mask_T_T.to(device)
        mask_T_V = mask_T_V.to(device)
        labels = labels.to(device)
        # with torch.no_grad():
        gc_clause_list = model(imgs=imgs, encoded_texts=encoded_texts,
                               mask_batch_T=mask_batch_text, mask_batch_TT=mask_T_T,
                               mask_batch_TV=mask_T_V,  adj_matrix=adj_matrix,
                               word_len=word_len,  word_spans=word_spans, device=device)
        # N, 2
        probability = calculate_probability(device, gc_clause_list, args.hop, args.normtype)
        # final_test_loss = final_test_loss + F.cross_entropy(probability, labels).item()
        real_labels = real_labels + labels.cpu().detach().clone().numpy().tolist()
        # the [1] is score
        predicted_labels = predicted_labels + (probability[:, 0] < probability[:, 1]).cpu().long().numpy().tolist()
        torch.cuda.empty_cache()
    period = time.time() - end
    metrics_test = all_metrics(real_labels, predicted_labels)
    # [acc, r_sarcasm, p_sarcasm, f1_sarcasm, r_non_sarcasm, p_non_sarcasm, f1_non_sarcasm]
    # print("Test: Time: {:.4f}, Acc: {:.4f}, Loss: {:.4f}, Rumor_R: {:.4f}, Rumor_P: {:.4f}, "
    #       "Rumor_F: {:.4f}, Non_Rumor_R: {:.4f}, Non_Rumor_P: {:.4f}, Non_Rumor_F1: {:.4f}".format(period,
    #                                                                                            metrics_test[0],
    #                                                                                            final_test_loss/(len(dataloader)),
    #                                                                                            metrics_test[1],
    #                                                                                            metrics_test[2],
    #                                                                                            metrics_test[3],
    #                                                                                            metrics_test[4],
    #                                                                                            metrics_test[5],
    #                                                                                            metrics_test[6]))
    print("Test: Time: {:.4f}, Acc: {:.4f}, Rumor_R: {:.4f}, Rumor_P: {:.4f}, "
          "Rumor_F: {:.4f}, Non_Rumor_R: {:.4f}, Non_Rumor_P: {:.4f}, Non_Rumor_F1: {:.4f}".format(period,
        metrics_test[0], metrics_test[1], metrics_test[2], metrics_test[3], metrics_test[4], metrics_test[5], metrics_test[6]))
    return final_test_loss, metrics_test


def train_model(dataloader, model, optimizer, epoch, args):
    model.train()
    final_train_loss = 0
    end = time.time()
    real_labels  = []
    predicted_labels = []
    for i, (imgs, encoded_texts, mask_batch_text,
               adj_matrix, word_len,  mask_T_T, mask_T_V, word_spans, labels) in tqdm(enumerate(dataloader)):
        imgs = imgs.to(device)
        encoded_texts = {k: v.to(device) for k, v in encoded_texts.items()}
        # mask_batch_text = mask_batch_text.cuda()
        mask_batch_text = mask_batch_text.to(device)
        adj_matrix = adj_matrix.to(device)
        mask_T_T = mask_T_T.to(device)
        mask_T_V = mask_T_V.to(device)
        labels = labels.to(device)
        # Gc_number_layer, 2, N;
        # print(imgs.size())
        gc_clause_list = model(imgs=imgs, encoded_texts=encoded_texts,  mask_batch_T=mask_batch_text, mask_batch_TT=mask_T_T,
                mask_batch_TV=mask_T_V, adj_matrix=adj_matrix,
                               word_len=word_len,  word_spans=word_spans, device=device)
        # N, 2
        probability = calculate_probability(device, gc_clause_list, args.hop, args.normtype)
        # train_loss = F.cross_entropy(probability, labels)
        # introduce another kind of loss
        # train_loss = F.cross_entropy(probability, labels)
        train_loss = F.cross_entropy(probability, labels) +\
                     F.cross_entropy(torch.stack([probability[:, 0], (1-probability)[:, 0]], dim=1), labels) + \
                     F.cross_entropy(torch.stack([(1 - probability)[:, 1], probability[:, 1]], dim=1), labels)
        train_loss0 = F.cross_entropy(probability, labels)
        train_loss1 = F.cross_entropy(torch.stack([probability[:, 0], (1-probability)[:, 0]], dim=1), labels)
        train_loss2 = F.cross_entropy(torch.stack([(1 - probability)[:, 1], probability[:, 1]], dim=1), labels)
        train_loss = train_loss2
        # print(train_loss, train_loss0, train_loss1, train_loss2)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        final_train_loss = final_train_loss + train_loss.item()
        real_labels = real_labels + labels.cpu().detach().clone().numpy().tolist()
        # the [1] is score
        predicted_labels = predicted_labels + (probability[:, 0] < probability[:, 1]).cpu().long().numpy().tolist()
        torch.cuda.empty_cache()
    period = time.time() - end
    metrics_train = all_metrics(real_labels, predicted_labels)
        # [acc, r_sarcasm, p_sarcasm, f1_sarcasm, r_non_sarcasm, p_non_sarcasm, f1_non_sarcasm]
    print("Train Epoch {}: Time {:.4f}, Acc: {:.4f}, Loss: {:.4f}, Rumor_R: {:.4f}, Rumor_P: {:.4f}, "
              "Rumor_F: {:.4f}, Non_Rumor_R: {:.4f}, Non_Rumor_P: {:.4f}, Non_Rumor_F1: {:.4f}".format(epoch, period,
        metrics_train[0], final_train_loss/(len(dataloader)), metrics_train[1],  metrics_train[2], metrics_train[3],
        metrics_train[4],  metrics_train[5], metrics_train[6]))

    return final_train_loss, metrics_train


if __name__ == '__main__':
    # must record every result
    args = get_parser()
    # experiment = Experiment(
    #     api_key="",
    #     project_name="",
    #     workspace="",
    # )
    # experiment.set_name("LogicDM"+str(args.data)+"final-model")
    # experiment.add_tag(args.tag)
    # experiment.log_parameters(
    #     {
    #         "lr": args.lr,
    #         "split": args.split,
    #         "epochs": args.epochs,
    #         "log": args.log,
    #         "sizeclues": args.sizeclues,
    #         "topk": args.topk,
    #         "threshold": args.threshold,
    #         "rate": args.rate,
    #         "hop": args.hop,
    #         "finetune": args.finetune,
    #         "wd": args.wd
    #
    #     }
    # )
    acc_seeds = []
    r_rumor_seeds = []
    p_rumor_seeds = []
    f1_rumor_seeds = []
    r_non_rumor_seeds = []
    p_non_rumor_seeds = []
    f1_non_rumor_seeds = []
    ori_log = args.log
    seeds_list = []
    for i in range(1):
        seeds_list.append(random.randint(0, 100000))
    for i in range(1):
        args.seed = seeds_list[i]
        seed_everything(seed=args.seed)
        args.log = "-".join([ori_log, str(args.seed)])
        # metrics_train_final = main(args, experiment=experiment)
        metrics_train_final = main(args)
        acc_seeds.append(metrics_train_final[0])
        r_rumor_seeds.append(metrics_train_final[1])
        p_rumor_seeds.append(metrics_train_final[2])
        f1_rumor_seeds.append(metrics_train_final[3])
        r_non_rumor_seeds.append(metrics_train_final[4])
        p_non_rumor_seeds.append(metrics_train_final[5])
        f1_non_rumor_seeds.append(metrics_train_final[6])
    avg_acc = np.average(acc_seeds)
    avg_r_rumor = np.average(r_rumor_seeds)
    avg_p_rumor = np.average(p_rumor_seeds)
    avg_f1_rumor = np.average(f1_rumor_seeds)
    avg_r_non_rumor = np.average(r_non_rumor_seeds)
    avg_p_non_rumor = np.average(p_non_rumor_seeds)
    avg_f1_non_rumor = np.average(f1_non_rumor_seeds)

    final_best_metircs = {"-final_acc": avg_acc,
                          "-final_r_romor": avg_r_rumor, "-final_p_rumor": avg_p_rumor,
                          "-final_f1_romor": avg_f1_rumor, "-final_r_non_rumor": avg_r_non_rumor,
                          "-final_p_non_romor": avg_p_non_rumor, "-final_f1_non_rumor": avg_f1_non_rumor
                          }
    # experiment.log_metrics(final_best_metircs)
    # experiment.end()


