
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import argparse
import torch.nn.parallel.data_parallel as data_parallel
import numpy as np
import scipy
import scipy.io as sio
import torch.optim as optim
import time
import multiprocessing
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(
    os.path.dirname(sys.path[0]) + '/tools/speech_processing_toolbox')

from model.sruc import SRUC as Model
from tools.misc import get_learning_rate, save_checkpoint, reload_for_eval, reload_model
from tools.dataset import make_loader, Processer, DataReader

import voicetool.base as voicebox
import voicetool.multiworkers as worker
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


def train(model, args, device, writer):
    print('preparing data...')
    dataloader, dataset = make_loader(
        args.tr_list,
        args.batch_size,
        num_workers=args.num_threads,
        processer=Processer(
            win_len=args.win_len,
            win_inc=args.win_inc,
            left_context=args.left_context,
            right_context=args.right_context,
            fft_len=args.fft_len,
            target_mode=args.target_mode,
            window_type=args.win_type,
            ))
    print_freq = 100
    num_batch = len(dataloader)
    params = model.get_params(args.weight_decay)
    optimizer = optim.Adam(params, lr=args.learn_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=1, verbose=True)
    
    if args.retrain:
        start_epoch, step = reload_model(model, optimizer, args.exp_dir,
                                         args.use_cuda)
    else:
        start_epoch, step = 0, 0
    print('---------PRERUN-----------')
    lr = get_learning_rate(optimizer)
    print('(Initialization)')
    val_loss = validation(model, args, lr, -1, device)
    writer.add_scalar('Loss/Train', val_loss, step)
    writer.add_scalar('Loss/Cross-Validation', val_loss, step)

    for epoch in range(start_epoch, args.max_epoch):
        torch.manual_seed(args.seed + epoch)
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed + epoch)
        model.train()
        loss_total = 0.0
        loss_print = 0.0
        stime = time.time()
        lr = get_learning_rate(optimizer)
        for idx, data in enumerate(dataloader):
            inputs, labels, lengths = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths
            
            model.zero_grad()
            outputs, _ = data_parallel(model, (inputs, lengths))
            loss = F.mse_loss(outputs, labels, reduction='sum')/lengths.sum()
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            step += 1
            loss_total += loss.data.cpu()
            loss_print += loss.data.cpu()
            
            del lengths, outputs, labels, inputs, loss, _
            if (idx+1) % 3000 == 0:
                save_checkpoint(model, optimizer, epoch + 1, step, args.exp_dir)
            if (idx + 1) % print_freq == 0:
                eplashed = time.time() - stime
                speed_avg = eplashed / (idx+1)
                loss_print_avg = loss_print / print_freq
                print('Epoch {:3d}/{:3d} | batches {:5d}/{:5d} | lr {:1.4e} |'
                      '{:2.3f}s/batches | loss {:2.6f}'.format(
                          epoch, args.max_epoch, idx + 1, num_batch, lr,
                          speed_avg, loss_print_avg))
                sys.stdout.flush()
                writer.add_scalar('Loss/Train', loss_print_avg, step)
                loss_print = 0.0
        eplashed = time.time() - stime
        loss_total_avg = loss_total / num_batch
        print(
            'Training AVG.LOSS |'
            ' Epoch {:3d}/{:3d} | lr {:1.4e} |'
            ' {:2.3f}s/batch | time {:3.2f}mins |'
            ' loss {:2.6f}'.format(
                                    epoch + 1,
                                    args.max_epoch,
                                    lr,
                                    eplashed/num_batch,
                                    eplashed/60.0,
                                    loss_total_avg.item()))
        val_loss = validation(model, args, lr, epoch, device)
        writer.add_scalar('Loss/Cross-Validation', val_loss, step)
        writer.add_scalar('learn_rate', lr, step) 
        if val_loss > scheduler.best:
            print('Rejected !!! The best is {:2.6f}'.format(scheduler.best))
        else:
            save_checkpoint(model, optimizer, epoch + 1, step, args.exp_dir)
        scheduler.step(val_loss)
        sys.stdout.flush()
        stime = time.time()


def validation(model, args, lr, epoch, device):
    dataloader, dataset = make_loader(
        args.cv_list,
        args.batch_size,
        num_workers=args.num_threads,
        processer=Processer(
            win_len=args.win_len,
            win_inc=args.win_inc,
            left_context=args.left_context,
            right_context=args.right_context,
            fft_len=args.fft_len,
            target_mode=args.target_mode,
            window_type=args.win_type))
    model.eval()
    loss_total = 0.0
    num_batch = len(dataloader)
    stime = time.time()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            inputs, labels, lengths = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            outputs, _ = data_parallel(model, (inputs, lengths))
            loss = F.mse_loss(outputs, labels, reduction='sum')/lengths.sum()
            loss_total += loss.data.cpu()
            del loss, data, inputs, labels, lengths, _,outputs
        etime = time.time()
        eplashed = (etime - stime) / num_batch
        loss_total_avg = loss_total / num_batch

    print('CROSSVAL AVG.LOSS | Epoch {:3d}/{:3d} '
          '| lr {:.6e} | {:2.3f}s/batch| time {:2.1f}mins '
          '| loss {:2.8f}'.format(epoch + 1, args.max_epoch, lr, eplashed,
                                  (etime - stime)/60.0, loss_total_avg))
    sys.stdout.flush()
    return loss_total_avg


def reconstruct(inputs, angles, save_id, nsamples, win_len, win_inc,sample_rate):

    hamming = np.hamming(win_len)
    rec_data = np.zeros(nsamples)
    spec = inputs[0]
    for step in range(spec.shape[0]):
        angle = np.cos(angles[step]) + np.sin(angles[step]) * 1.0j
        time_sample = np.fft.irfft(
            spec[step] * angle, n=512)[:win_len] * hamming
        rec_data[step * win_inc:step * win_inc + win_len] += time_sample

    rec_data = np.where(rec_data > 1., 1., np.where(rec_data < -1, -1., rec_data))
    voicebox.audiowrite(save_id, rec_data, sample_rate=sample_rate)


def decode(model, args, device):
    model.eval()
#    pool = multiprocessing.Pool(args.num_threads)
    with torch.no_grad():
        
        data_reader = DataReader(
            args.tt_list,
            win_len=args.win_len,
            win_inc=args.win_inc,
            left_context=args.left_context,
            right_context=args.right_context,
            fft_len=args.fft_len,
            window_type=args.win_type,
            sample_rate=args.sample_rate)
        if not os.path.isdir(os.path.join(args.exp_dir, 'rec_wav/')):
            os.mkdir(os.path.join(args.exp_dir, 'rec_wav/'))
        num_samples = len(data_reader)
        print('Decoding...')
        for idx in range(num_samples):
            inputs, lengths, angles, utt_id, nsamples = data_reader[idx]
            
            inputs = torch.from_numpy(inputs)
            lengths = torch.from_numpy(np.array(lengths))
            inputs = inputs.to(device)
            
            lengths = lengths.to(device)
            _, outputs = model(inputs, lengths)
            
            reconstruct(outputs.cpu().numpy(), angles,
                      os.path.join(args.exp_dir, 'rec_wav/' + utt_id),
                      nsamples,
                      args.win_len, args.win_inc,args.sample_rate)
            # pool.apply_async(
            #     reconstruct,
            #     args=(outputs, angles,
            #           os.path.join(args.exp_dir, 'rec_wav/' + utt_id),
            #           nsamples,
            #           args.win_len, args.win_inc))
        # pool.close()
        # pool.join()
        print('Decode Done!!!')


def main(args):
    device = torch.device('cuda' if args.use_cuda else 'cpu')
    args.sample_rate = {
        '8k':8000,
        '16k':16000,
        '24k':24000,
        '48k':48000,
    }[args.sample_rate]
    model = Model(
        left_context=args.left_context,
        right_context=args.right_context,
        hidden_layers=args.rnn_layers,
        hidden_units=args.rnn_units,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        kernel_size=args.kernel_size,
        kernel_num=args.kernel_num,
        target_mode=args.target_mode,
        dropout=args.dropout)
    if not args.log_dir:
        writer = SummaryWriter(os.path.join(args.exp_dir, 'tensorboard'))
    else:
        writer = SummaryWriter(args.log_dir)
    model.to(device)
    if not args.decode:
        train(model, FLAGS, device, writer)
    reload_for_eval(model, FLAGS.exp_dir, FLAGS.use_cuda)
    decode(model, args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('PyTorch Version ENhancement')
    parser.add_argument('--decode', type=int, default=0, help='if decode')
    parser.add_argument(
        '--exp-dir',
        dest='exp_dir',
        type=str,
        default='exp/cldnn',
        help='the exp dir')
    parser.add_argument(
        '--tr-list', dest='tr_list', type=str, help='the train data list')
    parser.add_argument(
        '--cv-list',
        dest='cv_list',
        type=str,
        help='the cross-validation data list')
    parser.add_argument(
        '--tt-list', dest='tt_list', type=str, help='the test data list')
    parser.add_argument(
        '--rnn-layers',
        dest='rnn_layers',
        type=int,
        default=2,
        help='the num hidden rnn layers')
    parser.add_argument(
        '--rnn-units',
        dest='rnn_units',
        type=int,
        default=512,
        help='the num hidden rnn units')
    parser.add_argument(
        '--learn-rate',
        dest='learn_rate',
        type=float,
        default=0.001,
        help='the learning rate in training')
    parser.add_argument(
        '--max-epoch',
        dest='max_epoch',
        type=int,
        default=20,
        help='the max epochs')

    parser.add_argument(
        '--dropout',
        dest='dropout',
        type=float,
        default=0.2,
        help='the probility of dropout')
    parser.add_argument(
        '--left-context',
        dest='left_context',
        type=int,
        default=1,
        help='the left context to add')
    parser.add_argument(
        '--right-context',
        dest='right_context',
        type=int,
        default=1,
        help='the right context to add')
    parser.add_argument(
        '--input-dim',
        dest='input_dim',
        type=int,
        default=257,
        help='the input dim')
    parser.add_argument(
        '--output-dim',
        dest='output_dim',
        type=int,
        default=257,
        help='the output dim')
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        help='the batch size in train')
    parser.add_argument(
        '--use-cuda', dest='use_cuda', default=1, type=int, help='use cuda')
    parser.add_argument(
        '--seed', dest='seed', type=int, default=20, help='the random seed')
    parser.add_argument(
        '--log-dir',
        dest='log_dir',
        type=str,
        default=None,
        help='the random seed')
    parser.add_argument(
        '--num-threads', dest='num_threads', type=int, default=10)
    parser.add_argument(
        '--window-len',
        dest='win_len',
        type=int,
        default=400,
        help='the window-len in enframe')
    parser.add_argument(
        '--window-inc',
        dest='win_inc',
        type=int,
        default=100,
        help='the window include in enframe')
    parser.add_argument(
        '--fft-len',
        dest='fft_len',
        type=int,
        default=512,
        help='the fft length when in extract feature')
    parser.add_argument(
        '--window-type',
        dest='win_type',
        type=str,
        default='hamming',
        help='the window type in enframe, include hamming and None')
    parser.add_argument(
        '--kernel-size',
        dest='kernel_size',
        type=int,
        default=6,
        help='the kernel_size')
    parser.add_argument(
        '--target-mode',
        dest='target_mode',
        type=str,
        default='MSA',
        help='the type of target, MSA, PSA, PSM, IBM, IRM...')
    
    parser.add_argument(
        '--kernel-num',
        dest='kernel_num',
        type=int,
        default=9,
        help='the kernel_num')
    parser.add_argument(
        '--num-gpu',
        dest='num_gpu',
        type=int,
        default=1,
        help='the num gpus to use')
    parser.add_argument(
        '--weight-decay', dest='weight_decay', type=float, default=0.00001)
    parser.add_argument(
        '--clip-grad-norm', dest='clip_grad_norm', type=float, default=5.)
    parser.add_argument(
        '--sample-rate', dest='sample_rate', type=str, default='16k')
    parser.add_argument('--retrain', dest='retrain', type=int, default=0)
    FLAGS, _ = parser.parse_known_args()
    FLAGS.use_cuda = FLAGS.use_cuda and torch.cuda.is_available()
    os.makedirs(FLAGS.exp_dir, exist_ok=True)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.backends.cudnn.benchmark = True
    if FLAGS.use_cuda:
        torch.cuda.manual_seed(FLAGS.seed)
    import pprint
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__dict__)
    print(FLAGS.win_type)
    main(FLAGS)
