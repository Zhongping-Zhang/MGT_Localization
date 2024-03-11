import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import os
import json
from sklearn.metrics import average_precision_score
from dataloaders import Sample_Sentence_from_Article
from AdaLoc.roberta_adaloc import RobertaSentenceHead

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        article_id, label, input_sentences = sample['article_id'],sample['label_np'],sample['input_sentences']
        sentence_feature = model.extract_roberta_feature(input_sentences) # (batch_size, 512, 1024)
        label = label.type(LongTensor)

        optimizer.zero_grad()
        output = model(sentence_feature) # (batch_size, sentences_in_window)

        loss = F.binary_cross_entropy_with_logits(output, label.float())
        loss.backward()
        optimizer.step()


        if batch_idx % args.process_interval == 0:
            labels_np = label.cpu().numpy()
            predictions_np = output.data.cpu().numpy()
            mAP = average_precision_score(y_true=labels_np, y_score=sigmoid(predictions_np))

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAP: {:.6f}'.format(
                epoch, batch_idx * len(label), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), mAP))

def val():
    global val_loader

    return_predictions = []
    return_labels = []
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, sample in enumerate(val_loader):
            article_id, label, input_sentences = sample['article_id'],sample['label_np'],sample['input_sentences']
            sentence_feature = model.extract_roberta_feature(input_sentences)
            label = label.type(LongTensor)  # convert to gpu computation

            output = model(sentence_feature)

            val_loss += BCE_criterion(output, label.float())  # outputs – (N,C); target – (N)
            predicted = output.data
            predicted_np = predicted.cpu().numpy()
            label_np = label.cpu().numpy()

            return_predictions.append(predicted_np)
            return_labels.append(label_np)

        predictions_np, labels_np = np.concatenate(return_predictions), np.concatenate(return_labels)

        mAP = average_precision_score(labels_np, sigmoid(predictions_np))
        val_loss /= len(val_loader)

        print('\nValidation set: Average loss: {:.4f}, mAP: {:.4f} \n'
              .format(val_loss, mAP))
    return labels_np, output.data, val_loss, mAP

def main(args):
    global BCE_criterion, train_loader, val_loader, model, optimizer
    BCE_criterion = nn.BCEWithLogitsLoss()

    log_name = os.path.join("logs", args.model_name)
    os.makedirs(log_name, exist_ok=True)

    """ Step1: Configure dataset & model & optimizer """
    traindata = Sample_Sentence_from_Article(
                                        file_path=args.train_file,
                                        sentences_in_window=args.sentences_in_window,
                                        n_sample=args.n_train_sample,
                                        )
    valdata = Sample_Sentence_from_Article(
                                        file_path=args.test_file,
                                        sentences_in_window=args.sentences_in_window,
                                        n_sample=args.n_test_sample,
                                        )

    train_loader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
    val_loader = torch.utils.data.DataLoader(valdata, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)


    model = RobertaSentenceHead(hidden_size=1024,
                                num_labels=args.sentences_in_window,
                                roberta_detector_name=args.roberta_detector_name,
                                ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=0, verbose=True)

    best_macro_mAP = 0

    f_logger = open(log_name + "/logger_info.txt", 'w')
    for epoch_org in range(args.num_epoch):
        epoch = epoch_org + 1
        train(epoch)
        _, _, val_loss, macro_mAP = val()
        scheduler.step(macro_mAP)

        f_logger.write("epoch-{}: val: {:.4f}; mAP: {:.4f} \n".format(epoch, val_loss, macro_mAP))
        if macro_mAP > best_macro_mAP:
            best_macro_mAP = macro_mAP
            torch.save(model, log_name + "/epoch-best.pkl")
            best_epoch = epoch
        if epoch % args.save_interval == 0:
            print('saving the %d epoch' % (epoch))
            torch.save(model, log_name + "/epoch-%d.pkl" % (epoch))

    f_logger.write("best epoch num: %d" % best_epoch)
    f_logger.close()

    results = vars(args)
    results.update({'best_epoch_mAP': best_macro_mAP, 'best_epoch': best_epoch})

    with open(os.path.join(log_name, "train_info.json"), 'w') as f:
        json.dump(results, f, indent=2)


if __name__=="__main__":
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='adaloc_goodnews', help='model name')
    parser.add_argument('--train_file', type=str, default='data/Text_Localization/Goodnews/goodnews_train-gpt2-xl-art10000-seg1.json',
                        help='path to training data file')
    parser.add_argument('--test_file', type=str, default="data/Text_Localization/Goodnews/goodnews_val-gpt2-xl-art1000-seg3.json",
                        help='path to test data file')
    parser.add_argument('--n_train_sample', type=int, default=10000, help="number of training samples")
    parser.add_argument('--n_test_sample', type=int, default=1000, help="number of test samples")
    parser.add_argument('--sentences_in_window', type=int, default=3, help="number of sentences with in the receptive field")
    parser.add_argument('--roberta_detector_name', type=str, default="roberta-large-openai-detector", help="sentence feature encoder")
    parser.add_argument('--num_epoch', type=int, default=10, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=16, help='size of the batches') # batch_size: 256
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')


    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--save_interval', type=int, default=1, help='the interval between saved epochs')
    parser.add_argument('--process_interval', type=int, default=2, help='the interval between process print')

    args = parser.parse_args()
    print(args)

    main(args)










