from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.autograd import Variable

from imdb_attn import EncoderRNN, AttnClassifier, Attn
import dill

import argparse


def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)

def mk_html(sentence, attns):
    html = ""
    for word, attn in zip(sentence, attns):
        html += ' ' + highlight(
            TEXT.vocab.itos[word],
            attn
        )
    return html + "<br><br>\n"

parser = argparse.ArgumentParser(description='View Attention')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(0)
if args.cuda:
    torch.cuda.manual_seed(0)


print("pkl load !!!")
TEXT = dill.load(open("TEXT.pkl",'rb'))
LABEL = dill.load(open("LABEL.pkl",'rb'))

print("split !!!")
train, test = datasets.IMDB.splits(TEXT, LABEL)
device = 0 if args.cuda else -1
train_iter, test_iter = data.BucketIterator.splits(
    (train, test), batch_size=1, device=device,
    repeat=False)

print("pkl load !!!")
encoder = dill.load(open("encoder.pkl","rb"))
classifier = dill.load(open("classifier.pkl","rb"))
if args.cuda:
    encoder.cuda()
    classifier.cuda()

print("write !!!")
f = open("attn.html", "w")
for batch in test_iter:
    x = batch.text[0]
    y = batch.label - 1
    encoder_outputs = encoder(x)
    output, attn = classifier(encoder_outputs)
    pred = output.data.max(1, keepdim=True)[1]
    a = attn.data[0,:,0]
    f.write( '\t'.join(("<br> answer:" , str(int(y[0].data)), ", predicted:",str(int(pred[0])), "<br> sentence with attention: ", mk_html(x.data[0], a))) )
f.close()

