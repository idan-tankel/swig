# import os, sys
# sys.path.append(os.path.abspath("."))
from JSL.verb.imsituDatasetGood import imSituDatasetGood
from JSL.verb.verbModel import ImsituVerb
import datetime
import h5py
import argparse
import json
import torch
from torchvision import datasets, models, transforms
from JSL.gsr import model as jsl_model
from JSL.gsr.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader


def get_mapping(word_file):
    dict = {}
    word_list = []
    with open(word_file) as f:
        k = 0
        for line in f:
            word = line.split('\n')[0]
            dict[word] = k
            word_list.append(word)
            k += 1
    return dict, word_list



parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default=None)
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.001)

parser.add_argument("--detach_epoch", type=int, default=12)
parser.add_argument("--gt_noun_epoch", type=int, default=5)
parser.add_argument("--hidden-size", type=int, default=1024)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--verb-path", type=str, default=None)
parser.add_argument("--jsl-path", type=str, default=None)
parser.add_argument("--image-file", type=str, default='SWiG_jsons/test.json')
parser.add_argument("--store-features", action="store_true", default=False)

args = parser.parse_args()

# if args.verb_path == None:
#     print('please input a path to the verb model weights')
#     return
# if args.jsl_path == None:
#     print('please input a path to the jsl model weights')
#     return
# if args.image_file == None:
#     print('please input a path to the image file')
#     return

# if args.store_features:
#     if not os.path.exists('local_features'):
#         os.makedirs('local_features')

kwargs = {"num_workers": args.workers} if torch.cuda.is_available() else {}
verbs = './global_utils/verb_indices.txt'
verb_to_idx, idx_to_verb = get_mapping(verbs)

print("initializing verb model")

test_dataset = imSituDatasetGood(verb_to_idx, json_file=args.image_file, inference=False, is_train=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, **kwargs)
iterator = iter(test_dataloader)
print("initializing jsl model")
example = next(iterator)
print(example)


