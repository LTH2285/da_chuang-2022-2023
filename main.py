import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="mdfend")
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--max_len", type=int, default=170)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--early_stop", type=int, default=3)
parser.add_argument(
    "--bert_vocab_file",
    default="./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt",
)
parser.add_argument("--root_path", default="./data/weibo21/")
parser.add_argument(
    "--bert", default="./pretrained_model/chinese_roberta_wwm_base_ext_pytorch"
)
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--gpu", default="0")
parser.add_argument("--bert_emb_dim", type=int, default=768)
parser.add_argument("--w2v_emb_dim", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--emb_type", default="bert")
parser.add_argument(
    "--w2v_vocab_file",
    default="./pretrained_model/w2v/Tencent_AILab_Chinese_w2v_model.kv",
)
parser.add_argument("--save_param_dir", default="./param_model")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from run import Run
import torch
import numpy as np
import random

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if args.emb_type == "bert":
    emb_dim = args.bert_emb_dim
    vocab_file = args.bert_vocab_file
elif args.emb_type == "w2v":
    emb_dim = args.w2v_emb_dim
    vocab_file = args.w2v_vocab_file

print(
    f"lr: {args.lr}; model name: {args.model_name}; emb_type: {args.emb_type}; batchsize: {args.batchsize}; epoch: {args.epoch}; gpu: {args.gpu}; emb_dim: {emb_dim}"
)


config = {
    "use_cuda": torch.cuda.is_available(),
    "batchsize": args.batchsize,
    "max_len": args.max_len,
    "early_stop": args.early_stop,
    "num_workers": args.num_workers,
    "vocab_file": vocab_file,
    "emb_type": args.emb_type,
    "bert": args.bert,
    "root_path": args.root_path,
    "weight_decay": 5e-5,
    "model": {"mlp": {"dims": [384], "dropout": 0.2}},
    "emb_dim": emb_dim,
    "lr": args.lr,
    "epoch": args.epoch,
    "model_name": args.model_name,
    "seed": args.seed,
    "save_param_dir": args.save_param_dir,
}


if __name__ == "__main__":
    Run(config=config).main()
