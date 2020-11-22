# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import os.path
from os import path
import collections
import json
import importlib
import random

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator

# Import any attacks here
from advertorch.attacks import *
# I have to do the following to instantiate a class from a string... it's a bit clunky.
advertorch_module = importlib.import_module('advertorch.attacks')

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VQADataset(splits)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VQA:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=1024,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple, adversarial=False, adv_batch_prob=0.0, attack_name=None, attack_params={}):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        use_adv_batch = False

        best_valid = 0.
        
        for epoch in range(args.epochs):
            quesid2ans = {}
            # Count the number of batches that were adversarially perturbed
            n_adv_batches = 0
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()

                # If doing adversarial training, perturb input features
                # with probability adv_batch_prob
                if adversarial:
                    rand = random.uniform(0,1)
                    use_adv_batch = rand <= adv_batch_prob
                if use_adv_batch:
                    # Create adversary from given class name and parameters
                    n_adv_batches += 1
                    AdversaryClass_ = getattr(advertorch_module, attack_name)
                    adversary = AdversaryClass_(
                        lambda x: self.model(x, boxes, sent),
                        loss_fn=self.bce_loss,
                        **attack_params
                        )
                    # Perturb feats using adversary
                    feats = adversary.perturb(feats, target)
               

                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.) + \
                        "Epoch %d: Num adversarial batches %d / %d\n" % (epoch, n_adv_batches, i+1)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def adversarial_predict(self, eval_tuple: DataTuple, dump=None, attack_name='GradientAttack', attack_params={}):
        """
        Predict the answers to questions in a data split, but
        using a specified adversarial attack on the inputs.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        sim_trace = [] # Track avg cos similarity across batches
        for i, datum_tuple in enumerate(tqdm(loader)):
            ques_id, feats, boxes, sent, target = datum_tuple
            feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()

            # Create adversary from given class name and parameters
            AdversaryClass_ = getattr(advertorch_module, attack_name)
            adversary = AdversaryClass_(
                lambda x: self.model(x, boxes, sent),
                loss_fn=self.bce_loss,
                **attack_params
            )

            # Perturb feats using adversary
            feats_adv = adversary.perturb(feats, target)

            # Compute average cosine similarity between true
            # and perturbed features
            sim_trace.append(self.avg_cosine_sim(feats, feats_adv))

            # Compute prediction on adversarial examples
            with torch.no_grad():
                feats_adv = feats_adv.cuda()
                logit = self.model(feats_adv, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        print(f"Average cosine similarity across batches: {torch.mean(torch.Tensor(sim_trace))}")
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    def adversarial_evaluate(self, eval_tuple: DataTuple, dump=None, attack_name='GradientAttack', attack_params={}):
        """Evaluate model on adversarial inputs"""
        quesid2ans = self.adversarial_predict(eval_tuple, dump, attack_name, attack_params)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    def avg_cosine_sim(self, feats: torch.Tensor, feats_adv: torch.Tensor):
        """Computes the average cosine similarity between true and adversarial examples"""
        return nn.functional.cosine_similarity(feats, feats_adv, dim=-1).mean()

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    vqa = VQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Load adversarial json (if it exists)
    adversarial_dict = None
    adversarial_dict_path = 'attacks.json'
    if path.exists(adversarial_dict_path):
        with open(adversarial_dict_path) as f:
            print('adversarial_dict found')
            adversarial_dict = json.load(f)
    print('Loaded adversarial dict:', adversarial_dict)
            
    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            if adversarial_dict is not None:
                for attack in adversarial_dict['attacksToUse']:
                    vqa.adversarial_predict(
                        get_data_tuple(args.test, bs=950,
                                       shuffle=False, drop_last=False),
                        dump=os.path.join(args.output, 'test_predict.json'),
                        attack_name=attack, attack_params=adversarial_dict['attacks'][attack]
                    )
            else:
                vqa.predict(
                    get_data_tuple(args.test, bs=950,
                                   shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, 'test_predict.json')
                )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            if adversarial_dict is not None:
                for attack in adversarial_dict['attacksToUse']:
                    result = vqa.adversarial_evaluate(
                        get_data_tuple('minival', bs=100,
                                       shuffle=False, drop_last=False),
                        dump=os.path.join(args.output, 'minival_predict.json'),
                        attack_name=attack, attack_params=adversarial_dict['attacks'][attack]
                    )
                    print(attack, ': ', result)
            else:
                result = vqa.evaluate(
                    get_data_tuple('minival', bs=100,
                                   shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, 'minival_predict.json')
                )
                print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', vqa.train_tuple.dataset.splits)
        if vqa.valid_tuple is not None:
            print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        print(f"attack: {args.attackName}")
        attack_params = adversarial_dict['attacks'][args.attackName] if args.attackName else {}
        print(f"attack params: {attack_params}")
        vqa.train(vqa.train_tuple, vqa.valid_tuple, \
                  adversarial=args.trainAdversarial, \
                  adv_batch_prob=args.adv_batch_prob, \
                  attack_name=args.attackName, \
                  attack_params=attack_params)
