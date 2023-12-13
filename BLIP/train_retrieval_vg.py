import sys
sys.path.insert(0, '/home/gamir/DER-Roei/alon/SGVL/BLIP')

'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip_retrieval_vg import blip_retrieval_vg
import utils
from utils import cosine_lr_schedule
from laion_dataset import get_data
from vg_dataset import VgDatasetText, get_vg_loader
from Winoground.evaluate_winoground import evaluate_winoground, blip_processor
from vsr.evaluate_vsr import evaluate_vsr
from VL_CheckList.example_models.blip.engine import BLIP
from VL_CheckList.vl_checklist.evaluate import Evaluate
from torchvision import transforms as transforms

from loralib import mark_only_lora_as_trainable
from torchvision.transforms.functional import convert_image_dtype
from detr.util.box_ops import box_cxcywh_to_xyxy
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm


def remove_repetitions(object_indexes, label_predictions_list):
    new_object_indexes = []
    seen_labels = []
    for idx in object_indexes:
        label = label_predictions_list[idx]
        if label in seen_labels:
            continue
        else:
            seen_labels.append(label)
            new_object_indexes.append(idx)
    return new_object_indexes

def organize_batch_classes(object_descriptions, valid_objects, vg_bbs, args, device, neg_object_descriptions = None):
    class_tokens = []
    object_samples = []
    tgt_boxes = []
    for i in range (valid_objects.shape[0]):
        valid_samples = valid_objects[i].item()
        invalid_samples = args.objects - valid_samples
        valid_for_sample = valid_samples * [True] + invalid_samples*[False]
        object_samples.append(valid_for_sample)

        if valid_samples == 0:
            tgt_boxes.append(torch.tensor([]).to(device=device, non_blocking=True))
        else:
            mask = torch.tensor(valid_for_sample).unsqueeze(1).expand(-1,4)
            boxes_for_sample = torch.masked_select(vg_bbs[i],mask).view(-1,4).to(device=device, non_blocking=True)
            tgt_boxes.append(boxes_for_sample)
    
    
    tgt_labels = []
    for i in range(len(object_descriptions)):
        labels_for_sample = []
        for j in range(len(object_descriptions[i])):
            if object_samples[i][j] == False:
                continue
            desc = object_descriptions[i][j]
            exists = False
            for k in range (len(class_tokens)):
                if desc == class_tokens[k]:
                    exists = True
                    labels_for_sample.append(k)
                    break
            if not exists:
                labels_for_sample.append(len(class_tokens))
                class_tokens.append(desc)
        tgt_labels.append(torch.tensor(labels_for_sample).type(torch.int64).to(device=device, non_blocking=True))
 
    targets = [{"labels": l, "boxes": b} for l,b in zip(tgt_labels,tgt_boxes)]

    if neg_object_descriptions != None:
        for lab in neg_object_descriptions:
            if lab != "":
                if lab not in class_tokens:
                    class_tokens.append(lab)


    return class_tokens, targets


def organize_batch_classes_relations(relation_descriptions, valid_relations, vg_bbs, args, device, neg_relation_descriptions = None):
    class_tokens = []
    relation_samples = []
    tgt_boxes = []
    for i in range (valid_relations.shape[0]):
        valid_samples = valid_relations[i].item()
        invalid_samples = args.relations - valid_samples
        valid_for_sample = valid_samples * [True] + invalid_samples*[False]
        relation_samples.append(valid_for_sample)

        if valid_samples == 0:
            tgt_boxes.append(torch.tensor([]).to(device=device, non_blocking=True))
        else:
            mask = torch.tensor(valid_for_sample).unsqueeze(1).expand(-1,4)
            boxes_for_sample = torch.masked_select(vg_bbs[i],mask).view(-1,4).to(device=device, non_blocking=True)
            tgt_boxes.append(boxes_for_sample)
    
    
    tgt_labels = []
    for i in range(len(relation_descriptions)):
        labels_for_sample = []
        for j in range(len(relation_descriptions[i])):
            if relation_samples[i][j] == False:
                continue
            desc = relation_descriptions[i][j]
            exists = False
            for k in range (len(class_tokens)):
                if desc == class_tokens[k]:
                    exists = True
                    labels_for_sample.append(k)
                    break
            if not exists:
                labels_for_sample.append(len(class_tokens))
                class_tokens.append(desc)
        tgt_labels.append(torch.tensor(labels_for_sample).type(torch.int64).to(device=device, non_blocking=True))
    targets = [{"labels": l, "boxes": b} for l,b in zip(tgt_labels,tgt_boxes)]

    if neg_relation_descriptions != None:
        for lab in neg_relation_descriptions:
            if lab != "":
                if lab not in class_tokens:
                    class_tokens.append(lab)



    return class_tokens, targets

def train(model, data_loader, optimizer, epoch, device, config, args, vg_data_loader = None):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(data_loader.num_batches, delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if args.negatives:
        metric_logger.add_meter('loss_neg', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if args.vg_loss_lambda > 0.0:
        metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_giou', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_sg', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('ce_correct', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    if vg_data_loader != None:
        vg_iter = iter(vg_data_loader)

    for i,(image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        idx = [int(id) for id in idx]
        idx = torch.IntTensor(idx)

        neg_mask = None
        objects_descs = None
        targets = None
        relations_descs = None
        relations_targets = None
        neg_object_descriptions = None
        neg_relation_descriptions = None
        if vg_data_loader != None:
            try:
                vg_batch = next(vg_iter)
            except StopIteration:
                vg_iter = iter(vg_data_loader)
                vg_batch = next(vg_iter)
            if args.vg_loss_lambda > 0.0:
                if args.relations > 0:
                    if args.negatives:
                        vg_image, vg_text, valid_objects, bounding_boxes, object_descriptions_t, valid_relations, relations_bounding_boxes, relation_descriptions_t, neg_text, neg_mask, vg_idx =  vg_batch
                        object_descriptions = [list(x) for x in zip(*object_descriptions_t)]
                        relation_descriptions = [list(x) for x in zip(*relation_descriptions_t)]
                        vg_text += neg_text
                        neg_mask = neg_mask.to(device,non_blocking=True)
                    else:
                        vg_image, vg_text, valid_objects, bounding_boxes, object_descriptions_t, valid_relations, relations_bounding_boxes, relation_descriptions_t, vg_idx =  vg_batch
                        object_descriptions = [list(x) for x in zip(*object_descriptions_t)]
                        relation_descriptions = [list(x) for x in zip(*relation_descriptions_t)]
                    relations_descs, relations_targets = organize_batch_classes_relations(relation_descriptions,valid_relations,relations_bounding_boxes,args,device,neg_relation_descriptions=neg_relation_descriptions)
                    if len(relations_descs) > args.vg_batch_size * args.relations:
                        relations_descs = relations_descs[:args.vg_batch_size * args.relations]
                else:
                    if args.negatives:
                        vg_image, vg_text, valid_objects, bounding_boxes, object_descriptions_t, neg_text, neg_mask, vg_idx =  vg_batch
                        object_descriptions = [list(x) for x in zip(*object_descriptions_t)]
                        vg_text += neg_text
                        neg_mask = neg_mask.to(device,non_blocking=True)
                    else:
                        vg_image, vg_text, valid_objects, bounding_boxes, object_descriptions_t, vg_idx =  vg_batch
                        object_descriptions = [list(x) for x in zip(*object_descriptions_t)]
                objects_descs, targets = organize_batch_classes(object_descriptions, valid_objects, bounding_boxes, args, device, neg_object_descriptions=neg_object_descriptions)
                if len(objects_descs) > args.vg_batch_size * args.objects:
                    objects_descs = objects_descs[:args.vg_batch_size * args.objects]
            else:
                if args.negatives:
                    vg_image, vg_text, neg_text, neg_mask, vg_idx = vg_batch
                    vg_text += neg_text
                    neg_mask = neg_mask.to(device,non_blocking=True) 
                else:
                    vg_image, vg_text, vg_idx = vg_batch

            caption += vg_text
            image = torch.cat([image,vg_image])
            idx = torch.cat([idx,vg_idx])

        
        image = image.to(device,non_blocking=True)   
        idx = idx.to(device,non_blocking=True)

        vg_batch_size = 0 if vg_data_loader == None else vg_image.shape[0]
       
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/data_loader.num_batches)

        loss_ita, loss_itm, loss_neg, loss_dict, weight_dict = model(image, caption, alpha=alpha, idx=idx, vg_batch_size=vg_batch_size, ignore_mask=neg_mask, objects_descs = objects_descs, targets = targets, relations_descs = relations_descs, relations_targets=relations_targets)
        loss = loss_ita + loss_itm
        if args.negatives:
            loss += loss_neg * args.negatives_loss_lambda
        if args.vg_loss_lambda > 0.0:
            loss_ce = loss_dict["loss_ce"]
            loss_bbox = loss_dict["loss_bbox"]
            loss_giou = loss_dict["loss_giou"]
            ce_correct = loss_dict["ce_correct"]
            class_error = loss_dict["class_error"]
            loss_dict.pop("ce_correct")
            loss_sg = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) * args.vg_loss_lambda
            if args.relations > 0:
                loss_sg /= 2                 
            loss +=  loss_sg
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        if args.negatives:
            metric_logger.update(loss_neg=loss_neg.item())
        if args.vg_loss_lambda > 0.0:
            metric_logger.update(loss_ce=loss_ce.item())
            metric_logger.update(loss_bbox=loss_bbox.item())
            metric_logger.update(loss_giou=loss_giou.item())
            metric_logger.update(loss_sg=loss_sg.item())
            metric_logger.update(ce_correct=ce_correct)
            metric_logger.update(class_error=class_error)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  



def main(args, config):
    utils.init_distributed_mode(args) 
 
    
    device = torch.device(args.device)

    processor = blip_processor(config["image_size"])

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    args.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 
    cudnn.benchmark = True
    args.world_size = utils.get_world_size()

    if utils.is_main_process():
        params_file = os.path.join(args.output_dir, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                f.write(f"{name}: {val}\n")   

    #### Model #### 
    print("Creating model")
    model = blip_retrieval_vg(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                             queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'], args = args)

    if args.evaluate != "":
        if os.path.isfile(args.evaluate):
            checkpoint = torch.load(args.evaluate, map_location='cpu')
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            sd = checkpoint["state_dict"]
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)

    if args.lora != -1:
        mark_only_lora_as_trainable(model)
    
    if args.lock:
        for param in model.parameters():
            param.requires_grad = False

    if args.object_tokens > 0:
        model.visual_encoder.object_tokens.requires_grad_()

    if args.relation_tokens > 0:
        model.visual_encoder.relation_tokens.requires_grad_()
    
    if args.prompt_attention:
        if not args.prompts_lora > 0:
            for a in model.visual_encoder.blocks:
                for param in a.attn.qkv_prompts.parameters():
                    param.requires_grad_()
                for param in a.attn.proj_prompts.parameters():
                    param.requires_grad_()
    
    if args.prompt_attention_full:
            for b in model.visual_encoder.blocks:
                if not args.prompts_lora > 0:
                    for param in b.mlp_prompts.parameters():
                        param.requires_grad_()
                for param in b.norm1_prompts.parameters():
                    param.requires_grad_()
                for param in b.norm2_prompts.parameters():
                    param.requires_grad_()

    if args.vg_loss_lambda > 0.0:
        model.random_row.requires_grad_()
        model.no_object_row.requires_grad_()
        for param in model.bb_head.parameters():
            param.requires_grad_()
        for param in model.class_head.parameters():
            param.requires_grad_()
        if args.relations > 0:
            model.no_relation_row.requires_grad_()


    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay']) 

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))        

    #### laion Dataset ####
    if args.evaluate == "":
        print("Creating laion dataset")
        data = get_data(args, epoch=start_epoch)
        train_loader = data["train"].dataloader

    #### vg Dataset ####
    vg_dataloader = None
    if args.evaluate == "": 
        if args.vg_data:
            print("Creating vg dataset")
            vg_train_dataset = VgDatasetText(args.vg_data, processor, args.objects, args.vg_loss_lambda, negatives = args.negatives,  relations = args.relations)
            vg_dataloader = get_vg_loader(vg_train_dataset, args, args.vg_batch_size)
   
    print("Start training")
    start_time = time.time()  
    dist.barrier()  

    for epoch in range(start_epoch, config['max_epoch']):    
        if args.evaluate == "":        
            if args.distributed:
                data["train"].set_epoch(epoch)
                if vg_dataloader != None:
                    vg_dataloader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            
            if args.vg_data:
                train_stats = train(model, train_loader, optimizer, epoch, device, config, args,vg_data_loader=vg_dataloader)
            else:
                train_stats = train(model, train_loader, optimizer, epoch, device, config, args)
            

            #log train stats
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                        }
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            #save checkpoint
            if not args.checkpoint_frequency > 0:
                checkpoint_dict = {
                    "epoch": epoch + 1,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.output_dir, f"epoch_latest.pt")
                )
            else:
                if (epoch + 1) % args.checkpoint_frequency == 0:
                    checkpoint_dict = {
                        "epoch": epoch + 1,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(
                        checkpoint_dict,
                        os.path.join(args.output_dir, f"epoch_{epoch + 1}.pt")
                    )

                
        else:
            if args.winoground:
                processor = blip_processor(config["image_size"])
                winoground_dict, detailed_dict = evaluate_winoground(model_without_ddp, processor,device)
                winoground_folder = os.path.join(args.output_dir,"winoground")
                if not os.path.exists(winoground_folder):
                    os.mkdir(winoground_folder)
                winoground_dict_path = os.path.join(winoground_folder,"winoground")
                with open(os.path.join(winoground_dict_path), 'w',encoding='utf-8') as f:
                    json.dump(winoground_dict, f)


            if args.vlchecklist:
                vl_model = BLIP(f'epoch {epoch}',model_without_ddp, processor, device)
                vl_eval = Evaluate(config_file="VL_CheckList/configs/blip1.yaml", model = vl_model,epoch = epoch,args = args)
                vl_eval.start()
                vl_eval = Evaluate(config_file="VL_CheckList/configs/blip2.yaml", model = vl_model,epoch = epoch,args = args)
                vl_eval.start()
                vl_eval = Evaluate(config_file="VL_CheckList/configs/blip3.yaml", model = vl_model,epoch = epoch,args = args)
                vl_eval.start()
                vl_eval = Evaluate(config_file="VL_CheckList/configs/blip4.yaml", model = vl_model,epoch = epoch,args = args)
                vl_eval.start()
            
            if args.vsr:
                vsr_folder = os.path.join(args.output_dir,"vsr")
                vsr_dict_path2 = os.path.join(vsr_folder,"vsr_meta_" + str(epoch))
                results_by_cat, results_by_meta_cat = evaluate_vsr(model_without_ddp,blip_processor(config["image_size"]),device)
                if not os.path.exists(vsr_folder):
                    os.mkdir(vsr_folder)
                with open(os.path.join(vsr_dict_path2), 'w',encoding='utf-8') as f:
                    json.dump(results_by_meta_cat, f)


                    
        if args.evaluate != "" or epoch == (args.stop_after - 1): 
            break

        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/laion_vg.yaml')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument("--name", default="test")
    parser.add_argument('--evaluate', default = "", type=str)        
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--train-data', default = None, type=str)
    parser.add_argument('--train-num-samples', default = 0, type=int)
    parser.add_argument('--dataset-type', default = "auto", type=str)
    parser.add_argument('--workers', default = 4, type=int)
    parser.add_argument('--vg-data', default = "../Data", type=str)
    parser.add_argument('--vg-loss-lambda', default = 1.0, type=float)
    parser.add_argument('--negatives-loss-lambda', default = 1.0, type=float)
    parser.add_argument('--negatives', action='store_true')
    parser.add_argument('--batch-size', default = 32, type=int)
    parser.add_argument('--vg-batch-size', default = 8, type=int)
    parser.add_argument('--objects', default = 10, type=int)
    parser.add_argument('--object-tokens', default = 25, type=int)
    parser.add_argument('--relations', default = 7, type=int)
    parser.add_argument('--relation-tokens', default = 7, type=int)
    parser.add_argument('--head-layers', default = 3, type=int)
    parser.add_argument('--winoground', action='store_true')
    parser.add_argument('--vlchecklist', action='store_true')
    parser.add_argument('--checkpoint-frequency', default = 6, type=int)
    parser.add_argument('--vsr', action='store_true')
    parser.add_argument('--lora', default = 16, type=int)
    parser.add_argument('--text-lora', action='store_false')
    parser.add_argument('--image-lora', action='store_false')
    parser.add_argument('--prompts-lora', default = 32, type=int)
    parser.add_argument('--resume', default = None, type=str)
    parser.add_argument('--lr', default = 0.00005, type=float)
    parser.add_argument('--prompt-attention', action='store_false')
    parser.add_argument('--prompt-attention-full', action='store_false')
    parser.add_argument('--lora-cross',default = 32, type=int)
    parser.add_argument('--lock', action='store_true')
    parser.add_argument('--epochs', default = 8, type=int)
    parser.add_argument('--stop-after', default = 6, type=int)
    parser.add_argument("--loss-ce", default = 1.0, type=float)


        
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config["init_lr"] = args.lr
    if args.epochs != 0:
        config["max_epoch"] = args.epochs
    args.output_dir = os.path.join(args.output_dir,args.name)
    Path(os.path.join(args.output_dir)).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w')) 


    if args.train_num_samples == 0:
        args.train_num_samples = int(750000 * args.batch_size / 32) 
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"{name}: {val}\n") 


    
    main(args, config)