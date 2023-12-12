import sys
sys.path.insert(0, '/home/gamir/DER-Roei/alon/SGVL/BLIP')

from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import pandas as pd
import json
import logging
from .vsr_dataset import VsrDataset, get_vsr_loader




def evaluate_vsr(blip_model, blip_processor, device):
    blip_model.eval()
    vsr_dataset = VsrDataset(blip_processor=blip_processor)
    vsr_loader = get_vsr_loader(vsr_dataset,32)
    all_results = []
    all_relations = []
    for batch in tqdm(vsr_loader):
        images, captions, labels, relations = batch
        images = images.to(device)
        labels = labels.to(device)
        image_embeds = blip_model.visual_encoder(images) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(images.device)        
        
        text = blip_model.tokenizer(captions, padding='max_length', truncation=True, max_length=35, 
                                return_tensors="pt").to(images.device) 

                    
        output = blip_model.text_encoder(text.input_ids,
                                    attention_mask = text.attention_mask,
                                    encoder_hidden_states = image_embeds,
                                    encoder_attention_mask = image_atts,      
                                    return_dict = True,
                                    )
        itm_output = blip_model.itm_head(output.last_hidden_state[:,0,:])
        predicted_label =  torch.argmax(itm_output,1)
        results = torch.eq(labels,predicted_label).detach().cpu().tolist()
        all_results += results
        all_relations += relations

    rel2cat = {}
    cats = []
    rel_count = 0
    with open("vsr/rel_meta_category_dict.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            cat, rels = line.strip().split(": ")
            cat = cat.strip()
            rel_list = rels.split(",")
            rel_count+=len(rel_list)
            rel_list = [rel.strip() for rel in rel_list]
            for rel in rel_list:
                rel2cat[rel] = cat
            cats.append(cat)
    print (f"# rel: {rel_count}")
    cats = list(set(cats))
    results_by_cat = {}
    results_by_meta_cat = {"total":{"corrects":0,"samples":0}}
    for i in range(len(all_results)):
        result = all_results[i]
        results_by_meta_cat["total"]["samples"] +=1
        if result == True:
            results_by_meta_cat["total"]["corrects"] += 1
        results_by_meta_cat["total"]["accuracy"] = results_by_meta_cat["total"]["corrects"]/results_by_meta_cat["total"]["samples"]
        
    return results_by_cat, results_by_meta_cat