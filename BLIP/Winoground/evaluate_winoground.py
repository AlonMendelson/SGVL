import sys
sys.path.insert(0, '/home/gamir/DER-Roei/alon/SGVL/BLIP')
from datasets import load_dataset
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import json

def blip_processor(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    return transform


def compute_itm(blip_model, caption, image):
    image_embeds = blip_model.visual_encoder(image) 
    image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
    
    text = blip_model.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                            return_tensors="pt").to(image.device) 

                
    output = blip_model.text_encoder(text.input_ids,
                                attention_mask = text.attention_mask,
                                encoder_hidden_states = image_embeds,
                                encoder_attention_mask = image_atts,      
                                return_dict = True,
                                )
    itm_output = blip_model.itm_head(output.last_hidden_state[:,0,:])     
    return itm_output

def evaluate_winoground(blip_model, blip_processor, device):
    blip_model.eval()
    def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(result):
        return image_correct(result) and text_correct(result)

    result_dict_itm = {}
    auth_token = "" #FILL IN HF AUTHENTICATION TOKEN
    winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]
    categories_blip_scores_itm = {}
    categories_blip_scores_itm["All Dataset"] = []

    #load tag assignments
    f = open("Winoground/tag_assignments.json")
    tag_assignments = json.load(f)

    for example in tqdm(winoground):
        image_0 = blip_processor(example["image_0"].convert("RGB")).unsqueeze(0)
        image_1 = blip_processor(example["image_1"].convert("RGB")).unsqueeze(0)
        caption_0 = example["caption_0"]
        caption_1 = example["caption_1"]
        image_0 = image_0.to(device)
        image_1 = image_1.to(device)
        with torch.no_grad():
            output_c0_i0 = compute_itm(blip_model=blip_model,caption=caption_0, image=image_0)
            output_c1_i0 = compute_itm(blip_model=blip_model,caption=caption_1, image=image_0)
            output_c0_i1 = compute_itm(blip_model=blip_model,caption=caption_0, image=image_1)
            output_c1_i1 = compute_itm(blip_model=blip_model,caption=caption_1, image=image_1)

            blip_itm_scores_c0_i0 = torch.nn.functional.softmax(output_c0_i0, dim=1)[:, 1].item()
            blip_itm_scores_c1_i0 = torch.nn.functional.softmax(output_c1_i0, dim=1)[:, 1].item()
            blip_itm_scores_c0_i1 = torch.nn.functional.softmax(output_c0_i1, dim=1)[:, 1].item()
            blip_itm_scores_c1_i1 = torch.nn.functional.softmax(output_c1_i1, dim=1)[:, 1].item()

        example_id = str(example["id"])
        all_tags = []
        all_tags.append("All Dataset")
        sample_dict_itm = {"id" : example["id"], "c0_i0": blip_itm_scores_c0_i0, "c0_i1": blip_itm_scores_c0_i1, "c1_i0": blip_itm_scores_c1_i0, "c1_i1": blip_itm_scores_c1_i1}
        for tag in all_tags:
            categories_blip_scores_itm[tag].append(sample_dict_itm)
        
        sample_result_dict_itm = {"text": True if text_correct(sample_dict_itm) else False, "image": True if image_correct(sample_dict_itm) else False, "group": True if group_correct(sample_dict_itm) else False}

        result_dict_itm[example_id] = sample_result_dict_itm

    winoground_dict = {}
    for category in categories_blip_scores_itm:
        category_blip_scores_itm = categories_blip_scores_itm[category]
        text_correct_count = 0
        image_correct_count = 0
        group_correct_count = 0
        for result in category_blip_scores_itm:
            text_correct_count += 1 if text_correct(result) else 0
            image_correct_count += 1 if image_correct(result) else 0
            group_correct_count += 1 if group_correct(result) else 0

        denominator = len(category_blip_scores_itm)

        metrics = {category + " text score": text_correct_count/denominator, 
        category + " image score": image_correct_count/denominator,
        category + " group score": group_correct_count/denominator,}

        winoground_dict[category] = metrics
    return winoground_dict, result_dict_itm
