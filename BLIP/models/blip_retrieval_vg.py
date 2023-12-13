import sys
sys.path.insert(0, '/home/gamir/DER-Roei/alon/SGVL/BLIP')
import sys
sys.path.insert(0, '/home/gamir/DER-Roei/alon/SGVL/BLIP')
from models.med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

from models.blip import create_vit, init_tokenizer, load_checkpoint

from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

class MLP_Head(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class HNLoss(nn.Module):
    def __init__(
            self,
            alpha = 1.0
    ):
        super().__init__()
        self.alpha = alpha


    def forward(self, image_features, text_features, ignore_mask, vg_batch_size):
        vg_image_features = image_features[-vg_batch_size:,:]
        positive_text_features = text_features[-2*vg_batch_size:-vg_batch_size,:]
        negative_text_features = text_features[-vg_batch_size:,:]
        positive_similarity = torch.diagonal(vg_image_features @ positive_text_features.T)
        negative_similarity = torch.diagonal(vg_image_features @ negative_text_features.T)
        positive_similarity = torch.exp(positive_similarity)
        negative_similarity = torch.exp(negative_similarity)
        denominator = positive_similarity + negative_similarity
        loss_per_sample = -torch.log(torch.div(positive_similarity,denominator))
        loss = self.alpha * torch.dot(loss_per_sample, ignore_mask)/torch.sum(ignore_mask)
        return loss 

class BLIP_Retrieval_vg(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                      
                 embed_dim = 256,     
                 queue_size = 57600,
                 momentum = 0.995,
                 negative_all_rank = False,
                 args = None 
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()

        device = torch.device(args.device)

        self.objects = args.objects
        self.object_tokens = args.object_tokens
        self.relations = args.relations
        self.relation_tokens = args.relation_tokens
        self.prompt_attention = True if args.prompt_attention else False
        self.prompt_attention_full = True if args.prompt_attention_full else False
        self.text_lora = args.text_lora
        self.image_lora = args.image_lora
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, lora = args.lora if self.image_lora else -1, prompts_lora = args.prompts_lora, objects = self.object_tokens, relations = self.relation_tokens, prompt_attention = self.prompt_attention, prompt_attention_full = self.prompt_attention_full)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False, lora = args.lora if self.text_lora else -1, lora_cross = args.lora_cross)          

        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2) 



        
        # create momentum encoders  
        self.visual_encoder_m, vision_width = create_vit(vit,image_size, lora = args.lora if self.image_lora else -1, prompts_lora = args.prompts_lora, objects = self.object_tokens, relations = self.relation_tokens, prompt_attention = self.prompt_attention , prompt_attention_full = self.prompt_attention_full)              
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False, lora = args.lora if self.text_lora else -1, lora_cross = args.lora_cross)    
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]       
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1,queue_size),-100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))  

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))   
        
        self.negative_all_rank = negative_all_rank

        self.negatives_loss = args.negatives

        if self.negatives_loss:
            self.hn_loss = HNLoss()

        self.vg_loss_lambda = args.vg_loss_lambda

        if self.vg_loss_lambda > 0.0:
            weight_dict = {'loss_ce': args.loss_ce, 'loss_bbox': 5}
            weight_dict['loss_giou'] = 2
            losses = ['labels','boxes','cardinality']
            matcher = HungarianMatcher(cost_class=weight_dict["loss_ce"],cost_bbox=weight_dict["loss_bbox"],cost_giou=weight_dict["loss_giou"]) 


            self.num_matcher_classes = args.vg_batch_size * args.objects

            self.vgcriterion = SetCriterion(self.num_matcher_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=(5.5/self.object_tokens), losses=losses)
            self.vgcriterion.to(args.device)

            self.class_head = MLP_Head(vision_width, vision_width, embed_dim,args.head_layers).to(device)
            self.bb_head = MLP_Head(vision_width, vision_width, 4, args.head_layers).to(device)
            self.random_row = nn.Parameter(torch.zeros(1,embed_dim))
            self.no_object_row = nn.Parameter(torch.zeros(1,embed_dim))
            if self.relations > 0:
                self.num_relation_classes = args.vg_batch_size * args.relations
                self.vgrelcriterion = SetCriterion(self.num_relation_classes, matcher=matcher, weight_dict=weight_dict,
                                eos_coef=(1.8/self.relation_tokens), losses=losses)
                self.vgrelcriterion.to(args.device)
                self.no_relation_row = nn.Parameter(torch.zeros(1,embed_dim))
                self.rel_class_head = MLP_Head(vision_width, vision_width, embed_dim,args.head_layers).to(device)
                self.rel_bb_head = MLP_Head(vision_width, vision_width, 4, args.head_layers).to(device)
        
    def forward(self, image, caption, alpha, idx, vg_batch_size = 0, ignore_mask = None, objects_descs = None, targets = None, relations_descs = None, relations_targets = None):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)
        

        #extract object tokens from image encoder and add object descriptions to caption
        if self.vg_loss_lambda > 0:
            object_tokens = image_embeds[-vg_batch_size:,1 : 1 + self.object_tokens ,:]    
            caption += objects_descs
            if self.relations > 0:
                relation_tokens = image_embeds[-vg_batch_size:,1 + self.object_tokens : 1 + self.object_tokens + self.relation_tokens,:]
                caption += relations_descs

        text_no_adds = self.tokenizer(caption[:image.shape[0]],padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device)
        
        if self.negatives_loss:
                        text_negs = self.tokenizer(caption[image.shape[0]:image.shape[0] + vg_batch_size],padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device)
        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device) 
        
        
        
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)

        #here text features includes laion, vg, vg_neg and object_descs + ""
        #we separate object descs from the rest
        if self.vg_loss_lambda > 0:
            if self.relations > 0:
                num_relation_descs = len(relations_descs)
                text_feat = text_feat[:-(num_relation_descs)]
            num_object_descs = len(objects_descs)
            text_feat = text_feat[:-(num_object_descs)]

        
        #calculate negatives loss
        neg_loss = 0.0
        if self.negatives_loss:
            neg_loss = self.hn_loss(image_feat, text_feat, ignore_mask, vg_batch_size)
            #remove neagtives
            text_feat = text_feat[:-vg_batch_size]


                
        ###============== Image-text Contrastive Learning ===================###
        idx = idx.view(-1,1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
        pos_idx = torch.eq(idx, idx_all).float()       
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
        
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image) 
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
            image_feat_m_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                   
            
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1)


            #separate object descs and remove negatives
            if self.vg_loss_lambda > 0.0:
                if self.relations > 0:
                    relations_descs_feat_m = text_feat_m[-num_relation_descs:]
                    text_feat_m = text_feat_m[:-num_relation_descs]
                objects_descs_feat_m = text_feat_m[-num_object_descs:]
                text_feat_m = text_feat_m[:-num_object_descs]

            if self.negatives_loss:
                    text_feat_m = text_feat_m[:-vg_batch_size]
            

            text_feat_m_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = image_feat_m @ text_feat_m_all / self.temp  
            sim_t2i_m = text_feat_m @ image_feat_m_all / self.temp   

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        sim_i2t = image_feat @ text_feat_m_all / self.temp 
        sim_t2i = text_feat @ image_feat_m_all / self.temp 
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2
        
        idxs = concat_all_gather(idx)
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idxs)   

        ###============== Hungarian Matching ====================###
        if self.vg_loss_lambda > 0.0:
            no_object_rows_to_add = self.num_matcher_classes - num_object_descs
            random_rows = self.random_row
            no_object_row = self.no_object_row.to(image.device)
            random_rows = random_rows.expand(no_object_rows_to_add,-1).to(image.device)
            objects_descs_feat_m = torch.cat([objects_descs_feat_m,random_rows,no_object_row])
            label_embeddings = self.class_head(object_tokens)
            label_predictions = label_embeddings @ objects_descs_feat_m.t() / self.temp 
            bb_predictions = self.bb_head(object_tokens).sigmoid()
            predictions_dict = {"pred_logits" : label_predictions, "pred_boxes": bb_predictions}
            loss_dict = self.vgcriterion(predictions_dict, targets)
            weight_dict = self.vgcriterion.weight_dict
            if self.relations > 0:
                no_relation_rows_to_add = self.num_relation_classes - num_relation_descs
                random_rows = self.random_row
                no_relation_row = self.no_relation_row.to(image.device)
                random_rows = random_rows.expand(no_relation_rows_to_add,-1).to(image.device)
                relations_descs_feat_m = torch.cat([relations_descs_feat_m,random_rows,no_relation_row])
                label_embeddings = self.rel_class_head(relation_tokens)
                label_predictions = label_embeddings @ relations_descs_feat_m.t() / self.temp 
                bb_predictions = self.rel_bb_head(relation_tokens).sigmoid()
                predictions_dict = {"pred_logits" : label_predictions, "pred_boxes": bb_predictions}
                relation_loss_dict = self.vgrelcriterion(predictions_dict, relations_targets)
                loss_dict = {k: loss_dict[k] + relation_loss_dict[k] for k in loss_dict}
                  
        else:
            loss_dict = None
            weight_dict = None 

        ###============== Image-text Matching ===================###
        encoder_input_ids = text_no_adds.input_ids.clone()
        encoder_input_ids[:,0] = self.tokenizer.enc_token_id

        # forward the positve image-text pair
        bs = image.size(0)
        

        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask = text_no_adds.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )  
        
        
        if self.negative_all_rank:    
            # compute sample similarity
            with torch.no_grad():                
                mask = torch.eq(idx, idxs.t())

                image_feat_world = concat_all_gather(image_feat)
                text_feat_world = concat_all_gather(text_feat)

                sim_i2t = image_feat @ text_feat_world.t() / self.temp 
                sim_t2i = text_feat @ image_feat_world.t() / self.temp 

                weights_i2t = F.softmax(sim_i2t,dim=1)
                weights_i2t.masked_fill_(mask, 0)            

                weights_t2i = F.softmax(sim_t2i,dim=1)
                weights_t2i.masked_fill_(mask, 0)     

            image_embeds_world = all_gather_with_grad(image_embeds) 

            # select a negative image (from all ranks) for each text
            image_embeds_neg = []    
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

            # select a negative text (from all ranks) for each image
            input_ids_world = concat_all_gather(encoder_input_ids)
            att_mask_world = concat_all_gather(text_no_adds.attention_mask)        

            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(input_ids_world[neg_idx])
                text_atts_neg.append(att_mask_world[neg_idx])
                
        else:
            with torch.no_grad():                
                mask = torch.eq(idx, idx.t())
                
                sim_i2t = image_feat @ text_feat.t() / self.temp 
                sim_t2i = text_feat @ image_feat.t() / self.temp 

                weights_i2t = F.softmax(sim_i2t,dim=1)
                weights_i2t.masked_fill_(mask, 0)            

                weights_t2i = F.softmax(sim_t2i,dim=1)
                weights_t2i.masked_fill_(mask, 0)     

            # select a negative image (from same rank) for each text
            image_embeds_neg = []    
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

            # select a negative text (from same rank) for each image    
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(text_no_adds.attention_mask[neg_idx])            
            
        text_ids_neg = torch.stack(text_ids_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg],dim=0)     
        text_atts_all = torch.cat([text_no_adds.attention_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask = text_atts_all,
                                       encoder_hidden_states = image_embeds_all,
                                       encoder_attention_mask = image_atts_all,      
                                       return_dict = True,
                                      )                         
          

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)  

        #vg negatives loss
        if self.negatives_loss:
            text_negs_input_ids = text_negs.input_ids.clone()
            text_negs_input_ids[:,0] = self.tokenizer.enc_token_id
            output_neg_vg = self.text_encoder(text_negs_input_ids,
                                        attention_mask = text_negs.attention_mask,
                                        encoder_hidden_states = image_embeds[-vg_batch_size:],
                                        encoder_attention_mask = image_atts[-vg_batch_size:],      
                                        return_dict = True,
                                        )  

            vl_vg_embeddings = torch.cat([output_pos.last_hidden_state[-vg_batch_size:,0,:], output_neg_vg.last_hidden_state[:,0,:]],dim=0)
            vl_vg_output = self.itm_head(vl_vg_embeddings)
            itm_vg_labels = torch.cat([torch.ones(vg_batch_size,dtype=torch.long),torch.zeros(vg_batch_size,dtype=torch.long)],
                               dim=0).to(image.device)
            loss_vg_itm = F.cross_entropy(vl_vg_output, itm_vg_labels)
            neg_loss += loss_vg_itm
            neg_loss /= 2

        return loss_ita, loss_itm, neg_loss, loss_dict, weight_dict
 

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        

        batch_size = image_feats.shape[0]

        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size # move pointer

        self.ptr_queue[0] = ptr  


def blip_retrieval_vg(pretrained='',**kwargs):
    model = BLIP_Retrieval_vg(**kwargs)
    args = kwargs["args"]    
    if pretrained and args.evaluate == "":
        model,msg = load_checkpoint(model,pretrained)
        with torch.no_grad():
            if args.prompts_lora != -1:
                for b in model.visual_encoder.blocks:
                    b.mlp_prompts.fc1.weight.copy_(b.mlp.fc1.weight)
                    b.mlp_prompts.fc1.bias.copy_(b.mlp.fc1.bias)
                    b.mlp_prompts.fc2.weight.copy_(b.mlp.fc2.weight)
                    b.mlp_prompts.fc2.bias.copy_(b.mlp.fc2.bias)
                    b.attn.qkv_prompts.weight.copy_(b.attn.qkv.weight)
                    b.attn.qkv_prompts.bias.copy_(b.attn.qkv.bias)
                    b.attn.proj_prompts.weight.copy_(b.attn.proj.weight)
                    b.attn.proj_prompts.bias.copy_(b.attn.proj.bias)
                    b.norm1_prompts.weight.copy_(b.norm1.weight)
                    b.norm1_prompts.bias.copy_(b.norm1.bias)
                    b.norm2_prompts.weight.copy_(b.norm2.weight)
                    b.norm2_prompts.bias.copy_(b.norm2.bias)
            if args.lora_cross == True:
                for l in model.text_encoder.encoder.layer:
                    l.crossattention.self.key_prompts.weight.copy_(l.crossattention.self.key.weight)
                    l.crossattention.self.key_prompts.bias.copy_(l.crossattention.self.key.bias)
                    l.crossattention.self.value_prompts.weight.copy_(l.crossattention.self.value.weight)
                    l.crossattention.self.value_prompts.bias.copy_(l.crossattention.self.value.bias)


    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output      


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)
