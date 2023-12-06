import os
from VL_CheckList.vl_checklist.vlp_model import VLPModel
from VL_CheckList.example_models.utils.helpers import LRUCache, chunks
import torch.cuda
from PIL import Image
import torch.nn.functional as F


class BLIP(VLPModel):
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    MAX_CACHE = 20

    def __init__(self, model_id, model, preprocess, device):
        self._models = LRUCache(self.MAX_CACHE)
        self.batch_size = 32
        self.device = device
        self.model_dir = "resources"
        self.model_id = model_id
        self.model = model
        self.preprocess = preprocess


    def model_name(self):
        return self.model_id

    def _load_model(self, model_id):
        if model_id is None:
            raise Exception("Model ID cannot be None.")
        if not self._models.has(model_id):
            self._models.put(model_id, [self.model, self.preprocess])
        return self._models.get(model_id)

    def _load_data(self, src_type, data):
        pass

    def predict(self,
                images: list,
                texts: list,
                src_type: str = 'local'
                ):

        model_list = self._load_model(self.model_id)
        model = model_list[0]
        preprocess = model_list[1]
        # process images by batch
        probs = []
        images_collect = []
        for chunk_i, chunk_t in zip(chunks(images, self.batch_size),chunks(texts, self.batch_size)):
            for j in range(len(chunk_i)):
                image = preprocess(Image.open(chunk_i[j]).convert("RGB")).to(self.device)
                images_collect.append(image)
            images_collect = torch.stack(images_collect)
            text = model.tokenizer(chunk_t, padding='max_length', truncation=True, max_length=35, 
                        return_tensors="pt").to(image.device)
            with torch.no_grad():
                image_embeds = model.visual_encoder(images_collect)
                image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
                output = model.text_encoder(text.input_ids,
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            )
                itm_output = model.itm_head(output.last_hidden_state[:,0,:])
                probabilities = torch.nn.functional.softmax(itm_output, dim=1)[:, 1].cpu().tolist()
                probs.extend(probabilities)
        return {"probs":probs}
        


