from oscar.modeling.modeling_bert import BertForImageCaptioning
from transformers.pytorch_transformers import BertTokenizer, BertConfig
from oscar.run_captioning import CaptionTensorizer
import numpy as np
import torch

class Decoder(object):
	def __init__(self, checkpoint, device):
		self.checkpoint = checkpoint
		
		self.config = BertConfig.from_pretrained(self.checkpoint)
		self.config.output_hidden_states = True
		
		self.tokenizer = BertTokenizer.from_pretrained(self.checkpoint)
		self.model = BertForImageCaptioning.from_pretrained(self.checkpoint, config = self.config)

		self.device = device
		self.model.to(self.device)
		self.model.eval()

		self.tensorizer = CaptionTensorizer(self.tokenizer, max_img_seq_length = 50, max_seq_length = 70, \
											max_seq_a_length = 40, mask_prob = 0.15, max_masked_tokens = 3, is_train = False)
		

	def decode(self, features, classes):
		examples = [[],[],[],[],[]]
		caption = ""
		for i in range(len(classes)):
			od_labels = " ".join([str(t) for t in np.array(classes[i])])
			example =   self.tensorizer.tensorize_example(caption, torch.Tensor(features[i]), text_b = od_labels)
			cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
						self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token, self.tokenizer.sep_token, \
						self.tokenizer.pad_token, self.tokenizer.mask_token, '.'])
			for j in range(5):
				examples[j].append(example[j])

		examples = tuple(torch.stack(ex) for ex in examples)
		with torch.no_grad():
			batch = tuple(t.to(self.device) for t in examples)
			inputs = {
				'input_ids': batch[0], 'attention_mask': batch[1],
				'token_type_ids': batch[2], 'img_feats': batch[3],
				'masked_pos': batch[4],
			}
			inputs_param = {   'is_decode': True,
								'do_sample': False,
								'bos_token_id': cls_token_id,
								'pad_token_id': pad_token_id,
								'eos_token_ids': [sep_token_id],
								'mask_token_id': mask_token_id,
								# for adding od labels
								'add_od_labels': True, 'od_labels_start_posid': 40,
								# hyperparameters of beam search
								'max_length': 20,
								'num_beams': 5,
								"temperature": 1,
								"top_k": 0,
								"top_p": 1,
								"repetition_penalty": 1,
								"length_penalty": 1,
								"num_return_sequences": 3,
								"num_keep_best": 3,
							}
			inputs.update(inputs_param)
			output = self.model(**inputs)
		
		captions = []

		for i in range(len(classes)):
			captions.append(self.tokenizer.decode(output[0][i * 3][0].tolist(), skip_special_tokens=True))
		return captions

		
