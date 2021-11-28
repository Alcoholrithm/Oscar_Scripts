from oscar.modeling.modeling_bert import ImageBertForSequenceClassification
from transformers.pytorch_transformers import BertTokenizer, BertConfig
from oscar.utils.task_utils import _truncate_seq_pair
import torch
import pickle
import json
from decoder.base_decoder import BaseDecoder

class Decoder(BaseDecoder):
    """
	VQA Decoder Class to decode Encoded Information

	"""
    def __init__(self, \
                checkpoint = '/workspace/shared/vqa_models/large/checkpoint-24-396575', \
                ans2label = '/workspace/shared/vqa/trainval_ans2label.pkl', \
                label2ans = '/workspace/shared/vqa/trainval_label2ans.pkl', \
                device = 'cuda', \
                max_seq_length = 128, \
                max_img_seq_length = 50, \
                cls_token_at_end = False, \
                cls_token_segment_id = 1, \
                pad_on_left = False, \
                pad_token_segment_id = 0, \
                idx2word = '/workspace/scene_graph_benchmark/visualgenome/VG-SGG-dicts-vgoi6-clipped.json'
    ):
        """
		Initialized the VQA Decoder 
		: param checkpoint        (string) : Location of Checkpoint
        : param ans2label         (string) : Location of ans2label file
        : param label2ans         (string) : Location of label2ans file
		: param device            (string) : Device to load the Decoder
        : param max_seq_length       (int) : Maximum length of sequence
        : param max_img_seq_length   (int) : Maximum length of image sequence
        : param cls_token_at_end (boolean) : Is the cls token is at the end
        : param cls_token_segment_id (int) : ID of cls token segment
        : param pad_on_left      (boolean) : Is the pad on left
        : param pad_token_segment_id (int) : ID of pad token segment
        : param idx2word          (string) : Location of idx2word file that convert integer object label to string object label
		
		: return : None
		"""

        self.label2ans = pickle.load(open(label2ans, 'rb'))
        self.ans2label = list(pickle.load(open(ans2label, 'rb')).values())
        self.device = device

        config_class = BertConfig
        model_class = ImageBertForSequenceClassification
        tokenizer_class = BertTokenizer

        config = config_class.from_pretrained(
            checkpoint,
            num_labels = len(self.ans2label),
            finetuning_task = 'vqa_text'
        )

        self.tokenizer = tokenizer_class.from_pretrained(checkpoint, do_lower_case = True)

        config.img_feature_dim = 2054
        config.img_feature_type = 'faster_r-cnn'
        config.code_voc = 512
        config.hidden_dropout_prob = 0.3
        config.loss_type = 'bce'
        config.classifier = 'linear'
        config.cls_hidden_scale = 3
        
        self.model = model_class.from_pretrained(checkpoint, from_tf = False, config = config)

        _ = self.model.to(self.device)
        self.model.eval()

        self.max_seq_length = max_seq_length
        self.max_img_seq_length = max_img_seq_length

        self.cls_token_at_end = cls_token_at_end
        self.cls_token_segment_id = cls_token_segment_id
        self.pad_on_left = pad_on_left
        self.pad_token_segment_id = pad_token_segment_id

        self.idx2word = json.load(open(idx2word,'r'))

    class example():
        """
        A data structure that converts raw data to be suitable for use in a decoder model

        """
        def __init__(self, question, od_tag, idx2word):
            """
            Construct appropriate structure from raw data

            : param question (string) : A question about the encoded image
            : param od_tag   (Tensor) : A Tensor of object labels in the encoded image

            : return : None
            """
            self.text_a = question
            self.text_b = ' '.join([idx2word['idx_to_label'][str(t)] for t in od_tag])

    def tensorize_example(self, ex, img_feat,
                    cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                    sequence_a_segment_id=0, sequence_b_segment_id=1,
                    mask_padding_with_zero=True):

        tokens_a = self.tokenizer.tokenize(ex.text_a)

        tokens_b = self.tokenizer.tokenize(ex.text_b)

        # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        tokens = [cls_token] + tokens
        segment_ids = [self.cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([self.pad_token_segment_id] * padding_length)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        if img_feat.shape[0] > self.max_img_seq_length:
            img_feat = img_feat[0:self.max_img_seq_length, ]
            if self.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
        else:
            if self.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
            padding_matrix = torch.zeros((self.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            if self.max_img_seq_length > 0:
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])

        label_id = [0]
        score = [0]

        new_scores = self.target_tensor(len(self.ans2label), label_id, score)

        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor([label_id[0]], dtype=torch.long),
                torch.tensor(new_scores, dtype=torch.float),
                img_feat,
                torch.tensor([0], dtype=torch.long))
    
    def target_tensor(self, len, labels, scores):
        """ 
        create the target by labels and scores 
        
        : param len (int) : Length of the target
        : parma labels (list) : list of label
        : param scores (list) : list of score

        return target (list) : list of target
        """
        target = [0]*len
        for id, l in enumerate(labels):
            target[l] = scores[id]

        return target
    
    def decode(self, args):
        """
        Main function to decode the encoded information to get answers from some questions

        : param args     (list) : A list of (feature vector, object labels, question)
        
        : return answers (list) : A list of answers to a given question.
        """

        examples = []
        for i in range(len(args)):
            ex = self.example(args[i][2], args[i][1].cpu().numpy(), self.idx2word)
            examples.append(self.tensorize_example(ex, \
                                                    torch.Tensor(args[i][0]), \
                                                    cls_token = self.tokenizer.cls_token, \
                                                    sep_token = self.tokenizer.sep_token ))
        
        examples = tuple(map(torch.stack, zip(*examples)))

        batch = tuple(t.to(self.device) for t in examples)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],  
                      'labels':         None,
                      'img_feats':      batch[5]}
            
            output = self.model(**inputs)
            logits = output[0]
            arg_logits = torch.argsort(logits, dim = 1, descending = True).cpu().numpy()

            res = []
            for i in range(len(arg_logits)):
                temp = {}
                for j in range(5):
                    temp[self.label2ans[arg_logits[i][j]]] = logits[i][arg_logits[i][j]].cpu().numpy().item()
                res.append(temp)
            return res


    





