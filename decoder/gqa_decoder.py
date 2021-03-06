from oscar.modeling.modeling_bert import ImageBertForSequenceClassification
from transformers.pytorch_transformers import BertTokenizer, BertConfig
from oscar.utils.task_utils import _truncate_seq_pair
import torch
import pickle
import json
from decoder.base_decoder import BaseDecoder

class Decoder(BaseDecoder):
    """
	GQA Decoder Class to decode Encoded Information

	"""
    def __init__(self, \
                checkpoint = '/workspace/shared/gqa_output/checkpoint-9', \
                ans2label = '/workspace/shared/gqa/trainval_testdev_all_ans2label.pkl', \
                label2ans = '/workspace/shared/gqa/trainval_testdev_all_label2ans.pkl', \
                device = 'cuda', \
                max_seq_length = 165, \
                max_img_seq_length = 45, \
                cls_token_at_end = False, \
                cls_token_segment_id = 1, \
                pad_on_left = False, \
                pad_token_segment_id = 0, \
                idx2word = '/workspace/scene_graph_benchmark/visualgenome/VG-SGG-dicts-vgoi6-clipped.json'):
		
        """
		Initialized the GQA Decoder 
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
        self.device = device

        config_class = BertConfig
        model_class = ImageBertForSequenceClassification
        tokenizer_class = BertTokenizer

        config = config_class.from_pretrained(
            checkpoint,
            num_labels = len(list(pickle.load(open(ans2label, 'rb')).values())),
            finetuning_task = 'gqa'
        )

        self.tokenizer = tokenizer_class.from_pretrained(checkpoint, do_lower_case = True)

        config.img_feature_dim = 2054
        config.img_feature_type = 'faster_r-cnn'
        config.code_voc = 512
        config.hidden_dropout_prob = 0.2
        config.loss_type = 'xe'
        config.classifier = 'linear'
        config.cls_hidden_scale = 2
        config.spatial_dim = 6

        self.model = model_class.from_pretrained(checkpoint, from_tf = False, config = config)
        _ = self.model.to(self.device)

        self.max_seq_length = max_seq_length
        self.max_img_seq_length = max_img_seq_length

        self.cls_token_at_end = cls_token_at_end
        self.cls_token_segment_id = cls_token_segment_id
        self.pad_on_left = pad_on_left
        self.pad_token_segment_id = pad_token_segment_id

        self.idx2word = json.load(open(idx2word, 'r'))


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
        """
        Tensorized the example and feature vector of the encoded image

        : param ex                     (example) : example class of the encoded image
        : param img_feat               (Tensor)  : feature vector of the encoded image
        : param cls_token              (string)  : Token of cls
        : param sep_token              (string)  : Token of sep_token
        : param pad_token              (int)     : Token of pad
        : param sequence_a_segment_id  (int)     : Segment id of the a segment
        : param sequence_b_segment_id  (int)     : Segment id of the b segment
        : param mask_padding_with_zero (boolean) : Is mask padding with zero

        : return : Tensorized example to input the decoder
        """
        tokens_a = self.tokenizer.tokenize(ex.text_a)

        tokens_b = None
        if ex.text_b:
            txt_b_arr = ex.text_b.split(';')
            txt_label_ixs = []
            for txt_b_ix, txt_b_ele in enumerate(txt_b_arr):
                tokens_b_ele = self.tokenizer.tokenize(txt_b_ele)
                txt_label_ixs.extend([txt_b_ix] * len(tokens_b_ele))
            txt_b = ex.text_b.replace(';', ' ').strip()
            tokens_b = self.tokenizer.tokenize(txt_b)
            assert len(tokens_b) == len(txt_label_ixs)

            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
            txt_label_ixs = txt_label_ixs[0:len(tokens_b)]

        # original
        #if example.text_b:
        #    txt_b = example.text_b.replace(';', ' ').strip()
        #    tokens_b = self.tokenizer.tokenize(txt_b)
        #    _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)
        else: # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[:(self.max_seq_length - 2)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
        #print(segment_ids)
        if self.cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [self.cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [self.cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_length - len(input_ids)
        if self.pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([self.pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([self.pad_token_segment_id] * padding_length)

        #assert len(input_ids) == self.args.max_seq_length
        #assert len(input_mask) == self.args.max_seq_length
        #assert len(segment_ids) == self.args.max_seq_length


        #img_feat = self.img_features[example.img_key] #[:, 0:self.args.img_feature_dim]  # torch

        if img_feat.shape[0] > self.max_img_seq_length:
            img_feat = img_feat[0:self.max_img_seq_length, ]
            if self.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
        else:
            if self.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
            padding_matrix = torch.zeros((self.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            if self.max_img_seq_length > 0:
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])
                # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]

        label_id = [0]

        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor([label_id[0]], dtype=torch.long),
                torch.tensor([label_id[0]], dtype=torch.long),
                img_feat,
                torch.tensor([0], dtype=torch.long)) #torch.tensor([example.q_id], dtype=torch.long))

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
                                                    torch.Tensor(args[i][0]),\
                                                    cls_token=self.tokenizer.cls_token,\
                                                    sep_token=self.tokenizer.sep_token ))


        examples = tuple(map(torch.stack, zip(*examples)))

        with torch.no_grad():
            batch = tuple(t.to(self.device) for t in examples)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         None,
                      'img_feats':      batch[5]}
            
            outputs = self.model(**inputs)
            logits = outputs[0]

            arg_logits = torch.argsort(logits, dim = 1, descending=True).cpu().numpy()

            res = []
            for i in range(len(arg_logits)):
                temp = {}
                for j in range(5):
                    temp[self.label2ans[arg_logits[i][j]]] = logits[i][arg_logits[i][j]].cpu().numpy().item()
                res.append(temp)
            return res

        