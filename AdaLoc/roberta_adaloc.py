import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import transformers

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def DensewithBN(in_fea, out_fea, batch_norm=True, dropout=0):
    layers = [nn.Linear(in_fea, out_fea)]
    if batch_norm == True:
        layers.append(nn.BatchNorm1d(num_features=out_fea))
    layers.append(nn.ReLU())
    if dropout>0:
        layers.append(nn.Dropout(p=dropout))
    return layers

def DensewithLN(in_fea, out_fea, layer_norm=True, dropout=0, gelu=False):
    """It has been the standard to use batchnorm in CV tasks, and layernorm in NLP tasks. """
    layers = [nn.Linear(in_fea, out_fea)]
    if layer_norm == True:
        layers.append(nn.LayerNorm(normalized_shape=(out_fea,))) # normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None
    if dropout>0:
        layers.append(nn.Dropout(p=dropout))
    if gelu is True:
        layers.append(nn.GELU())
    return layers


class RobertaSentenceHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self,
                 hidden_size=1024,
                 num_labels=3,
                 dropout=0.1,
                 roberta_detector_name=None,
                 cache_dir: str = "/projectnb/ivc-ml/zpzhang/checkpoints/transformers_cache",
                 ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

        if roberta_detector_name:
            self.roberta_tokenizer = transformers.AutoTokenizer.from_pretrained(roberta_detector_name,cache_dir=cache_dir)
            self.roberta_detector = transformers.AutoModelForSequenceClassification.from_pretrained(
                roberta_detector_name, cache_dir=cache_dir).to(DEVICE)
            self.roberta_detector.eval()
    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x # (batch_size, num_labels)

    def extract_roberta_feature(self, text):
        sample_manipulated_article_token = self.roberta_tokenizer(text,
                                                             padding='max_length',  # longest, max_length, False
                                                             truncation=True,
                                                             max_length=512,
                                                             return_tensors="pt").to(
            DEVICE)  # (1, text_length), text_length should be smaller than 512
        sample_manipulated_article_embeddings = self.roberta_detector(**sample_manipulated_article_token,
                                                                 output_hidden_states=True, return_dict=True)
        last_hidden_state = sample_manipulated_article_embeddings['hidden_states'][-1]  # (1,512,1024)
        return last_hidden_state

if __name__=="__main__":
    pass





# class RobertaSentenceHead(nn.Module):
#     def __init__(self,
#                  embeddings_dim: int = 1024,  # dimension of input embeddings
#                  text_length: int = 512,
#                  hidden_dim: list = [1024, 16],
#                  classifier_dim: list = [1024, 3], # 3: output_dim
#                  layer_norm: bool=True,
#                  dropout: bool = 0.1,
#                  roberta_detector_name: str=None,
#                  cache_dir: str="/projectnb/ivc-ml/zpzhang/checkpoints/transformers_cache",
#                  ):
#         super(RobertaSentenceHead, self).__init__()
#         self.embeddings_dim = embeddings_dim
#         self.text_length = text_length
#         self.hidden_dim = hidden_dim
#
#         self.layer_norm = layer_norm
#         self.dropout=dropout
#
#         if len(hidden_dim)==2:
#             self.localization_dense1 = nn.Sequential(*DensewithLN(in_fea=embeddings_dim, out_fea=hidden_dim[0],
#                                                               layer_norm=layer_norm, dropout=dropout, gelu=True)) # with timeline
#             self.localization_dense2 = nn.Linear(hidden_dim[0], hidden_dim[1]) # with timeline
#
#         self.SentenceClassificationHead = nn.Sequential(
#                                 *DensewithLN(in_fea=text_length*hidden_dim[1], out_fea=classifier_dim[0],
#                                                layer_norm=False, dropout=dropout, gelu=False),
#                                 nn.Linear(classifier_dim[0], classifier_dim[1]))
#         if roberta_detector_name:
#             self.roberta_tokenizer = transformers.AutoTokenizer.from_pretrained(roberta_detector_name,cache_dir=cache_dir)
#             self.roberta_detector = transformers.AutoModelForSequenceClassification.from_pretrained(
#                 roberta_detector_name, cache_dir=cache_dir).to(DEVICE)
#
#     def forward(self, roberta_embeddings): # roberta_embeddings/last hidden states: (batch_size, 512, 1024)
#         output_timeline = self.localization_dense1(roberta_embeddings) # output_timeline: (batch_size, 512, 256)
#         output_timeline = self.localization_dense2(output_timeline) # output_timeline: (batch_size, 512, 16)
#         output = output_timeline.view(output_timeline.shape[0],-1) # output: (batch_size, 8192)
#         output = self.SentenceClassificationHead(output) # (batch_size, n_sentences_window)
#         return output
#
#     def extract_roberta_feature(self, text):
#         sample_manipulated_article_token = self.roberta_tokenizer(text,
#                                                              padding='max_length',  # longest, max_length, False
#                                                              truncation=True,
#                                                              max_length=512,
#                                                              return_tensors="pt").to(
#             DEVICE)  # (1, text_length), text_length should be smaller than 512
#         sample_manipulated_article_embeddings = self.roberta_detector(**sample_manipulated_article_token,
#                                                                  output_hidden_states=True, return_dict=True)
#         last_hidden_state = sample_manipulated_article_embeddings['hidden_states'][-1]  # (1,512,1024)
#         return last_hidden_state







