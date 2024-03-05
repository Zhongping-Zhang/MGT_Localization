import os
import torch
import ujson as json
from tqdm import tqdm
from typing import Dict, List, Optional
from torch.utils.data.dataset import Dataset


class Article_base(Dataset):
    def __init__(self, file_path: str, article_num=10000):
        self.file_path=file_path
        self.article_num = article_num
        self._init_dataset()
        print("file_path", self.file_path)

    def __len__(self):
        return len(self.keys)

    def _init_dataset(self):
        assert os.path.isfile(self.file_path), f"Input_file_path {self.file_path} not found"
        with open(self.file_path, 'r') as f:
            self.articles = []
            for l_no, l in enumerate(f):
                self.articles.append(json.loads(l))
                if l_no>=self.article_num:
                    break
        self.data = {}
        print("total number of articles: ", len(self.articles))

        article_num=0
        for article in tqdm(self.articles, desc="load articles into self.data"):
            if "id" not in article:
                article['id'] = article_num
            article_pieces = {
            'title': "title: "+article['title']+". ",
            'article': "article: "+article['text'],
            }
            article_string = ''
            for piece_string in list(article_pieces.values()):
                article_string += piece_string
            self.data[article['id']] = article_string
            article_num+=1

        self.keys = list(self.data.keys())


class Goodnews_article(Article_base):
    """
    load original goodnews articles
    __getitem__ return: article_id, article_data
    """
    def __init__(self, **kwargs):
        super(Goodnews_article, self).__init__(**kwargs)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        article_id = self.keys[i]
        return str(article_id), self.data[article_id]

class Visualnews_article(Article_base):
    """
    load original visualnews articles
    __getitem__ return: article_id, article_data
    """
    def __init__(self, **kwargs):
        super(Visualnews_article, self).__init__(**kwargs)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        article_id = self.keys[i]
        return str(article_id), self.data[article_id]

class Wikitext_article(Article_base):
    """
    load original WikiText articles
    __getitem__ return: article_id, article_data
    """
    def __init__(self, **kwargs):
        super(Wikitext_article, self).__init__(**kwargs)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        article_id = self.keys[i]
        return str(article_id), self.data[article_id]

