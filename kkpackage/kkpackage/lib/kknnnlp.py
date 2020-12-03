import datetime, re
import pandas as pd
import mojimoji
from typing import List
import torch
import torchtext
import transformers as trf

# local package
from kkpackage.lib.kknn import BaseNN, TorchNN, Layer

class DataFrameDataset(torchtext.data.Dataset):
    """
    pandas DataFrameからtorchtextのdatasetを作成
    https://stackoverflow.com/questions/52602071/dataframe-as-datasource-in-torchtext
    """
    def __init__(self, examples, fields: dict, filter_pred=None):
        """
         Create a dataset from a pandas dataframe of examples and Fields
         Arguments:
             examples pd.DataFrame: DataFrame of examples
             fields {str: Field}: The Fields to use in this tuple. The
                 string is a field name, and the Field is the associated field.
             filter_pred (callable or None): use only exanples for which
                 filter_pred(example) is true, or use all examples if None.
                 Default is None
        Usage::
            #TEXT: torchtext.data.field.Field, LABEL: torchtext.data.field.Field
            DataFrameDataset(df_train, fields={'Colname1': TEXT, 'Colname2': LABEL})
        """
        self.examples = examples.apply(SeriesExample.fromSeries, args=(fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

class SeriesExample(torchtext.data.Example):
    """Class to convert a pandas Series to an Example"""
    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()

        for key, field in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                "the input data".format(key))
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex



class NLPNN(BaseNN):
    def __init__(
        self, num_labels: int, mtype: str="cls", 
        # Nlp
        nn_add: torch.nn.Module=None, fine_tuning_type: str="full", add_tokens: List[str] = None,
        # loss functions
        loss_funcs: List[object]=None, loss_funcs_valid: List[object]=None,
        # optimizer
        optimizer: torch.optim.Optimizer=torch.optim.SGD, optim_params: dict={"lr":0.001, "weight_decay":0},
        scheduler: torch.optim.lr_scheduler._LRScheduler=None ,scheduler_params: dict=None,
        # train parameter
        epoch: int=100, batch_size: int=-1, valid_step: int=-1, batch_size_valid: int=-1,
        early_stopping_rounds: int=-1, early_stopping_loss_diff: float=None, 
        # output
        outdir: str="./output_"+datetime.datetime.now().strftime("%Y%m%d%H%M%S"), save_step: int=None, 
        # others
        random_seed: int=0, num_workers: int=1
    ):
        # BERT + Torknizer 定義
        model          = trf.BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking', num_labels=num_labels)
        self.tokenizer = trf.BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        # add tokens
        if (not add_tokens) == False:
            self.tokenizer.add_tokens(add_tokens)
            model.resize_token_embeddings(len(self.tokenizer)) 
        # Tuning type
        if fine_tuning_type == 'fast':
            # 1. まず全部を、勾配計算Falseにしてしまう
            for name, param in model.named_parameters():
                param.requires_grad = False
            # 2. 最後のBertLayerモジュールを勾配計算ありに変更
            for name, param in model.bert.encoder.layer[-1].named_parameters():
                param.requires_grad = True
            # 3. 識別器を勾配計算ありに変更
            for name, param in model.classifier.named_parameters():
                param.requires_grad = True
        elif fine_tuning_type == 'full':
            pass
        else:
            logger.error('please input fine_tuning_type "fast" or "full"')
            raise ValueError
        self.nn_nlp = TorchNN(
            0, 
            Layer("", model,  None, None, None, None),
            Layer("", nn_add, None, None, None, None),
        ) if nn_add is not None else model
        # NLP Pre-processing
        self.TEXT = torchtext.data.Field(
            tokenize=self.nlp_preprocessing,
            sequential=True, use_vocab=True, lower=False,
            include_lengths=True, batch_first=True, fix_length=self.tokenizer.model_max_length,
            init_token='[CLS]', eos_token='[SEP]', pad_token='[PAD]', unk_token='[UNK]'
        )
        self.LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
        # Others
        self.dataset_colname: tuple = None
        self.dataset_train = None
        self.dataset_valid = None
        self.dataloader_train  = None
        self.dataloader_valids = None
        # init
        super().__init__(
            self.nn_nlp, mtype, 
            loss_funcs=loss_funcs, loss_funcs_valid=loss_funcs_valid,
            optimizer=optimizer, optim_params=optim_params,
            scheduler=scheduler, scheduler_params=scheduler_params,
            epoch=epoch, batch_size=batch_size, valid_step=valid_step, batch_size_valid=batch_size_valid,
            early_stopping_rounds=early_stopping_rounds, early_stopping_loss_diff=early_stopping_loss_diff, 
            outdir=outdir, save_step=save_step, random_seed=random_seed, num_workers=num_workers
        )

    def _build_vocab(self, dataset: torchtext.data.Dataset=None, min_freq: int=1):
        self.TEXT.build_vocab(dataset if dataset is not None else [], min_freq=min_freq) # vocab を作成(初期化)する
        self.TEXT.vocab.stoi = self.tokenizer.vocab.copy() # string to index の変換辞書, tokenizerで置き換え
        for x, y in self.tokenizer.added_tokens_encoder.items(): self.TEXT.vocab.stoi[x] = y
        self.TEXT.vocab.itos = [x for x in self.TEXT.vocab.stoi.keys()]

    def nlp_preprocessing(self, text: str):
        # 半角、全角の変換
        text = mojimoji.han_to_zen(text)
        # 改行、半角スペース、全角スペースを削除
        text = text.replace("\r", "").replace("\n", "").replace(" ", "").replace("　", "")
        # 数字文字の一律「0」化
        text = re.sub(r'[0-9 ０-９]', '0', text)  # 数字
        ret = self.tokenizer.tokenize(text)
        return ret
    
    def create_df_dataloader(self, df: pd.DataFrame, batch_size: int, num_workers: int= 1):
        dataset_colname = self.dataset_colname if self.dataset_colname is not None else (df.columns[0], df.columns[1], )
        dataset    = DataFrameDataset(df, {dataset_colname[0]: self.TEXT, dataset_colname[1]: self.LABEL})
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size if batch_size > 0 else df.shape[0], 
            shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=self.collate_fn
        )
        return dataset, dataloader

    def train(self, df_train: pd.DataFrame, df_valid: pd.DataFrame=None):
        # dataset (build_vocab のためにtorchtext.data.Dataset を使う)
        ## torchtext.data.Iterator(self.dataset_train, batch_size=self.batch_size, train=True) これは使い辛いので不採用
        self.dataset_colname = (df_train.columns[0], df_train.columns[1], )
        dataset, dataloader = self.create_df_dataloader(df_train, self.batch_size, self.num_workers)
        self.dataset_train    = dataset
        self.dataloader_train = dataloader
        if df_valid is not None:
            dataset, dataloader = self.create_df_dataloader(df_valid, self.batch_size_valid, self.num_workers)
            self.dataset_valid     = dataset
            self.dataloader_valids = [dataloader, ]
        if not hasattr(self.TEXT, 'vocab'):
            self._build_vocab(self.dataset_train, min_freq=1) # vocab instance の作成
        super().train()
    
    def collate_fn(self, batch, to_tensor=True, ):
        sequences, labels = [], []
        for data in batch:
            sequence = data.__getattribute__(self.dataset_colname[0])
            sequence = self.TEXT.process([sequence])[0]
            label    = data.__getattribute__(self.dataset_colname[1])
            if isinstance(label, list): label = self.LABEL.process([label])[0]
            sequences.append(sequence)
            labels.   append(label)
        if to_tensor:
            sequences = torch.cat(sequences, axis=0)
            labels    = torch.Tensor(labels)
        return sequences, labels
