import copy
from typing import List
import torch
import pandas as pd
import numpy as np
from scipy import stats

# local package
from kkpackage.util.learning import evalate
from kkpackage.util.common import check_type
from kkpackage.util.logger import set_logger
_logname = __name__
logger = set_logger()

class TorchNN(torch.nn.Module):
    """
    nn.Moduleを継承して新しいクラスを作る
    Layerの名前を固定して、複数入力ゆや複数出力に対応できるようにする
    *layers  : ((layer1), (layer2), (layer3), ...)
    ※layer1 : (layer_type, layer_name, () or (出力ノード数) or (入力ノード数, 出力ノード数), *param, **params)
    ※layer_type(string) : 基本的に定義の上から順に同typeのlayerを計算する
                         : InputX(Xは1から始まる番号. 別特徴量の入力を指す)
                         : Common(全結合層. InputXが横並びで入力される層)
                         : OutputX(Xは1から始まる番号. マルチタスク用に別出力層を定義)
    """    
    def __init__(self, *layers):
        self.logger = set_logger(_logname + ".TorchNN", log_level="info")
        self.logger.info("START")
        # 親クラスのコンストラクタ
        super().__init__()
        self.num_modules = 0
        self.num_inputs  = 0
        self.num_outputs = 0
        self.outjoin_list = []

        # 初めに、Layer構造チェック. 下記並びを原則とする
        # Input1, Input1, ..., Input2, ..., Common, Common, ..., Output1, Output1, ..., Output2, ..
        prev_type, prev_nodes = None, 0
        ilayer_fist_common, ilayer_last_common = 0, []
        fi_size_common,     lo_size_common     = 0, 0
        for i, layer in enumerate(layers):
            # layer_nameだけ取り出す
            layer_type, layer_name, nodes, *param = layer
            # 辞書形式があれば取り出す
            params = {}
            if (len(param) > 0) and (type(param[-1]) == dict):
                params = param[ -1]
                param  = param[:-1]
            if prev_type is None:
                if layer_type.find("Input1") == 0:
                    prev_type = layer_type
                    self.num_inputs += 1
                else:
                    self.logger.raise_error(f"unexpected value. layer_type: {layer_type}")
            elif prev_type.find("Input") == 0:
                if   layer_type.find("Common") == 0:
                    # Commonになる場合. これ以降はInputは来ない
                    prev_type = layer_type
                    fi_size_common += prev_nodes
                    ilayer_fist_common = i
                elif prev_type == layer_type:
                    # InputXが続く場合
                    prev_type = layer_type
                elif (layer_type.find("Input") == 0) and ((int(prev_type[-1])+1)==int(layer_type[-1])):
                    # InputXからInputX+1に変わる場合
                    prev_type = layer_type
                    fi_size_common += prev_nodes
                    self.num_inputs += 1
                else:
                    self.logger.raise_error(f"unexpected value. layer_type: {layer_type}")
            elif prev_type.find("Common") == 0:
                if   layer_type.find("Common") == 0:
                    prev_type = layer_type
                elif (layer_type.find("Output1") == 0):
                    # Commonが終わってOutputXに移る場合
                    prev_type = layer_type
                    lo_size_common += prev_nodes
                    ilayer_last_common.append(i)
                    self.num_outputs += 1
                else:
                    self.logger.raise_error(f"unexpected value. layer_type: {layer_type}")
            elif prev_type.find("Output") == 0:
                if   prev_type == layer_type:
                    # OutputXが続く場合
                    prev_type = layer_type
                elif (layer_type.find("Output") == 0) and ((int(prev_type[-1])+1)==int(layer_type[-1])):
                    # OutputXからOutputX+1に変わる場合
                    prev_type = layer_type
                    ilayer_last_common.append(i)
                    self.num_outputs += 1
                else:
                    self.logger.raise_error(f"unexpected value. layer_type: {layer_type}")

            # 出力ノードの記録
            if   len(nodes) == 0: continue
            elif len(nodes) == 1: prev_nodes = nodes[0]
            elif len(nodes) == 2: prev_nodes = nodes[1]
            else: self.logger.raise_error(f"unexpected value. nodes: {nodes}")
            # オプションによって出力のノード数が変わる場合があるのでその修正
            if (params.get("bidirectional") is not None) and (params.get("bidirectional") == True):
                prev_nodes = prev_nodes * 2
        
        # 層を追加していく
        for i, layer in enumerate(layers):
            # 出力内容の分解
            layer_type, layer_name, nodes, outjoin, *param = layer
            # 辞書形式があれば取り出す
            params = {}
            if (len(param) > 0) and (type(param[-1]) == dict):
                params = param[ -1]
                param  = param[:-1]

            # 入力ノード数, 出力ノード数の埋め込み
            ## InpuptX -> Common, Common -> OutputXの箇所が特殊
            if   len(nodes) == 0:
                pass
            elif len(nodes) == 1:
                self.out_size = nodes[0]
            elif len(nodes) == 2:
                self.in_size  = nodes[0]
                self.out_size = nodes[1]
            else:
                self.logger.raise_error(f"unexpected value. nodes: {nodes}")

            # 初回Common層や初回OutputX層の場合
            if i == ilayer_fist_common:   self.in_size = fi_size_common
            elif i in ilayer_last_common: self.in_size = lo_size_common

            # 出力の結合先を定義する(ResNetを表現したい)
            # 出力結合先の格納. add_modules と同期をとりたい
            self.outjoin_list.append(outjoin if type(outjoin) == type("") else "")
            
            # Layerの追加
            self.__AddModule(layer_type, layer_name, *param, **params)

        # 計算処理のコンパイル
        self.__Compile()
        self.logger.info("END")


    def __Compile(self):
        """
        出力のコンパイル(forwardでの計算処理をできるだけコンパクトにするため)
        基本的にnameにあるmoduleを実行する層と、その層に対して追加で行う層を追加していく
        """
        ## add_calcには関数を渡す. その関数を定義する
        def __preproc_common(output_list, output_list_index_i):
            output_list[1][0] = torch.cat(output_list[0], dim=1)
        def __preproc_output(output_list, output_list_index_i):
            for i, _ in enumerate(output_list[2]):
                output_list[2][i] = output_list[1][0]
        def __proc_rnn(output_list, output_list_index_i):
            # output, *hidden = self.__getattr__(name)(output) RNN層. [0]を参照するのはhidden層を弾くため
            output_list[output_list_index_i[0]][output_list_index_i[1]] = \
                    output_list[output_list_index_i[0]][output_list_index_i[1]][0][:, -1, :] #RNNでは各時間点での結果が返却されるので最新のものだけ採用する
        def __proc_resnet_in(output_list, output_list_index_i):
            output_list[output_list_index_i[0]][output_list_index_i[1]] = \
                    output_list[output_list_index_i[0]][output_list_index_i[1]] + output_list[1][1]
        def __proc_resnet_out(output_list, output_list_index_i):
            output_list[1][1] = output_list[output_list_index_i[0]][output_list_index_i[1]]

        ## self に計算用のlistを追加する
        ## list: [[input1, input2, ..., inputX],
        ##        [Common],
        ##        [Output1, Output2, ..., OutputX]]
        self.output_list = []
        self.output_list.append([]) # Input用
        for i in range(self.num_inputs):  self.output_list[-1].append(None)
        self.output_list.append([]) # Common用
        for i in range(2):                self.output_list[-1].append(None)
        self.output_list.append([]) # Output用
        for i in range(self.num_outputs): self.output_list[-1].append(None)

        ## output_listの参照用indexを格納するlistを作成する. 追加処理概要も追加する
        self.output_list_index    = []
        self.output_list_add_calc = []
        bool_from_input_to_common  = 0
        bool_from_common_to_output = 0
        i = 0
        for name, module in self.named_modules():
            self.output_list_index.append(None)
            ## index の格納
            if   name.find("Input") == 0:
                for j in range(self.num_inputs):
                    if name.find("Input"+str(j+1)) == 0:
                        self.output_list_index[-1] = (0, j)
            elif name.find("Common") == 0:
                bool_from_input_to_common += 1
                self.output_list_index[-1] = (1, 0)
            elif name.find("Output") == 0:
                bool_from_common_to_output += 1
                for j in range(self.num_outputs):
                    if name.find("Output"+str(j+1)) == 0:
                        self.output_list_index[-1] = (2, j)
            ## add_calc の格納
            self.output_list_add_calc.append([])
            if name != "" and (self.outjoin_list[i] == "01" or self.outjoin_list[i] == "11"):
                ## ResNet で前の出力を結合
                self.output_list_add_calc[-2].append(__proc_resnet_in)
            if bool_from_input_to_common == 1:
                ## Common層への突入に際してInput層の結果をまとめる(-2に追加していることに注意)
                self.output_list_add_calc[-2].append(__preproc_common)
            if bool_from_common_to_output == 1:
                ## Output層への突入に際してCommon層の結果をコピーする(-2に追加していることに注意)
                self.output_list_add_calc[-2].append(__preproc_output)
            if sum([(name.find(_x) >= 0) for _x in ["RNN","GRU","LSTM"]]) > 0:
                # RNN層の出力. 最終seqのみを抽出
                self.output_list_add_calc[-1].append(__proc_rnn)
            if name != "" and (self.outjoin_list[i] == "10" or self.outjoin_list[i] == "11"):
                ## ResNet で出力を保存
                self.output_list_add_calc[-1].append(__proc_resnet_out)

            if name != "": i += 1


    def __AddModule(self, layer_type: str, name: str, *param, **params):
        """
        add_module関数
        self に対して, self.in_size, self.out_size は外から指定しておく
        Params::
            layer_type: Input や Output
            name: layer の種類
        """
        self.logger.info("START")
        if   name.find("Linear") == 0:
            self.add_module(layer_type+"_No"+str(self.num_modules)+"_Linear", torch.nn.Linear(self.in_size, self.out_size, *param, **params))
            self.num_modules += 1
            self.in_size = self.out_size
        elif name.find("Identity") == 0:
            self.add_module(layer_type+"_No"+str(self.num_modules)+"_Identity", torch.nn.Identity(*param, **params))
            self.num_modules += 1
            self.in_size = self.out_size
        elif name.find("RNN") == 0:
            self.add_module(layer_type+"_No"+str(self.num_modules)+"_RNN", torch.nn.RNN(self.in_size, self.out_size, *param, **params))
            self.num_modules += 1
            if (params.get("bidirectional") is not None) and (params.get("bidirectional") == True):
                self.in_size = self.out_size * 2
            else:
                self.in_size = self.out_size
        elif name.find("LSTM") == 0:
            self.add_module(layer_type+"_No"+str(self.num_modules)+"_LSTM", torch.nn.LSTM(self.in_size, self.out_size, *param, **params))
            self.num_modules += 1
            if (params.get("bidirectional") is not None) and (params.get("bidirectional") == True):
                self.in_size = self.out_size * 2
            else:
                self.in_size = self.out_size
        elif name.find("GRU") == 0:
            self.add_module(layer_type+"_No"+str(self.num_modules)+"_GRU", torch.nn.GRU(self.in_size, self.out_size, *param, **params))
            self.num_modules += 1
            if (params.get("bidirectional") is not None) and (params.get("bidirectional") == True):
                self.in_size = self.out_size * 2
            else:
                self.in_size = self.out_size
        elif name.find("Dropout") == 0:
            self.add_module(layer_type+"_No"+str(self.num_modules)+"_Dropout", torch.nn.Dropout(*param, **params))
            self.num_modules += 1
        elif name.find("ReLU") == 0:
            self.add_module(layer_type+"_No"+str(self.num_modules)+"_ReLU", torch.nn.ReLU())
            self.num_modules += 1
        elif name.find("LeakyReLU") == 0:
            self.add_module(layer_type+"_No"+str(self.num_modules)+"_LeakyReLU", torch.nn.LeakyReLU(*param, **params))
            self.num_modules += 1
        elif name.find("Mish") == 0:
            self.add_module(layer_type+"_No"+str(self.num_modules)+"_Mish", self.Mish())
            self.num_modules += 1
        elif name.find("BatchNorm1d") == 0:
            ## バッチ正規化の補足
            ## バッチ正規化は、train()時のミニバッチ単位で学習し、
            ## 全体分布の平均と分散に近づくように momentum の値ずつ更新しているようである
            ## そのため、ミニバッチを1回訓練していない場合、train時はそのミニバッチ群で標準化し、
            ## eval()時は、momentumの値だけ少し更新した平均と分散でほんの少し標準化される感じ
            self.add_module(layer_type+"_No"+str(self.num_modules)+"_BatchNorm1d", torch.nn.BatchNorm1d(self.in_size, *param, **params))
            self.num_modules += 1
        elif name.find("Sigmoid") == 0:
            self.add_module(layer_type+"_No"+str(self.num_modules)+"_Sigmoid", torch.nn.Sigmoid())
            self.num_modules += 1
        elif name.find("Tanh") == 0:
            self.add_module(layer_type+"_No"+str(self.num_modules)+"_Tanh", torch.nn.Tanh())
            self.num_modules += 1
        elif name.find("Softmax") == 0:
            # dim=0で列、dim=1で行方向を確率変換
            self.add_module(layer_type+"_No"+str(self.num_modules)+"_Softmax", torch.nn.Softmax(dim=1))
            self.num_modules += 1
        elif name.find("LogSoftmax") == 0:
            self.add_module(layer_type+"_No"+str(self.num_modules)+"_LogSoftmax", torch.nn.LogSoftmax(dim=1))
            self.num_modules += 1
        else:
            self.logger.raise_error(f"unexpected value. name: {name}")


    # 重みの初期化
    def set_weight(self, weight):
        self.logger.info("START")
        # 重みの初期化
        for name, _ in self.named_modules():
            if name != "":
                try:
                    torch.nn.init.constant_(self.__getattr__(name).weight, weight)
                except AttributeError:
                    pass
                try:
                    torch.nn.init.constant_(self.__getattr__(name).bias, weight)
                except AttributeError:
                    pass
        self.logger.info("END")


    # 順伝播処理はforward関数に記載
    def forward(self, *inputs, hidden0=None):
        # Input分の初期値をoutput_listに格納する
        for i, _wk in enumerate(self.output_list[0]):
            self.output_list[0][i] = inputs[i]

        i = 0
        for name, _ in self.named_modules():
            # 各moduleの実行
            if self.output_list_index[i] is not None:
                self.output_list[self.output_list_index[i][0]][self.output_list_index[i][1]] = \
                        self.__getattr__(name)(self.output_list[self.output_list_index[i][0]][self.output_list_index[i][1]])
            # 追加処理の実行
            for _func in self.output_list_add_calc[i]:
                _func(self.output_list, self.output_list_index[i])

            i += 1

        return self.output_list[2]


# pytorch を使用した自作NNクラス
class MyTorch():

    def __init__(self, model: torch.nn.Module):
        self.logger = set_logger(_logname + ".MyTorch", log_level="info")
        self.logger.info("START")
        self.model           = model
        self.model_init      = copy.deepcopy(self.model)
        self.model_type_list = []  # 1:分類, 2:回帰.※損失関数の数だけ存在
        self.is_cuda         = False
        self.learning_rate   = None
        self.n_epoch         = None
        self.batch_size      = None
        self.criterion_list  = [] # 複数のOutputがある場合、複数の損失関数を考慮する(マルチタスク学習)
        self.optimizer       = None
        self.optimizer_init  = None
        self.classes_        = None
        self.logger.info("END")


    def __str__(self):
        return str(self.model)

    
    def set_parameters(self, learning_rate: float, n_epoch: int, batch_size: int, optimizer, criterion_list: List[str]):
        """
        torch の fit に関係のある parameter を set する
        learning_rate: learning rate
        n_epoch: エポック数
        batch_size: バッチサイズ
        optimizer: 勾配降下法
        criterion_list: 損失関数のリスト. 複数の場合はマルチタスクに対応する
        """
        self.logger.info("START")
        check_type(optimizer, [str, torch.optim.Optimizer])

        self.model_type_list = []
        self.criterion_list  = []
        self.criterion_list_init = criterion_list
        self.learning_rate   = learning_rate
        self.n_epoch         = n_epoch
        self.batch_size      = batch_size
        self.optimizer_init  = optimizer
        # 損失関数
        for criterion in criterion_list:
            if   (type(criterion) == str) and (criterion in ["CrossEntropyLoss","ce"]):
                self.criterion_list.append(torch.nn.CrossEntropyLoss(reduction='mean'))
                self.model_type_list.append(1)
            elif (type(criterion) == str) and (criterion in ["BCELoss","bce"]): 
                self.criterion_list.append(torch.nn.BCELoss(reduction='mean'))
                self.model_type_list.append(1)
            elif (type(criterion) == str) and (criterion in ["FocalLoss","fl"]): 
                self.criterion_list.append(FocalLoss(gamma=3))
                self.model_type_list.append(1)
            elif (type(criterion) == str) and (criterion in ["MSELoss","mse"]): 
                self.criterion_list.append(torch.nn.MSELoss(reduction='mean'))
                self.model_type_list.append(2)
            else:
                self.criterion_list.append(criterion)
                self.model_type_list.append(2)

        if self.is_cuda == True:
            for _i in range(len(self.criterion_list)):
                self.criterion_list[_i] = self.criterion_list[_i].to(torch.device("cuda:0"))

        # 勾配計算関数
        if   (type(optimizer) == str) and (optimizer in ["SGD","sgd"]):
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0)
        elif (type(optimizer) == str) and (optimizer in ["Adam","adam"]):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optimizer
        # GPUモードの場合
        if self.is_cuda == True:
            self.to_cuda()
        self.logger.info("END")


    # 各変数に対してcudaを使用するように変更
    # ※どこまでのオブジェクトにto()をする必要があるのか不明...
    def to_cuda(self):
        self.logger.info("START")
        # cudaへのtoは return で変数を上書きしなくても大丈夫
        # moduleがcudaにのっているかどうかは、next(model.parameters()).is_cuda で確認できる
        self.model.to(torch.device("cuda:0"))
        for _i in range(len(self.criterion_list)):
            self.criterion_list[_i] = self.criterion_list[_i].to(torch.device("cuda:0"))
        self.is_cuda = True
        self.logger.info("END")

    
    def fit(self, x, y, _x_test=None, _y_test=None, eval_autostop: str=None, eval_list: List[List[str]]=[["roc_auc","accuracy"]]):
        """
        model の fit (train)
        x: 訓練データ
        y: 訓練データ正解ラベル
        _x_test: テストデータ
        _y_test: テストデータ正解ラベル
        eval_autostop: 過学習を判断するために使う指標
        eval_list: 評価リスト. リストのリストになっているのはマルチタスクの場合を考慮してのこと
        """
        self.logger.info("START")
        check_type(x, [tuple, np.ndarray])
        check_type(y, [tuple, np.ndarray])
        check_type(_x_test, [type(None), tuple, np.ndarray])
        check_type(_y_test, [type(None), tuple, np.ndarray])

        # モデルを初期化する
        self.model = copy.deepcopy(self.model_init)
        ## この記述をしないと損失関数に上記で初期化したモデルのWeightが繋がらない
        self.set_parameters(self.learning_rate, self.n_epoch, self.batch_size, \
                            self.optimizer_init, self.criterion_list_init)

        # x_test にのみテコ入れをする.本来、lgbmなどでは1つが当たり前であり、
        # 入力が1つでも関数での受け取りをtupleにすると他のestimaterで使用できなくなる。
        # 入力が1つの場合はx_testはnumpyで入ってくるため、tupleでない場合はtuple化してやる
        if type(x) != tuple: x = (x, )
        if type(y) != tuple: y = (y, )
        x_test = _x_test
        if _x_test is not None and type(_x_test) != tuple: x_test = (_x_test, )
        y_test = _y_test # yも同様
        if _y_test is not None and type(_y_test) != tuple: y_test = (_y_test, )
        
        # 入力変数はnumpyを要求する
        for _x in x: check_type(_x, [np.ndarray])
        for _x in y: check_type(_x, [np.ndarray])
        if _x_test is not None:
            for _x in x_test: check_type(_x, [np.ndarray])
        if _y_test is not None:
            for _x in y_test: check_type(_x, [np.ndarray])

        # 訓練データの変換
        X, Y = self.__Conv(*x, y=y)

        # 検証データの変換
        is_test_data = False
        if (type(x_test) != type(None)) and (type(y_test) != type(None)):
            X_test, Y_test = self.__Conv(*x_test, y=y_test)
            is_test_data = True

        # クラス分類の場合は classes_ にクラスを記録する(初回現れた箇所)
        for i in range(len(self.model_type_list)):
            if self.model_type_list[i] == 1:
                self.classes_ = np.sort(np.unique(y[i]))
                break

        # input data
        # epoch loop
        self.eval_by_epoch = pd.DataFrame() # epoch毎の重みを保存する
        eval_step    = 10
        eval_n_epoch = 50
        score_best    = np.array([1])
        for i_epoch in range(self.n_epoch):
            # 学習データ数のランダム配列を定義(配列は同期がとれているはずなので、0インデックスを採用する)
            ndf = np.random.permutation(np.arange(X[0].shape[0]))

            # iteration
            for i_batch in range((X[0].shape[0] // self.batch_size)):
                
                ## ndf の 先頭からbatch_size分のインデックスを取り出して予測させる
                ndfwk = ndf[(i_batch * self.batch_size):((i_batch + 1) * self.batch_size)]

                # マルチタスク学習を考慮する(複数の損失関数)
                for i_task in range(len(self.criterion_list)):
                    # 勾配の初期化. model.zero_grad() の方が、損失関数が複数ある場合により安全に作用する
                    self.model.zero_grad()
                    self.optimizer.zero_grad() # 念のため. model.parameters()を渡しているので同じ結果なはずだが
                    for _i in range(len(self.criterion_list)): self.criterion_list[_i].zero_grad() #念のため

                    # Forward pass: compute predicted y by passing x to the model. Module objects
                    # override the __call__ operator so you can call them like functions. When
                    # doing so you pass a Tensor of input data to the Module and it produces
                    # a Tensor of output data.
                    # train()をしておく事でDropoutが機能する.
                    # BatchNormも、train()時はその時の分布で行い, eval()時はtrain()時のfitでtransformする
                    self.model.train()

                    # modelからは複数の返り値を考慮する
                    Y_pred = self.model(*[_X[ndfwk] for _X in X])
                    
                    # Compute and print loss. We pass Tensors containing the predicted and true
                    # values of y, and the loss function returns a Tensor containing the
                    # loss.
                    loss = self.criterion_list[i_task](Y_pred[i_task], Y[i_task][ndfwk])
                    
                    # Before the backward pass, use the optimizer object to zero all of the
                    # gradients for the variables it will update (which are the learnable
                    # weights of the model). This is because by default, gradients are
                    # accumulated in buffers( i.e, not overwritten) whenever .backward()
                    # is called. Checkout docs of torch.autograd.backward for more details.
                    #self.optimizer.zero_grad() # 先頭で行う
                    
                    # Backward pass: compute gradient of the loss with respect to model
                    # parameters
                    # backward()はスカラーに対してのみ作用する。ここでこのlossに使った全ての層の重みの勾配(grad)が格納される
                    loss.backward()
                    
                    # Calling the step function on an Optimizer makes an update to its
                    # parameters
                    # gradの情報を参照して、勾配降下法のアルゴリズムを加えて重みを更新する
                    self.optimizer.step()

            # Here we don't need to train, so the code is wrapped in torch.no_grad()
            with torch.no_grad():
                # 評価
                self.model.eval() # eval()をする事でDropoutがOFFになる(重みが有効)
                
                ## model の weight やらを途中保存
                sewk = pd.Series()
                sewk["i_epoch"] = i_epoch
                sewk["state"]   = self.model.state_dict().copy()

                ## 損失関数の情報を記録(訓練)
                Y_pred = self.model(*X)
                loss_list = []
                for i_task in range(len(self.criterion_list)):
                    loss = self.criterion_list[i_task](Y_pred[i_task], Y[i_task])
                    loss_list.append(loss.item())
                sewk["loss_value_train"] = tuple([round(_x, 5) for _x in loss_list])
                
                if is_test_data == True:
                    ## 損失関数の情報を記録(検証)
                    Y_pred_test = self.model(*X_test)
                    loss_list = []
                    for i_task in range(len(self.criterion_list)):
                        loss = self.criterion_list[i_task](Y_pred_test[i_task], Y_test[i_task])
                        loss_list.append(loss.item())
                    sewk["loss_value_test"] = tuple([round(_x, 5) for _x in loss_list])

                # 他評価を記録
                Y_pred_proba, Y_pred, Y_pred_proba_test, Y_pred_test = [], [], [], []
                Y_pred_proba = [_x.cpu().detach().numpy() for _x in self.predict_proba(X)]
                Y_pred       = [_x.cpu().detach().numpy() for _x in self.predict(      X)]
                if is_test_data == True:
                    Y_pred_proba_test = [_x.cpu().detach().numpy() for _x in self.predict_proba(X_test)]
                    Y_pred_test       = [_x.cpu().detach().numpy() for _x in self.predict(      X_test)]
                if eval_list is not None:
                    for itask, _eval_list in enumerate(eval_list):
                        for _x in _eval_list:
                            sewk[_x+"_"+str(itask)+"_train"] = evalate(_x, y[itask], Y_pred[itask], y_pred_proba=Y_pred_proba[itask]) # y, y_testはnumpy
                            if is_test_data == True:
                                sewk[_x+"_"+str(itask)+"_test"] = evalate(_x, y_test[itask], Y_pred_test[itask], y_pred_proba=Y_pred_proba_test[itask])
                self.logger.info(f'i_epoch:{i_epoch}, loss:{sewk["loss_value_train"]}')
                if is_test_data == True:
                    self.logger.info(f'i_epoch:{i_epoch}, loss_test:{sewk["loss_value_test"]}')

                # loss_value_*** の次から保存した値をループして表示する
                strwk = ""
                for _x in sewk.iloc[np.where(sewk.index.str.contains("loss_value_"))[0].max()+1:].index: strwk += (", " +_x+ ":" + sewk[_x])
                if strwk != "": self.logger.info(f"{strwk}")

                ## 評価値の保存
                self.eval_by_epoch = self.eval_by_epoch.append(sewk, ignore_index=True)

                # 自動終了の判定
                # ※eval_step epoch 毎に判定
                if (is_test_data == True) and (eval_autostop is not None):
                    if (i_epoch > eval_n_epoch*2) and ((i_epoch + 1) % eval_step == 0):
                        score_now = self.eval_by_epoch.iloc[-1*eval_n_epoch:               ][eval_autostop+"_test"].values
                        score_bef = self.eval_by_epoch.iloc[-2*eval_n_epoch:-1*eval_n_epoch][eval_autostop+"_test"].values
                        score_best = score_now if score_best.mean() > score_now.mean() else score_best
                        ## 結果の比較
                        self.logger.info(f"score now   : mean = {score_now.mean() } +/- {score_now.std() }")
                        self.logger.info(f"score before: mean = {score_bef.mean() } +/- {score_bef.std() }")
                        self.logger.info(f"score best  : mean = {score_best.mean()} +/- {score_best.std()}")
                        if score_now.mean() < score_bef.mean():
                            ## score が下がっている場合は学習中のため問題なし
                            pass
                        else:
                            ## score が下がっていない場合は誤差の範囲かどうかチェックする
                            ### t検定(非等分散と仮定)
                            t_test = stats.ttest_ind(score_now, score_bef, axis=0, equal_var=False, nan_policy='propagate')
                            self.logger.info(f"p value = {t_test.pvalue}, statistic(t value) = {t_test.statistic}")
                            if t_test.pvalue < 0.05:
                                # 統計的にscoreの差が有意であれば、学習をとめる
                                state_dict = self.eval_by_epoch.loc[(self.eval_by_epoch[eval_autostop+"_test"].idxmin())]["state"]
                                self.model.load_state_dict(state_dict.copy())
                                break
        # 重すぎるので削除する
        self.eval_by_epoch = self.eval_by_epoch.drop(columns=["state"])
        self.logger.info("END")
                

    # 予測確率
    def predict_proba(self, x):
        self.logger.info("START")
        bool_tensor = False
        if   (type(x) == tuple) and (str(type(x[0])).find("Tensor") < 0):
            bool_tensor = False
        elif (type(x) != tuple) and (str(type(x)).find("Tensor") < 0):
            bool_tensor = False
        else:
            bool_tensor = True
        y_pred_proba = None
        if bool_tensor == False:
            # Tensor変換されていない場合の変換
            X, _ = self.__Conv(*x if type(x)==tuple else tuple([x]))
            # 予測
            y_pred_proba = self.model(*X)
            y_pred_proba = [_y.cpu().detach().numpy() for _y in y_pred_proba]
            # numpyで入ってきた場合はnumpyで返す
        else:
            # Tensor変換されている場合
            y_pred_proba = self.model(*x if type(x)==tuple else tuple([x]))
        self.logger.info("END")
        return y_pred_proba


    # 予測
    def predict(self, x):
        self.logger.info("START")
        bool_tensor = False
        if   (type(x) == tuple) and (str(type(x[0])).find("Tensor") < 0):
            bool_tensor = False
        elif (type(x) != tuple) and (str(type(x)).find("Tensor") < 0):
            bool_tensor = False
        else:
            bool_tensor = True
        answer_list = []
        if bool_tensor == False:
            ndf = self.predict_proba(x)
            for i, model_type in enumerate(self.model_type_list):
                if model_type == 1:
                    if len(ndf[i].shape) == 1:
                        # 1次元配列の場合(閾値を0.5としてラベルを分ける)
                        answer_list.append((ndf > 0.5).astype(int))
                    elif len(ndf[i].shape) == 2:
                        # 2次元配列の場合
                        answer_list.append(np.argmax(ndf[i], axis=1))
                    else:
                        self.logger.raise_error(f"shape: {ndf[i].shape} is not supported.")
                else:
                    answer_list.append(ndf[i])
        else:
            tensor = self.predict_proba(x)
            for i, model_type in enumerate(self.model_type_list):
                if model_type == 1:
                    if len(tensor[i].shape) == 1:
                        # 1次元配列の場合(閾値を0.5としてラベルを分ける)
                        answer_list.append((tensor[i] > 0.5).int())
                    elif len(tensor[i].shape) == 2:
                        # 1次元配列の場合
                        answer_list.append(torch.max(tensor[i], dim=1)[1])
                    else:
                        self.logger.raise_error(f"shape: {tensor[i].shape} is not supported.")
                else:
                    answer_list.append(tensor[i])
        self.logger.info("END")
        return answer_list


    # 変換処理を関数化しておく
    def __Conv(self, *x, y=()):
        X, Y = [], []
        for _x in x:
            X.append(_x)

        # 入力データのTensor変換(型変換も)
        for i in range(len(X)):
            X[i] = torch.from_numpy(X[i])
            if self.is_cuda == True:
                X[i] = X[i].to(torch.device("cuda:0"))

        for i, _y in enumerate(y):
            Y.append(torch.from_numpy(_y))
            if self.is_cuda == True:
                Y[i] = Y[i].to(torch.device("cuda:0"))

        return tuple(X), tuple(Y)


# 損失関数 不均衡データに効く
class FocalLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, gamma=0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, input, target):
        # まず確率に変換しておく
        output = torch.functional.F.softmax(input, dim=1)
        # log(0)はinfになるので下限を設定する
        # output = output.clamp(self.eps, self.eps)
        # 正解ラベル以外は1から引く
        output = 1 - output
        output[[True]*(output.size()[0]), target] = 1 - output[[True]*(output.size()[0]), target]
        # loss の計算
        loss = -1 * torch.pow(1 - output, self.gamma) * torch.log(output)
        return loss[[True]*(output.size()[0]), target].mean()


# 活性化関数 Mish
class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(torch.nn.functional.softplus(x)))


