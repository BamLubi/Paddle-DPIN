# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
from net import DPINLayer


class DygraphModel():
    # define model
    def create_model(self, config):
        K = config.get("hyper_parameters.K", 3)
        emb_dim = config.get("hyper_parameters.embedding_dim", 8)
        is_sparse = config.get("hyper_parameters.is_sparse", True)
        max_item = config.get("hyper_parameters.max_item", 4975)
        max_context = config.get("hyper_parameters.max_context", 1933)
        d_model = config.get("hyper_parameters.d_model", 64)
        h = config.get("hyper_parameters.h", 2)
        
        model = DPINLayer(K, emb_dim, max_item, max_context, d_model, h, is_sparse)
        return model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch, config):
        hist_item = batch[0]
        hist_cat = batch[1]
        target_item = batch[2]
        target_cat = batch[3]
        target_pos = batch[4]
        position = batch[5]
        label = batch[6]
        return hist_item, hist_cat, target_item, target_cat, target_pos, position, label

    # define loss function by predicts and label
    def create_loss(self, pred, label):
        loss = nn.functional.binary_cross_entropy(pred, label)
        return loss

    # define optimizer
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.05)
        optimizer = paddle.optimizer.SGD(
            learning_rate=lr,
            parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    def create_metrics(self):
        metrics_list_name = ["AUC"]
        auc_metric = paddle.metric.Auc()
        metrics_list = [auc_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        hist_item, hist_cat, target_item, target_cat, target_pos, position, label = self.create_feeds(batch_data, config)

        output = dy_model.forward(hist_item, hist_cat, target_item, target_cat, position)
        output = paddle.squeeze(output)
        target_pos = paddle.unsqueeze(target_pos-1, axis=1)
        line = paddle.arange(config.get("runner.train_batch_size", 32))
        line = paddle.unsqueeze(line, axis=1)
        target_pos = paddle.concat([line, target_pos], axis=1)
        output = paddle.gather_nd(output, target_pos)
        ## loss
        loss = self.create_loss(output, label)
        ## AUC
        output = paddle.unsqueeze(output, axis=1)
        output = paddle.concat([1-output, output], axis=1)
        label = paddle.unsqueeze(label, axis=1)
        metrics_list[0].update(preds=output, labels=label)

        print_dict = {'loss': loss}
        return loss, metrics_list, print_dict

    # construct infer forward phase
    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        hist_item, hist_cat, target_item, target_cat, target_pos, position, label = self.create_feeds(batch_data, config)

        output = dy_model.forward(hist_item, hist_cat, target_item, target_cat, position)
        output = paddle.squeeze(output)
        target_pos = paddle.unsqueeze(target_pos-1, axis=1)
        line = paddle.arange(config.get("runner.train_batch_size", 32))
        line = paddle.unsqueeze(line, axis=1)
        target_pos = paddle.concat([line, target_pos], axis=1)
        output = paddle.gather_nd(output, target_pos)
        ## AUC
        output = paddle.unsqueeze(output, axis=1)
        output = paddle.concat([1-output, output], axis=1)
        label = paddle.unsqueeze(label, axis=1)
        metrics_list[0].update(preds=output, labels=label)
        return metrics_list, None
