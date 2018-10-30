#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import time
import os
import math

import cProfile, pstats, StringIO

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.profiler as profiler
from imagenet_reader import train, val
import reader_fast

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "weight_decay": 1e-4,
        "batch_size": 32,  # per GPU
        "epochs": [30, 60, 80],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}

WEIGHT_DECAY = train_parameters["learning_strategy"]["weight_decay"]
BN_NO_DECAY = bool(os.getenv("BN_NO_DECAY", "1"))

def lr_warmup(learning_rate, warmup_steps, start_lr, end_lr):
    # increase lr in warmup_steps
    # lr = lr + (warmup_rate * step / warmup_steps)
    assert(isinstance(end_lr, float))
    assert(isinstance(start_lr, float))
    linear_step = end_lr - start_lr
    with fluid.default_main_program()._lr_schedule_guard():
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="learning_rate_warmup")

        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter()

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(global_step < warmup_steps):
                decayed_lr = start_lr + linear_step * (global_step / warmup_steps)
                fluid.layers.tensor.assign(decayed_lr, lr)
            with switch.default():
                fluid.layers.tensor.assign(learning_rate, lr)

        return lr


class ResNet():
    def __init__(self, layers=50, is_train=True):
        self.params = train_parameters
        self.layers = layers
        self.is_train = is_train

    def net(self, input, class_dim=1000):
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(
            input=input, num_filters=64, filter_size=7, stride=2, act='relu')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1)

        pool = fluid.layers.pool2d(
            input=conv, pool_size=7, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(input=pool,
                              size=class_dim,
                              act='softmax',
                              param_attr=fluid.ParamAttr(
                                  initializer=fluid.initializer.Uniform(-stdv,
                                                                        stdv),
                                  regularizer=fluid.regularizer.L2Decay(WEIGHT_DECAY)),
                              bias_attr=fluid.ParamAttr(
                                  regularizer=fluid.regularizer.L2Decay(WEIGHT_DECAY))
                              )
        return out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      bn_init_value=1.0):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False,
            param_attr=fluid.ParamAttr(regularizer=fluid.regularizer.L2Decay(WEIGHT_DECAY)))
        return fluid.layers.batch_norm(
                input=conv, act=act, is_test=not self.is_train,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(bn_init_value),
                    regularizer=None if BN_NO_DECAY else fluid.regularizer.L2Decay(WEIGHT_DECAY)))

    def shortcut(self, input, ch_out, stride):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(input, ch_out, 1, stride)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride):
        conv0 = self.conv_bn_layer(
            input=input, num_filters=num_filters, filter_size=1, act='relu')
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        # NOTE: default bias is 0.0 already
        conv2 = self.conv_bn_layer(
            input=conv1, num_filters=num_filters * 4, filter_size=1, act=None, bn_init_value=0.0)

        short = self.shortcut(input, num_filters * 4, stride)

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')


def _model_reader_dshape_classdim(args, is_train):
    model = None
    reader = None
    if args.data_set == "flowers":
        class_dim = 102
        if args.data_format == 'NCHW':
            dshape = [3, 224, 224]
        else:
            dshape = [224, 224, 3]
        if is_train:
            reader = paddle.dataset.flowers.train()
        else:
            reader = paddle.dataset.flowers.test()
    elif args.data_set == "imagenet":
        class_dim = 1000
        if args.data_format == 'NCHW':
            dshape = [3, 224, 224]
        else:
            dshape = [224, 224, 3]
        if not args.data_path:
            raise Exception(
                "Must specify --data_path when training with imagenet")
        if not args.use_reader_op:
            if is_train:
                reader = reader_fast.create_imagenet_local_rawdatareader("train", "imagenet")
                reader = reader_fast.transform_reader("train", reader, process_tensor=not args.use_reader_uint8)
                # reader = reader_fast.create_threaded_reader(reader)
            else:
                reader = val()
        else:
            if is_train:
                #reader = train()
                reader = reader_fast.create_imagenet_local_rawdatareader("train", "imagenet")
                reader = reader_fast.transform_reader("train", reader, process_tensor=not args.use_reader_uint8)
                # reader = reader_fast.create_threaded_reader(reader)
            else:
                reader = val()
    return reader, dshape, class_dim


def get_model(args, is_train, main_prog, startup_prog):
    reader, dshape, class_dim = _model_reader_dshape_classdim(args, is_train)

    pyreader = None
    trainer_count = int(os.getenv("PADDLE_TRAINERS"))
    if args.use_reader_uint8:
        reader_dtype = "uint8"
    else:
        reader_dtype = "float32"

    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            if args.use_reader_op:
                pyreader = fluid.layers.py_reader(
                    capacity=args.batch_size * args.gpus,
                    shapes=([-1] + dshape, (-1, 1)),
                    dtypes=(reader_dtype, 'int64'),
                    name="train_reader" if is_train else "test_reader",
                    use_double_buffer=True)
                input, label = fluid.layers.read_file(pyreader)
            else:
                input = fluid.layers.data(
                    name='data', shape=dshape, dtype='float32')
                label = fluid.layers.data(
                    name='label', shape=[1], dtype='int64')

            model = ResNet(is_train=is_train)
            if args.use_reader_uint8:
                cast = fluid.layers.cast(input, "float32")
                # setup values when run startup
                img_mean = fluid.layers.create_global_var([3, 1, 1], 0.0, "float32", name="img_mean", persistable=True)
                img_std = fluid.layers.create_global_var([3, 1, 1], 0.0, "float32", name="img_std", persistable=True)

                t1 = fluid.layers.elementwise_sub(cast / 255.0, img_mean, axis=1)
                t2 = fluid.layers.elementwise_div(t1, img_std, axis=1)
            else:
                t2 = input
            
            predict = model.net(t2, class_dim=class_dim)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(x=cost)

            batch_acc1 = fluid.layers.accuracy(input=predict, label=label, k=1)
            batch_acc5 = fluid.layers.accuracy(input=predict, label=label, k=5)

            # configure optimize
            optimizer = None
            if is_train:
                start_lr = args.learning_rate
                # n * worker * repeat
                end_lr = args.learning_rate * trainer_count * args.multi_batch_repeat
                total_images = 1281167 / trainer_count
                step = int(total_images / (args.batch_size * args.gpus * args.multi_batch_repeat) + 1)
                warmup_steps = step * 5  # warmup 5 passes
                epochs = [30, 60, 80]
                bd = [step * e for e in epochs]
                base_lr = end_lr
                lr = []
                lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
                if args.use_lars:
                    optimizer = fluid.optimizer.LarsMomentum(
                        learning_rate=lr_warmup(fluid.layers.piecewise_decay(
                                boundaries=bd, values=lr),
                            warmup_steps, start_lr, end_lr),
                        momentum=0.9,
                        lars_weight_decay=1e-4)
                        # regularization=fluid.regularizer.L2Decay(1e-4))
                else:
                    optimizer = fluid.optimizer.Momentum(
                        learning_rate=lr_warmup(fluid.layers.piecewise_decay(
                                boundaries=bd, values=lr),
                            warmup_steps, start_lr, end_lr),
                        momentum=0.9,
                        regularization=fluid.regularizer.L2Decay(1e-4))
                optimizer.minimize(avg_cost)


    # config readers
    if not args.use_reader_op:
        batched_reader = paddle.batch(
            reader if args.no_random else paddle.reader.shuffle(
                reader, buf_size=5120),
            batch_size=args.batch_size * args.gpus,
            drop_last=True)
    else:
        batched_reader = None
        # pyreader.decorate_paddle_reader(
        #     paddle.batch(
        #         reader if args.no_random else paddle.reader.shuffle(
        #             reader, buf_size=5120),
        #         batch_size=args.batch_size))

        pyreader.decorate_tensor_provider(
            reader_fast.batch_feeder(
                paddle.batch(reader, batch_size=args.batch_size),
                pin_memory=args.pin_memory,
                img_dtype=reader_dtype
            )
        )

    return avg_cost, optimizer, [batch_acc1,
                                 batch_acc5], reader, pyreader
