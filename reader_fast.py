import os
import math
import random
import functools
import numpy as np
import logging
import datareader
from datareader.tool.imgtransformer import reader_transformer
from datareader.tool import pytransformer
from datareader.datasets import dataset
from Queue import Queue
from threading import Thread
import time

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
def parse_kv(r):
    """ parse kv data from sequence file for imagenet
    """
    import cPickle
    k, v = r
    obj = cPickle.loads(v)
    #maybe we need to normalize this data to be in one schema
    #the schema in label is:
    #   [<line_num>, <contsign>, <file_name>, <cate_id>, <cate_name>, <cate_desc>]
    if len(obj['label']) >= 4:
        label = int(obj['label'][3])  # class id
    else:
        label = int(obj['label'][2])
    image = obj['image']  #binary jpeg
    return image, label

def create_imagenet_local_rawdatareader(which,
                                      datasetname,
                                      shuffle_size=10000,
                                      cachefolder=None):
    datareader.set_loglevel(logging.DEBUG)
    if which != "train":
        raise ValueError("local seq file reader only support train data for now")
    uri = "file://mnt/seqdata"
    cache_to = os.path.join(cachefolder, which) if cachefolder else None
    ds = dataset.Dataset()
    
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    port = os.getenv("PADDLE_PSERVER_PORT")
    worker_ips = os.getenv("PADDLE_TRAINER_IPS")
    worker_endpoints = []
    for ip in worker_ips.split(","):
        worker_endpoints.append(':'.join([ip, port]))
    trainer_num = len(worker_endpoints)

    #shuffle train
    ds.load(uri, cache_to, part_id=trainer_id, part_num=trainer_num)
    if shuffle_size > 0:
        ds.buffered(shuffle_size).shuffle(shuffle_size)

    return ds.map(parse_kv).reader()


def create_imagenet_local_datareader(which,
                                     datadir,
                                     train_listfile,
                                     test_listfile,
                                     maxloadnum=-1):
    shuffle = True
    if which == 'train' or which == 'val':
        file_list = train_listfile
    else:
        file_list = test_listfile
    with open(file_list) as flist:
        lines = [line.strip() for line in flist]
    logging.debug('after load imagenet label list')
    def reader():
        if shuffle:
            random.shuffle(lines)
        count = 0
        for line in lines:
            if which == 'train' or which == 'val':
                img_path, label = line.split()
                img_path = os.path.join(datadir, which, img_path)
                imagedata = open(img_path, 'rb').read()
                yield imagedata, int(label)
            elif which == 'test':
                img_path = os.path.join(datadir, line)
                imagedata = open(img_path, 'rb').read()
                yield [imagedata]
            if maxloadnum > 0:
                count += 1
                if count > maxloadnum:
                    break
    return reader
def create_imagenet100_local_datareader(which):
    maxloadnum = -1
    DATA_DIR = '.'
    TRAIN_LIST = 'dataset_100/train.list'
    TEST_LIST = 'dataset_100/val.list'
    return create_imagenet_local_datareader(which, DATA_DIR, TRAIN_LIST,
                                            TEST_LIST, maxloadnum)

def transform_reader(which,
                     reader,
                     color_jitter=False,
                     rotate=False,
                     concurrency=16,
                     crop_size=224,
                     process_tensor=False):
    queue_limit = 20000
    builder = pytransformer.Builder(
        thread_num=concurrency, queue_limit=queue_limit)
    builder._conf['swapaxis'] = 1
    builder.decode(mode='RGB')
    if which == 'train':
        if rotate:
            builder.rotate(random_range=10)
        if crop_size > 0:
            builder.random_crop(size=crop_size)
        builder.flip('FLIP_LEFT_RIGHT', random=True)
    else:
        if crop_size > 0:
            builder.resize_short(crop_size, interpolation='INTER_LANCZOS4')
            builder.center_crop(crop_size, center=True)

    def _mapper(r):
        ret = list(r)
        img = ret[0]
        img = img.astype('float32') / 255
        img -= img_mean
        img /= img_std
        #img = img.transpose((2, 0, 1))
        ret[0] = img
        ret = tuple(ret)
        return ret

    if process_tensor:
        newreader = reader_transformer(
            reader,
            builder,
            queue_limit,
            with_label=(which != 'test'),
            map_func=_mapper)
    else:
        newreader = reader_transformer(
            reader,
            builder,
            queue_limit,
            with_label=(which != 'test'),
            map_func=None)
    return newreader


def batch_feeder(batch_reader, pin_memory=False):
    # batch((sample, label)) => batch(sample), batch(label)
    def _feeder():
        for batch_data in batch_reader():
            sample_batch = []
            label_batch = []
            for sample, label in batch_data:
                sample_batch.append(sample)
                label_batch.append([label])
            tensor = core.LoDTensor()
            label = core.LoDTensor()
            place = core.CUDAPinnedPlace() if pin_memory else core.CPUPlace()
            tensor.set(np.array(sample_batch, dtype="float32", copy=False), place)
            label.set(np.array(label_batch, dtype="int64", copy=False), place)
            yield [tensor, label]

    return _feeder



# handler for fetch data from 'transformer' and put it to 'out_queue'
def create_threaded_reader(reader, queuesize=20000):
    def fetch_worker(reader, out_queue):
        for data in reader():
            out_queue.put(data)
        out_queue.put(None)
    def _reader():
        logging.debug('start threaded_reader')
        out_queue = Queue(queuesize)
        fetcher = Thread(target=fetch_worker, args=(reader, out_queue))
        fetcher.daemon = True
        fetcher.start()
        fetch_counter = 0
        while True:
            sample = out_queue.get()
            #fetch_counter += 1
            #if fetch_counter > 512:
            #    print("queue size: ", out_queue.qsize())
            #    fetch_counter = 0
            if sample is None:
                break
            yield sample
        logging.debug('quit fetcher thread')
        fetcher.join()
    return _reader

if __name__ == '__main__':
    reader = create_imagenet_local_rawdatareader("train", "imagenet")
    reader = transform_reader("train", reader)
    count = 0
    start_time = time.time()
    last_1000 = time.time()
    for d in reader():
        count += 1
        if count % 1000 == 0:
            print("current total avg speed %f" % count / (time.time() - start_time))
            print("1000 sample speed %s" % 1000.0 / (time.time() - last_1000))
            last_1000 = time.time()

