"""
-------------------------------------------------
   File Name：     check_point
   Description :
   Author :       weiwanshun
   date：          2018/9/10
-------------------------------------------------
   Change Activity:
                   2018/9/10:
-------------------------------------------------
"""
from keras.callbacks import Callback
import numpy as np
import warnings

__author__ = 'weiwanshun'


class ModelCheckpoint(Callback):
    """
        example :

        check_point = ModelCheckpoint(filepath='epoch{epoch:02d}_iou{iou:.2f}_valloss{val_loss:.2f}.hdf5',
                                  monitor='iou', verbose=1, weight_window=0.2, save_best_only=True,
                                  save_weights_only=False,
                                  mode='max', period=1)
    """

    def __init__(self, filepath, monitor='val_loss', weight_window=0.2, verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        """

        :param filepath:模型路径
        :param monitor:监控变量
        :param weight_window:权重窗口，仅在归一化的输出下使用，用于截取权重的前 weight_window 的类别
        :param verbose: 是否打印
        :param save_best_only: 是否只保存最佳模型
        :param save_weights_only: 是否只保存权重值不保存网络结构
        :param mode: monitor 选用iou 时 mode 需要选择为 max
        :param period: 几个epoch 保存一次
        """
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.weight_window = weight_window
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def original_on_epoch_end(self, epoch, logs=None):

        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, val_loss=logs["val_loss"])
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

    def on_epoch_end(self, epoch, logs=None):

        if self.monitor == "iou":

            val_x = self.validation_data[0]
            label = self.validation_data[1]

            predict_label = self.model.predict(val_x)
            predict_label_final = self._get_top_weight(weight_window=self.weight_window, predict_label=predict_label)
            iou = self._iou(label, predict_label_final)
            mean_iou = np.mean(iou)

            self.epochs_since_last_save += 1
            if self.epochs_since_last_save >= self.period:
                self.epochs_since_last_save = 0
                filepath = self.filepath.format(epoch=epoch + 1, val_loss=logs["val_loss"], iou=mean_iou)
                if self.save_best_only:
                    current = mean_iou
                    if current is None:
                        warnings.warn('Can save best model only with %s available, '
                                      'skipping.' % (self.monitor), RuntimeWarning)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s'
                                      % (epoch + 1, self.monitor, self.best,
                                         current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' %
                                      (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)



        else:
            self.original_on_epoch_end(self, epoch, logs)

    def _get_top_weight(self, weight_window, predict_label):
        predict_label_final = []
        for item in predict_label:
            weight_dic = {}
            for i, weight in enumerate(item):
                weight_dic[i] = weight
            weight_dic = sorted(weight_dic.items(), key=lambda x: x[1], reverse=True)

            weight_sum = 0.0
            top_weight = {}
            for item_weight in weight_dic:
                weight_sum += item_weight[1]
                if weight_sum >= weight_window:
                    break
                top_weight[item_weight[0]] = item_weight[1]

            arr = np.zeros_like(item)
            for idx in top_weight.keys():
                arr[idx] = top_weight[idx]

            predict_label_final.append(arr)
        return np.array(predict_label_final)

    def _iou(self, label, predict_label):
        iou = []
        for i, l in enumerate(label):
            p_l = predict_label[i]

            union_sum = 0
            interaction_sum = 0
            for j in range(len(l)):

                if l[j] > 0 and p_l[j] > 0:
                    union_sum += 1

                    interaction_sum += 1
                    continue
                elif l[j] > 0 or p_l[j] > 0:
                    union_sum += 1

            iou.append(interaction_sum / union_sum)

        return np.array(iou)
