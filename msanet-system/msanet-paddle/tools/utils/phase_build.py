from tools.callbacks import ModelEval
from tools.utils.eval import EvalModule
from tools.utils.infer import InferModule
from tools.utils.train import TrainModule


class PhaseBuilder(object):
    def __init__(self, cfg, model_bulider):
        self.phase = cfg.args.phase
        if self.phase == 'train':
            self.module = TrainModule(cfg=model_bulider['config'],
                                      train_dataset=model_bulider['train_dataset'],
                                      val_dataset=model_bulider['val_dataset'],
                                      model=model_bulider['model'],
                                      optim=model_bulider['optim'],
                                      loss_fn=model_bulider['loss_fn'],
                                      metrics=model_bulider['metrics'],
                                      callbacks=model_bulider['callbacks'])
        elif self.phase == 'eval':
            eval_callback = ModelEval(log_dir=model_bulider['config'].log_dir + "/epoch_eval_log.txt",
                                      save_dir=model_bulider['config'].save_dir, cfg=model_bulider['config'])
            callbacks = [eval_callback]
            self.module = EvalModule(cfg=model_bulider['config'],
                                     test_dataset=model_bulider['test_dataset'],
                                     model=model_bulider['model'],
                                     optim=model_bulider['optim'],
                                     loss_fn=model_bulider['loss_fn'],
                                     metrics=model_bulider['metrics'],
                                     callbacks=callbacks)

        elif self.phase == 'infer':
            self.module = InferModule(cfg=model_bulider['config'],
                                      test_dataset=model_bulider['test_dataset'],
                                      model=model_bulider['model'],
                                      optim=model_bulider['optim'],
                                      loss_fn=model_bulider['loss_fn'],
                                      metrics=model_bulider['metrics'],
                                      decoder=model_bulider['decoder'])

        else:
            raise Exception(print('Phase no accept!'))

    def execute(self):
        if self.phase == 'train':
            self.module.train_network()
        elif self.phase == 'eval':
            self.module.eval_network()
        elif self.phase == 'infer':
            self.module.inference()
            self.module.get_fps()
        else:
            raise Exception(print('Phase no accept!'))
