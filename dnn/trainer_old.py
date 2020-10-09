import torch
import torch.optim as optim
import numpy as np
import time
import progressbar
import logging

from .data import data_feeder

class trainer :
    def __init__( self, cfg, model, device, trainset, testset=None ):
        self._cfg = cfg
        self._device = device
        self._optimizer = None
        self._model = model

        self._trainset = trainset
        self._testset = testset
        self._epoch = 0
        #self._trainset_feeder = data_feeder( trainset )

        #if testset is not None :
        #    self._testset_feeder = data_feeder( testset )

        if trainset is not None:
            self._set_optimizer()

    #def _set_optimizer( self ):
    #    params = self._cfg.dnn.OPTIMIZER
    #    self._optimizer = optim.SGD( self._model['net'].parameters(),
    #                                 lr=params['lr'],
    #                                 momentum=params.get('momentum',0.9),
    #                                 weight_decay=params.get('weight_decay',1e-4))

    #    self._scheduler = optim.lr_scheduler.StepLR( self._optimizer,
    #                                                step_size=params['decay_step'],
    #                                                gamma=params['decay_rate'] )

    #    self._niter = self._cfg.dnn.NITER

    def _set_optimizer( self ):
        params = self._cfg.dnn.OPTIMIZER
        if params['type'] == 'GD':
            self._optimizer = optim.SGD( self._model.parameters(),
                                        lr=params['lr'],
                                        momentum=params.get('momentum',0.9),
                                        weight_decay=params.get('weight_decay',0))
        elif params['type'] == 'Adam':
            self._optimizer = optim.Adam(self._model.parameters(),
                                         lr = params['lr'],
                                         betas=params.get('betas',(0.9, 0.999)),
                                         eps = params.get('eps', 1e-8)
                                         )
        else:
            raise ValueError('Optimiser type wrong, {} is not a valid optimizer type!')

        self._scheduler = optim.lr_scheduler.MultiStepLR( self._optimizer,
                                                    milestones=params['decay_steps'],
                                                    gamma=params['decay_rate'] )
        self._niter = self._cfg.dnn.NITER

    def _set_device( self, blobs ):
        for n,d in blobs.items() :
            blobs[n] = d.to( self._device )
        return blobs

    def trainstep( self ):
        inputs, targets = self._trainset.next()
        inputs = self._set_device( inputs )
        targets = self._set_device( targets )

        outputs = self._model['net']( inputs )
        loss = self._model['loss']( outputs, targets )

        print( loss )
        self._optimizer.zero_grad()
        #loss.backward()
        #self._optimizer.step()

        return loss

    def test( self ):
        self._model['net'].test()
        images = self._batch_gen.next()
        data, targets = self._blobs_gen.get_blobs( images )
        return self._model( data )

    def _train( self ):
        self._model['net'].train()

        widgets = [ progressbar.Percentage(), ' ', progressbar.ETA(), ' ',
                    '(',progressbar.DynamicMessage('loss'),')' ]
        bar = progressbar.ProgressBar(widgets=widgets,max_value=len(self._trainset)).start()

        loss_values = []

        for idx in range( len(self._trainset) ):
            inputs, targets = self._trainset.next()

            inputs = self._set_device( inputs )
            targets = self._set_device( targets )

            nets, outputs = self._model['net']( inputs, targets )
            loss = self._model['loss']( inputs, nets, outputs, targets )

            loss_values.append( loss.cpu().detach().numpy() )

            # Computing gradient and do SGD step
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            bar.update(idx+1,loss=loss.item())
        bar.finish()

        print('Average loss : ', np.mean(loss_values))

    def _validate( self ):
        self._model['net'].eval()

        bench = self._testset.benchmark()

        with torch.no_grad() :
            for idx in range( len(self._testset) ):
                inputs, targets = self._testset.next()
                inputs = self._set_device( inputs )
                output = self._model['net']( inputs )
                bench.update( targets, output )

        bench.summary()

    def train( self ):
        if self._testset is not None :
            self._validate()

        for i in range( self._epoch, self._niter ):
            print('epoch {} / {}'.format(i+1, self._niter))
            self._train()

            if self._testset is not None :
                self._validate()
            self._epoch = i
            self._scheduler.step()

    def load_trained_model(self, model_path):
        state_dict = torch.load(model_path)['state_dict']
        self._model['net'].load_state_dict(state_dict)

    def save_training(self, path, to_print=True):
        torch.save({
            'epoch': self._epoch,
            'model_state_dict': self._model['net'].state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict()
        }, path)
        if to_print:
            print('The checkpoint has been saved to {}'.format(path))

    def load_training(self, path, to_print=True):
        checkpoint = torch.load(path)
        self._epoch = checkpoint['epoch']
        self._model['net'] = checkpoint['model_state_dict']
        self._optimizer = checkpoint['optimizer_state_dict']
        if to_print:
            print('Chekpoint has been loaded from {}'.format(path))
