import os, shutil, glob, subprocess
import time
from options.train_options import TrainOptions
from utils.visualizer import Visualizer
from data.data_loader import create_dataloader
from models.models import create_model
from utils import util
from tensorboardX import SummaryWriter

#<editor-fold desc="Setup">
# parse training options
opt = TrainOptions().parse()

# if debug flag was set, run debugger
if opt.debug:
    import ipdb; ipdb.set_trace()

# create dataset
data_loader, dataset, dataset_size = create_dataloader(opt)
# create model
model = create_model(opt)

# tensorboard + web dump
visualizer = Visualizer(opt)
## </editor-fold>

# Train
total_steps = opt.epoch_count * dataset_size
total_iters = 0
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_iters += 1
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize


        model.tick()
        data_loader.tick()
        model.set_input(data)
        model.optimize_parameters()
        errors = model.get_current_errors()


        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch, total_steps)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            lrs = model.get_current_lr()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t, opt.name)
            visualizer.plot_current_errors(epoch, total_steps, errors, 'loss')
            visualizer.plot_current_errors(epoch, total_steps, lrs, 'lr')

        if opt.eval and total_steps % opt.eval_freq == 0:
            eval_start_time = time.time()
            mse = model.eval_network()
            visualizer.plot_current_errors(epoch, total_steps, mse, 'mse')
            t = (time.time() - eval_start_time)
            print('eval time ', t)
            
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')
            # net_params = model.get_network_params()
            # visualizer.plot_params_hist(net_params, total_steps)
            # net_grads = model.get_network_grads()
            # visualizer.plot_params_hist(net_grads, total_steps)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
