
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'cycle_stack_v1final_gan':
        from .cycle_stack_v1final_model import CycleStackv1FinalModel
        model = CycleStackv1FinalModel()
    elif opt.model == 'cycle_stack_v2final_gan':
        from .cycle_stack_v2final_model import CycleStackv2FinalModel
        model = CycleStackv2FinalModel()
    elif opt.model == 'cycle_multid_gan':
        from .cycle_multid_model import CycleMultiDModel
        model = CycleMultiDModel()
    elif opt.model == 'cycle_gan_attr':
        from .cycle_attr_gan_model import CycleAttrGANModel
        model = CycleAttrGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    elif opt.model == 'AE_gan':
	assert(opt.dataset_mode == 'triple')
	from .ae_gan_model import AEGANModel
	model = AEGANModel()
    elif opt.model == 'pix2pix_ae':
        assert(opt.dataset_mode == 'triple')
        from .pix2pix import Pix2pix_ae_model 
        model = Pix2pix_ae_model()
    elif opt.model == 'sr_base':
        assert(opt.dataset_mode == 'triple')
        from .sr_base import Sr_base_model
        model = Sr_base_model()
    elif opt.model == 'AE_gan2':
        assert(opt.dataset_mode == 'triple')
        from .ae_gan_model2 import AEGANModel
        model = AEGANModel()

    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
