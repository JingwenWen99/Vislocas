from models.cct import *


def getClassifier(cfg, model_name="Vislocas", pretrain=False, SAE=None, adj_file=None):
    if model_name == "Vislocas_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified72(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)

    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified68_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified68(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified69_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified69(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified70_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified70(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified71_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified71(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified72_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified72(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified73_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified73(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified74_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified74(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)

    elif model_name == "cct_modified56_focalloss_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_weightedbce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_bce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)

    elif model_name == "cct_modified72_focalloss_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified72(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified72_weightedbce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified72(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified72_bce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified72(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)

    elif model_name == "cct_modified75_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified75(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified76_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified76(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified77_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified77(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified78_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified78(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)


    else:
        model = cct_modified72(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)

    return model