from models.cct import *


def getClassifier(cfg, model_name="lightViT_modified_480", pretrain=False, SAE=None, adj_file=None):
    if model_name == "cct_modified":
        model = cct_modified(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=160, n_input_channels=192)
    elif model_name == "cct_modified_lr4e-5_weight-decay1e-2":
        model = cct_modified(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=160, n_input_channels=192)
    elif model_name == "cct_modified_lr5e-5_weight-decay1e-1_1":
        model = cct_modified(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=160, n_input_channels=192)
    elif model_name == "cct_modified_lr5e-5_weight-decay1e-1_2":
        model = cct_modified_(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=160, n_input_channels=192)
    elif model_name == "cct_modified2":
        model = cct_modified2(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "cct_modified3":
        model = cct_modified3(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "cct_modified4":
        model = cct_modified4(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "cct_modified5":
        model = cct_modified5(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "cct_modified6":
        model = cct_modified6(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "cct_modified7":
        model = cct_modified7(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "cct_modified8":
        model = cct_modified8(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "cct_modified9":
        model = cct_modified9(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "cct_modified10":
        model = cct_modified10(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "cct_modified11":
        model = cct_modified11(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "cct_modified12":
        model = cct_modified12(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "cct_modified13":
        model = cct_modified13(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "_cct_modified13":
        model = cct_modified13(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "cct_modified14":
        model = cct_modified14(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "cct_modified14_":
        model = cct_modified14(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "cct_modified15":
        model = cct_modified15(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "cct_modified16":
        model = cct_modified16(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=3000, n_input_channels=3)
    elif model_name == "cct_modified17":
        model = cct_modified17(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3)
    elif model_name == "cct_modified17_":
        model = cct_modified17_(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3)
    elif model_name == "cct_modified18":
        model = cct_modified18(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3)
    elif model_name == "cct_modified19":
        model = cct_modified19(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE)
    elif model_name == "cct_modified20":
        model = cct_modified20(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE)
    elif model_name == "cct_modified21":
        model = cct_modified21(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE)
    elif model_name == "cct_modified22":
        model = cct_modified22(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE)
    elif model_name == "cct_modified22_":
        model = cct_modified22(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE)
    elif model_name == "cct_modified23":
        model = cct_modified23(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified24":
        model = cct_modified24(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified25":
        model = cct_modified25(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified26":
        model = cct_modified26(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified23_":
        model = cct_modified23(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified24_":
        model = cct_modified24(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified25_":
        model = cct_modified25(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified26_":
        model = cct_modified26(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified23_best-lr":
        model = cct_modified23(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified27":
        model = cct_modified27(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified28":
        model = cct_modified28(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified28_":
        model = cct_modified28(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified28_balanced":
        model = cct_modified28(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified28_lr":
        model = cct_modified28(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified28_balanced_lr1e-5":
        model = cct_modified28(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified28_balanced_lr15e-6":
        model = cct_modified28(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified28_balanced_lr15e-5":
        model = cct_modified28(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified28_balanced_lr5e-5":
        model = cct_modified28(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified28_balanced_lr":
        model = cct_modified28(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified28_mlce":
        model = cct_modified28(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified29_balanced_lr1e-5":
        model = cct_modified29(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified29_balanced_lr5e-5":
        model = cct_modified29(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified29_mlce":
        model = cct_modified29(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified30_mlce":
        model = cct_modified30(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified31_mlce":
        model = cct_modified31(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified32_mlce":
        model = cct_modified32(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified33_mlce":
        model = cct_modified33(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified34_mlce":
        model = cct_modified34(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified35_mlce":
        model = cct_modified35(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified34_mlce_balanced":
        model = cct_modified34(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified35_mlce_balanced":
        model = cct_modified35(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified36_mlce_balanced":
        model = cct_modified36(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified36_mlce_balanced_":
        model = cct_modified36(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified37_mlce_balanced":
        model = cct_modified37(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified38_mlce_balanced":
        model = cct_modified38(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified39_mlce_balanced":
        model = cct_modified39(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified40_mlce_balanced":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified40_mlce_balanced_":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified41_mlce_balanced":
        model = cct_modified41(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified41_mlce_balanced_":
        model = cct_modified41(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified42_mlce_balanced":
        model = cct_modified42(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified43_mlce_balanced":
        model = cct_modified43(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified42_mlce_softmax":
        model = cct_modified42(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified43_mlce_softmax":
        model = cct_modified43(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified42_mlce_balanced_":
        model = cct_modified42(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified43_mlce_balanced_":
        model = cct_modified43(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified42_mlce_balanced__":
        model = cct_modified42(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified44_mlce_balanced":
        model = cct_modified44(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified45_mlce_balanced":
        model = cct_modified45(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified42_mlce_balanced___":
        model = cct_modified42(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified40_mlce_balanced_test":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified40_mlce_balanced_test2":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified40_mlce_test":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified40_mlce_lr_test2":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified40_mlce_lr-00001":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified42_mlce_balanced_test":
        model = cct_modified42(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified46_mlce_balanced_test":
        model = cct_modified46(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified44_mlce_balanced_":
        model = cct_modified44(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified45_mlce_balanced_":
        model = cct_modified45(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified46_mlce_lr-00001":
        model = cct_modified46(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified46_mlce_lr-00001_bn_wd-00001":
        model = cct_modified46(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified47_mlce_lr-00001":
        model = cct_modified47(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified48_mlce_lr-000008":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified48_mlce_lr-0000025":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified48_mlce_lr-000008_bn":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified48_mlce_lr-000008_wd-00001":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=False, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified48_mlce_lr-000008_bn_wd-00001":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified48_mlce_lr-000008_wd-005":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=False, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified48_mlce_lr-000008_bn_wd-005":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified48_mlce_lr-000008_wd-02":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=False, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified48_mlce_lr-000008_bn_wd-02":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=cfg.CLASSIFIER.DROP_RATE, attention_dropout=cfg.CLASSIFIER.ATTN_DROP_RATE, stochastic_depth=cfg.CLASSIFIER.DROP_PATH_RATE)
    elif model_name == "cct_modified48_mlce_lr-000008_bn_drop-01_attn-drop-01_drop-path-01":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified48_mlce_lr-000008_bn_drop-03_attn-drop-03_drop-path-03":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.3, attention_dropout=0.3, stochastic_depth=0.3)
    elif model_name == "cct_modified48_mlce_lr-000008_bn_drop-03_attn-drop-0_drop-path-0":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.3, attention_dropout=0., stochastic_depth=0.)
    elif model_name == "cct_modified48_mlce_lr-000008_bn_drop-0_attn-drop-03_drop-path-0":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0., attention_dropout=0.3, stochastic_depth=0.)
    elif model_name == "cct_modified48_mlce_lr-000008_bn_drop-0_attn-drop-0_drop-path-03":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0., attention_dropout=0., stochastic_depth=0.3)
    elif model_name == "cct_modified48_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified48_mlce_lr-000005_bn_drop-02_attn-drop-02_drop-path-02":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.2)
    elif model_name == "cct_modified48_mlce_lr-000005_bn_drop-03_attn-drop-03_drop-path-03":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.3, attention_dropout=0.3, stochastic_depth=0.3)
    elif model_name == "cct_modified48_mlce_lr-000005_bn_drop-04_attn-drop-04_drop-path-04":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.4, attention_dropout=0.4, stochastic_depth=0.4)
    elif model_name == "cct_modified48_mlce_lr-000005_bn_drop-0_attn-drop-03_drop-path-0":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0., attention_dropout=0.3, stochastic_depth=0.)
    elif model_name == "cct_modified48_mlce_lr-000005_bn_drop-01_attn-drop-03_drop-path-01":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.3, stochastic_depth=0.1)
    elif model_name == "cct_modified48_mlce_lr-000005_bn_drop-02_attn-drop-03_drop-path-02":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.3, stochastic_depth=0.2)
    elif model_name == "cct_modified48_mlce_lr-000005_bn_drop-03_attn-drop-0_drop-path-0":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.3, attention_dropout=0., stochastic_depth=0.)
    elif model_name == "cct_modified48_mlce_lr-000005_bn_drop-0_attn-drop-0_drop-path-03":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0., attention_dropout=0., stochastic_depth=0.3)
    elif model_name == "cct_modified48_mlce_lr-000005_bn_drop-01_attn-drop-005_drop-path-025":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.05, stochastic_depth=0.25)
    elif model_name == "cct_modified48_mlce_lr-000004_bn_drop-02_attn-drop-01_drop-path-03":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.1, stochastic_depth=0.3)
    elif model_name == "cct_modified48_mlce_lr-000004_bn_drop-02_attn-drop-0_drop-path-05_batch8":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0., stochastic_depth=0.5)
    elif model_name == "cct_modified48_mlce_lr-000004_bn_drop-015_attn-drop-005_drop-path-05_batch8":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.15, attention_dropout=0.05, stochastic_depth=0.5)
    elif model_name == "cct_modified49_mlce_lr-000004_bn_drop-015_attn-drop-005_drop-path-05_batch8":
        model = cct_modified49(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.15, attention_dropout=0.05, stochastic_depth=0.5)
    elif model_name == "cct_modified50_mlce_lr-000004_bn_drop-015_attn-drop-005_drop-path-05_batch8":
        model = cct_modified50(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.15, attention_dropout=0.05, stochastic_depth=0.5)
    elif model_name == "cct_modified48_mlce_lr-0000016_bn_drop-01_attn-drop-01_drop-path-01_batch8":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified48_mlce_lr-000004_bn_drop-01_attn-drop-02_drop-path-03_batch8_gamma-08":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified50_mlce_lr-000004_bn_drop-01_attn-drop-02_drop-path-03_batch8_gamma-08":
        model = cct_modified50(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified51_mlce_lr-000004_bn_drop-01_attn-drop-02_drop-path-03_batch8_gamma-08":
        model = cct_modified51(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified48_mlce_lr-000004_drop-01_attn-drop-02_drop-path-03_batch8_gamma-08":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=False, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified52_mlce_lr-000004_bn_drop-01_attn-drop-02_drop-path-03_batch8_gamma-08":
        model = cct_modified52(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified51_mlce_lr-000004_bn1_drop-01_attn-drop-02_drop-path-03_batch8_gamma-08":
        model = cct_modified51(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified51_mlce_lr-000004_bn_drop-01_attn-drop-02_drop-path-03_batch8*2_gamma-08":
        model = cct_modified51(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified51_mlce_lr-000008_bn1_drop-01_attn-drop-0_drop-path-05_batch6*5":
        model = cct_modified51(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0., stochastic_depth=0.5)
    elif model_name == "cct_modified51_mlce_lr-000008_bn1_drop-01_attn-drop-0_drop-path-025_batch6*5":
        model = cct_modified51(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0., stochastic_depth=0.25)
    elif model_name == "cct_modified53_mlce_lr-000004_bn_drop-01_attn-drop-01_drop-path-01_batch6*2_gamma-085":
        model = cct_modified53(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified53_mlce_lr-000004_bn_drop-02_attn-drop-02_drop-path-02_batch8_gamma-085":
        model = cct_modified53(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.2)
    elif model_name == "cct_modified53_mlce_lr-000004_bn_drop-005_attn-drop-005_drop-path-01_batch6*2_reCos":
        model = cct_modified53(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.05, attention_dropout=0.05, stochastic_depth=0.1)
    elif model_name == "cct_modified50_mlce_lr-000006_bn_drop-01_attn-drop-01_drop-path-01_batch6*2_gamma-097":
        model = cct_modified50(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified50_mlce_lr-000006_bn_drop-02_attn-drop-02_drop-path-02_batch6*2_gamma-097":
        model = cct_modified50(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.2)
    elif model_name == "cct_modified50_mlce_lr-000006_bn_drop-01_attn-drop-02_drop-path-03_batch6*2_gamma-097":
        model = cct_modified50(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified50_mlce_lr-000006_bn_drop-02_attn-drop-02_drop-path-01_batch6*2_gamma-097":
        model = cct_modified50(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.1)
    elif model_name == "cct_modified50_mlce_lr-00001_bn_drop-01_attn-drop-01_drop-path-01_batch6*5_gamma-097":
        model = cct_modified50(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified50_mlce_lr-00001_bn_drop-02_attn-drop-02_drop-path-02_batch6*5_gamma-097":
        model = cct_modified50(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.2)
    elif model_name == "cct_modified50_mlce_lr-00001_bn_drop-01_attn-drop-02_drop-path-03_batch6*5_gamma-097":
        model = cct_modified50(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified50_mlce_lr-00001_bn_drop-02_attn-drop-02_drop-path-01_batch6*5_gamma-097":
        model = cct_modified50(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.1)

    elif model_name == "cct_modified48_mlce_lr-000004_bn_drop-005_attn-drop-005_drop-path-005_batch6_gamma-097":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.05, attention_dropout=0.05, stochastic_depth=0.05)
    elif model_name == "cct_modified48_mlce_lr-000004_bn_drop-01_attn-drop-01_drop-path-01_batch6_gamma-097":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified48_mlce_lr-000004_bn_drop-02_attn-drop-02_drop-path-02_batch6_gamma-097":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.2)
    elif model_name == "cct_modified48_mlce_lr-000004_bn_drop-01_attn-drop-02_drop-path-03_batch6_gamma-097":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified48_mlce_lr-000004_bn_drop-02_attn-drop-02_drop-path-01_batch6_gamma-097":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.1)
    elif model_name == "cct_modified48_mlce_lr-000004_bn_drop-01_attn-drop-02_drop-path-03_batch6_gamma-085_wd-00001":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified48_mlce_lr-00005_bn_drop-01_attn-drop-02_drop-path-03_batch6_momentum":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified48_mlce_lr-000004_bn_drop-015_attn-drop-015_drop-path-03_batch6_gamma-085":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.15, attention_dropout=0.15, stochastic_depth=0.3)
    elif model_name == "cct_modified48_mlce_lr-000004_bn_drop-01_attn-drop-02_drop-path-03_batch6_adabelief_wd-005":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified50_mlce_lr-000004_bn_drop-01_attn-drop-02_drop-path-03_batch6_gamma-084":
        model = cct_modified50(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified54_mlce_lr-000004_bn_batch8_gamma-084":
        model = cct_modified54(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0., attention_dropout=0., stochastic_depth=0.)
    elif model_name == "cct_modified54_mlce_lr-000004_bn_drop-0_attn-drop-0_drop-path-05_batch8_gamma-084":
        model = cct_modified54(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0., attention_dropout=0., stochastic_depth=0.5)
    elif model_name == "cct_modified50_mlce_lr-0000025_bn_drop-01_attn-drop-02_drop-path-05_batch6_gamma-084":
        model = cct_modified50(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.5)
    elif model_name == "cct_modified48_mlce_lr-0000025_bn_drop-01_attn-drop-02_drop-path-03_batch6_gamma-085":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified48_mlce_scale-2_lr-00002_bn_drop-015_attn-drop-02_drop-path-04_batch6*5_gamma-07":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.15, attention_dropout=0.2, stochastic_depth=0.4)
    elif model_name == "cct_modified48_mlce_scale-2_lr-00001_bn_drop-02_attn-drop-02_drop-path-03_batch6*2_gamma-085_wd-005":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified48_mlce_scale-2_lr-000003_bn_drop-02_attn-drop-01_drop-path-02_batch6_wd-02":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.1, stochastic_depth=0.2)
    elif model_name == "cct_modified48_mlce_scale-2_lr-000003_bn_drop-02_attn-drop-01_drop-path-02_batch6":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.1, stochastic_depth=0.2)
    elif model_name == "cct_modified48_mlce_lr-000003_bn_drop-02_attn-drop-01_drop-path-02_batch6_wd-02":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.1, stochastic_depth=0.2)
    elif model_name == "cct_modified48_mlce_lr-000003_bn_drop-02_attn-drop-01_drop-path-02_batch6":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.1, stochastic_depth=0.2)
    elif model_name == "cct_modified48_mlce_lr-000003_bn_drop-02_attn-drop-01_drop-path-02_batch6_gamma-085":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.1, stochastic_depth=0.2)
    elif model_name == "cct_modified48_mlce_lr-000003_bn_drop-02_attn-drop-0_drop-path-04_batch6":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0., stochastic_depth=0.4)
    elif model_name == "cct_modified48_mlce_lr-0000035_bn_drop-02_attn-drop-01_drop-path-02_batch6_gamma-075":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.1, stochastic_depth=0.2)
    elif model_name == "cct_modified48_mlce_lr-000004_bn_drop-01_attn-drop-02_drop-path-03_batch6_gamma-097_seed3407":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified48_mlce_lr-0000025_bn_drop-02_attn-drop-01_drop-path-03_batch6_seed6293":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.1, stochastic_depth=0.3)
    elif model_name == "cct_modified48_mlce_lr-000015_bn_drop-02_attn-drop-0_drop-path-0_batch6*5_seed6293":
        model = cct_modified48(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, norm=True, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0., stochastic_depth=0.)
    elif model_name == "cct_modified55_mlce_lr-000004_bn_drop-01_attn-drop-01_drop-path-01_batch6_seed6293":
        model = cct_modified55(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified55_mlce_lr-000004_bn_batch6_seed6293":
        model = cct_modified55(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0., attention_dropout=0., stochastic_depth=0.)
    elif model_name == "cct_modified55_mlce_lr-000004_bn_drop-02_attn-drop-02_drop-path-02_batch6_seed6293":
        model = cct_modified55(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.2)
    elif model_name == "cct_modified55_mlce_lr-000004_bn_drop-02_attn-drop-0_drop-path-0_batch6_seed6293":
        model = cct_modified55(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0., stochastic_depth=0.)
    elif model_name == "cct_modified29_mlce_lr-000004_bn_batch6_seed6293":
        model = cct_modified29(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0., attention_dropout=0., stochastic_depth=0.)
    elif model_name == "cct_modified29_mlce_lr-0000037_bn_drop-01_attn-drop-01_drop-path-01_batch6_seed6293":
        model = cct_modified29(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified29_mlce_lr-0000037_bn_drop-02_attn-drop-02_drop-path-02_batch6_seed6293":
        model = cct_modified29(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.2)
    elif model_name == "cct_modified29_mlce_lr-0000037_bn_drop-02_attn-drop-0_drop-path-0_batch6_seed6293":
        model = cct_modified29(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0., stochastic_depth=0.)
    elif model_name == "cct_modified29_mlce_lr-0000037_bn_drop-0_attn-drop-0_drop-path-05_batch6_seed6293":
        model = cct_modified29(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0., attention_dropout=0., stochastic_depth=0.5)
    elif model_name == "cct_modified29_mlce_lr-0000037_bn_drop-05_attn-drop-0_drop-path-0_batch6_seed6293":
        model = cct_modified29(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.5, attention_dropout=0., stochastic_depth=0.)
    elif model_name == "cct_modified29_mlce_lr-0000037_bn_drop-0_attn-drop-05_drop-path-0_batch6_seed6293":
        model = cct_modified29(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0., attention_dropout=0.5, stochastic_depth=0.)
    elif model_name == "cct_modified29_mlce_lr-0000037_bn_drop-03_attn-drop-03_drop-path-03_batch6_seed6293":
        model = cct_modified29(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.3, attention_dropout=0.3, stochastic_depth=0.3)
    elif model_name == "cct_modified29_mlce_lr-0000037_bn_drop-02_attn-drop-02_drop-path-04_batch6_seed6293":
        model = cct_modified29(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.4)
    elif model_name == "cct_modified29_mlce_lr-0000037_bn_drop-01_attn-drop-01_drop-path-02_batch6_seed6293":
        model = cct_modified29(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.2)
    elif model_name == "cct_modified29_mlce_lr-0000037_bn_drop-01_attn-drop-01_drop-path-01_batch6_seed6293_wd-005":
        model = cct_modified29(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified29_mlce_lr-0000037_bn_drop-01_attn-drop-01_drop-path-01_batch6_seed6293_wd-02":
        model = cct_modified29(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified29_mlce_lr-0000037_bn_drop-01_attn-drop-01_drop-path-01_batch6_seed6293_wd-00001":
        model = cct_modified29(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified40_mlce_lr-0000037_bn_batch6_seed6293":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0., attention_dropout=0., stochastic_depth=0.)
    elif model_name == "cct_modified40_mlce_lr-0000037_bn_drop-02_attn-drop-02_drop-path-04_batch6_seed6293_wd-00001":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.4)
    elif model_name == "cct_modified40_mlce_lr-0000037_bn_drop-02_attn-drop-02_drop-path-04_batch6_seed6293":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.4)
    elif model_name == "cct_modified42_mlce_lr-0000037_bn_drop-02_attn-drop-02_drop-path-02_batch6_seed6293":
        model = cct_modified42(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.2)
    elif model_name == "cct_modified40_mlce_lr-0000037_bn1_drop-02_attn-drop-02_drop-path-02_batch6_seed6293":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.2)
    elif model_name == "cct_modified40_mlce_lr-0000037_bn_batch6_seed6293_wd-02":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0., attention_dropout=0., stochastic_depth=0.)
    elif model_name == "cct_modified40_mlce_lr-0000037_bn_batch6_seed6293_wd-005":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0., attention_dropout=0., stochastic_depth=0.)
    elif model_name == "cct_modified40_mlce_lr-0000037_bn_drop-02_attn-drop-0_drop-path-0_batch6_seed6293_wd-005":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0., stochastic_depth=0.)
    elif model_name == "cct_modified40_mlce_lr-0000037_bn_drop-04_attn-drop-0_drop-path-0_batch6_seed6293_wd-005":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.4, attention_dropout=0., stochastic_depth=0.)
    elif model_name == "cct_modified42_mlce_lr-000004_bn_drop-03_attn-drop-01_drop-path-02_batch6_seed6293_wd-005_aug":
        model = cct_modified42(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.3, attention_dropout=0.1, stochastic_depth=0.2)
    elif model_name == "cct_modified40_mlce_lr-000004_bn_drop-02_attn-drop-01_drop-path-02_batch6_seed6293_wd-005_aug":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0.1, stochastic_depth=0.2)
    elif model_name == "cct_modified42_mlce_lr-000004_bn_drop-025_attn-drop-01_drop-path-03_batch6_seed6293_wd-005_aug":
        model = cct_modified42(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.25, attention_dropout=0.1, stochastic_depth=0.3)
    elif model_name == "cct_modified40_mlce_lr-000004_bn_drop-025_attn-drop-01_drop-path-03_batch6_seed6293_wd-005_aug":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.25, attention_dropout=0.1, stochastic_depth=0.3)
    elif model_name == "cct_modified40_mlce_scale-2_lr-000004_bn_drop-03_attn-drop-02_drop-path-03_batch6_seed6293_wd-005_aug":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.3, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified42_mlce_lr-000004_bn_drop-03_attn-drop-02_drop-path-03_batch6_seed6293_wd-005_aug":
        model = cct_modified42(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.3, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified40_mlce_lr-000004_bn_drop-03_attn-drop-02_drop-path-03_batch6_seed6293_wd-005_aug":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.3, attention_dropout=0.2, stochastic_depth=0.3)
    elif model_name == "cct_modified40_mlce_lr-000004_bn_drop-01_attn-drop-01_drop-path-03_batch6_seed6293_wd-005_aug":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.3)
    elif model_name == "cct_modified40_mlce_lr-000004_bn_drop-03_attn-drop-01_drop-path-02_batch6_seed6293_wd-005_aug":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.3, attention_dropout=0.1, stochastic_depth=0.2)
    elif model_name == "cct_modified40_mlce_lr-0000035_bn_drop-025_attn-drop-005_drop-path-03_batch6_seed6293_wd-005_aug":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.25, attention_dropout=0.05, stochastic_depth=0.25)
    elif model_name == "cct_modified40_mlce_lr-0000035_bn_drop-025_attn-drop-005_drop-path-03_batch6_seed6293_wd-01_aug":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.25, attention_dropout=0.05, stochastic_depth=0.3)
    elif model_name == "cct_modified40_mlce_scale-2_lr-0000035_bn_drop-02_attn-drop-0_drop-path-025_batch6_seed6293_wd-01_aug":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.2, attention_dropout=0., stochastic_depth=0.25)
    elif model_name == "cct_modified40_mlce_lr-0000035_bn_drop-03_attn-drop-0_drop-path-02_batch6_seed6293_wd-005_aug":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2500, n_input_channels=3, dropout=0.3, attention_dropout=0., stochastic_depth=0.2)
    elif model_name == "cct_modified40_mlce_lr-0000028_bn_drop-025_attn-drop-005_drop-path-03_batch8_seed6293_wd-005_aug_no-normalized":
        model = cct_modified40(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2400, n_input_channels=3, dropout=0.25, attention_dropout=0.05, stochastic_depth=0.3)
    elif model_name == "cct_modified56_mlce_lr-000004_bn_drop-0_attn-drop-0_drop-path-05_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0., attention_dropout=0., stochastic_depth=0.5)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-02_attn-drop-01_drop-path-02_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.2, attention_dropout=0.1, stochastic_depth=0.2)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-025_attn-drop-0_drop-path-025_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.25, attention_dropout=0., stochastic_depth=0.25)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-025_attn-drop-0_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.25, attention_dropout=0., stochastic_depth=0.1)
    elif model_name == "cct_modified57_mlce_lr-000007_bn_drop-03_attn-drop-01_drop-path-02_batch8*2_seed6293_wd-00001_aug_no-normalized":
        model = cct_modified57(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.3, attention_dropout=0.1, stochastic_depth=0.2)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-03_attn-drop-0_drop-path-02_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.3, attention_dropout=0., stochastic_depth=0.2)
    elif model_name == "cct_modified58_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch8_seed6293_wd-005_aug_no-normalized":
        model = cct_modified58(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified58_mlce_lr-000006_bn_drop-01_attn-drop-01_drop-path-01_batch8*2_seed6293_wd-005_aug_no-normalized":
        model = cct_modified58(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified59_mlce_lr-0000075_bn_drop-01_attn-drop-01_drop-path-01_batch24_seed6293_wd-005_aug_no-normalized":
        model = cct_modified59(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified61_mlce_lr-000007_bn_drop-02_attn-drop-01_drop-path-01_batch12*2_seed6293_wd-0001_aug_no-normalized":
        model = cct_modified61(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.2, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_lr-000007_bn_drop-015_attn-drop-0_drop-path-01_batch12*2_seed6293_wd-0001_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.15, attention_dropout=0., stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-015_attn-drop-0_drop-path-01_batch12_seed6293_wd-0001_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.15, attention_dropout=0., stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_lr-000015_bn_drop-01_attn-drop-01_drop-path-01_batch12*5_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified62_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified62(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified63_mlce_lr-000012_bn_drop-05_attn-drop-01_drop-path-01_batch8*5_seed6293_wd-005_aug_no-normalized":
        model = cct_modified63(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.5, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch8_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-01_attn-drop-005_drop-path-005_batch12_seed6293_wd-001_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.05, stochastic_depth=0.05)
    elif model_name == "cct_modified63_mlce_lr-000005_bn_drop-01_attn-drop-005_drop-path-005_batch8_seed6293_wd-001_aug_no-normalized":
        model = cct_modified63(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.05, stochastic_depth=0.05)
    elif model_name == "cct_modified63_mlce_lr-000005_bn_drop-02_attn-drop-01_drop-path-01_batch8_seed6293_wd-005_aug_no-normalized":
        model = cct_modified63(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.2, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified63_mlce_lr-0000035_bn_drop-025_attn-drop-01_drop-path-01_batch8_seed6293_wd-005_aug_no-normalized":
        model = cct_modified63(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.25, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_lr-0000045_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified63_mlce_lr-000005_bn_drop-025_attn-drop-01_drop-path-01_batch8*2_seed6293_wd-0025_aug_no-normalized":
        model = cct_modified63(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.25, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-02_attn-drop-005_drop-path-005_batch12_seed6293_wd-0025_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.2, attention_dropout=0.05, stochastic_depth=0.05)
    elif model_name == "cct_modified56_mlce_lr-0000125_bn_drop-01_attn-drop-01_drop-path-01_batch12*5_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified63_mlce_lr-000015_bn_drop-02_attn-drop-02_drop-path-02_batch8*10_seed6293_wd-005_aug_no-normalized":
        model = cct_modified63(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.2)
    elif model_name == "cct_modified56_mlce_lr-000025_bn_drop-01_attn-drop-01_drop-path-01_batch12*10_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_lr-000015_bn_drop-01_attn-drop-01_drop-path-01_batch12*5_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_lr-00001_bn_drop-05_attn-drop-05_drop-path-05_batch12*5_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.5, attention_dropout=0.5, stochastic_depth=0.5)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-04_attn-drop-04_drop-path-04_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.4, attention_dropout=0.4, stochastic_depth=0.4)
    elif model_name == "cct_modified64_mlce_lr-000005_bn_drop-025_attn-drop-025_drop-path-025_batch16_seed6293_wd-005_aug_no-normalized":
        model = cct_modified64(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.25, attention_dropout=0.25, stochastic_depth=0.25)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-05_attn-drop-04_drop-path-04_batch12_seed6293_wd-0005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.5, attention_dropout=0.4, stochastic_depth=0.4)
    elif model_name == "cct_modified56_mlce_lr-00001_bn_drop-05_attn-drop-05_drop-path-05_batch12*5_seed6293_wd-001_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.5, attention_dropout=0.5, stochastic_depth=0.5)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-025_attn-drop-025_drop-path-025_batch12*2_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.25, attention_dropout=0.25, stochastic_depth=0.25)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-03_attn-drop-03_drop-path-03_batch6*5_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.3, attention_dropout=0.3, stochastic_depth=0.3)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-03_attn-drop-03_drop-path-03_batch12_seed6293_wd-01_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.3, attention_dropout=0.3, stochastic_depth=0.3)
    elif model_name == "cct_modified67_mlce_lr-000005_bn_drop-025_attn-drop-025_drop-path-025_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified67(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.25, attention_dropout=0.25, stochastic_depth=0.25)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-02_attn-drop-02_drop-path-02_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.2)
    elif model_name == "cct_modified56_mlce_lr-0000035_bn_drop-02_attn-drop-02_drop-path-02_batch6*5_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.2, attention_dropout=0.2, stochastic_depth=0.2)
    elif model_name == "cct_modified56_mlce_lr-000003_bn_drop-025_attn-drop-025_drop-path-025_batch6_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.25, attention_dropout=0.25, stochastic_depth=0.25)
    elif model_name == "cct_modified56_mlce_lr-000003_bn_drop-01_attn-drop-01_drop-path-01_batch6_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_lr-000003_bn_drop-05_attn-drop-05_drop-path-05_batch6_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.5, attention_dropout=0.5, stochastic_depth=0.5)
    elif model_name == "cct_modified67_mlce_lr-000003_bn_drop-01_attn-drop-01_drop-path-01_batch6_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified67(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_scale-1_lr-000003_bn_drop-01_attn-drop-01_drop-path-01_batch6_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_scale-10_lr-000003_bn_drop-01_attn-drop-01_drop-path-01_batch6_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_scale-2_lr-000003_bn_drop-01_attn-drop-01_drop-path-01_batch6_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_scale-5_lr-000003_bn_drop-01_attn-drop-01_drop-path-01_batch6_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-025_attn-drop-025_drop-path-025_batch6*5_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.25, attention_dropout=0.25, stochastic_depth=0.25)
    elif model_name == "cct_modified56_mlce_scale-1_lr-000003_bn_drop-025_attn-drop-025_drop-path-025_batch6_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.25, attention_dropout=0.25, stochastic_depth=0.25)
    elif model_name == "cct_modified56_mlce_lr-000003_bn_drop-025_attn-drop-025_drop-path-025_batch6_seed6293_wd-00001_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.25, attention_dropout=0.25, stochastic_depth=0.25)
    elif model_name == "cct_modified56_mlce_wpos-06_lr-000003_bn_drop-025_attn-drop-025_drop-path-025_batch6_seed6293_wd-00001_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.25, attention_dropout=0.25, stochastic_depth=0.25)
    elif model_name == "cct_modified56_mlce_weighted2_lr-000003_bn_drop-025_attn-drop-025_drop-path-025_batch6_seed6293_wd-00001_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.25, attention_dropout=0.25, stochastic_depth=0.25)
    elif model_name == "cct_modified56_mlce_posWeighted_negWeighted_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_posWeighted_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_negWeighted_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_posWeighted_negWeighted_lr-000005_bn_drop-025_attn-drop-025_drop-path-025_batch12_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.25, attention_dropout=0.25, stochastic_depth=0.25)
    elif model_name == "cct_modified56_mlce_posWeighted_NWeighted_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_posWeighted_negWeighted_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-00001_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_posWeighted_lr-000005_bn_drop-025_attn-drop-025_drop-path-025_batch12_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.25, attention_dropout=0.25, stochastic_depth=0.25)
    elif model_name == "cct_modified56_mlce_posWeighted_lr-000003_bn_drop-025_attn-drop-025_drop-path-025_batch6_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.25, attention_dropout=0.25, stochastic_depth=0.25)
    elif model_name == "cct_modified56_mlce_posWeighted_lr-000003_bn_drop-01_attn-drop-01_drop-path-01_batch6_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_posWeighted_lr-000003_bn_drop-01_attn-drop-01_drop-path-01_batch6_seed6293_wd-00001_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_posWeighted_lr-0000025_bn_drop-01_attn-drop-01_drop-path-01_batch6_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_reWeight_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_ReWeight_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized_60epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce_ReWeight-sum_lr-000008_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_normalized_30epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_bce-logit-ReWeight_lr-000008_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_normalized_30epoch":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)
    elif model_name == "cct_modified56_mlce-logsumexp-_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized":
        model = cct_modified56(num_classes=cfg.CLASSIFIER.CLASSES_NUM if not pretrain else 0, img_size=2200, n_input_channels=3, dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1)


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