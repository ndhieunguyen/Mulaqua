from transformers import T5Tokenizer, T5Config


def prepare_tokenizer(args):
    try:
        return T5Tokenizer.from_pretrained(
            args.pretrained_name, cache_dir=args.cache_dir
        )
    except Exception as e:
        return T5Tokenizer.from_pretrained(args.pretrained_name)


def check_unfreeze_layer(name, trainable_layers):
    if name.startswith("transformer.decoder.my"):
        return True
    flag = False
    for layer in trainable_layers:
        if name.startswith(f"transformer.decoder.block.{layer}"):
            flag = True
            break
    return flag


def prepare_model(args, inference=False):
    from src.modeling_t5 import T5ForSequenceClassification

    id2lable = {0: "non-PMT/vPvM", 1: "PMT/vPvM"}
    label2id = {v: k for k, v in id2lable.items()}
    config = T5Config.from_pretrained(
        args.pretrained_name,
        cache_dir=args.cache_dir,
        num_labels=len(id2lable),
        id2label=id2lable,
        label2id=label2id,
    )
    config.dropout_rate = args.dropout if not inference else 0
    config.classifier_dropout = args.dropout if not inference else 0
    config.problem_type = "single_label_classification"
    config.swin_ocsr_path = args.swin_ocsr_path
    config.swin_molnextr_path = args.swin_molnextr_path
    config.cls_num_heads = args.cls_num_heads
    config.trainable_layers = args.trainable_layers if not inference else "None"
    config.swin_used = args.swin_used
    config.loss_function = args.loss_function
    config.focal_gamma = args.focal_gamma

    model = T5ForSequenceClassification.from_pretrained(
        args.pretrained_name,
        cache_dir=args.cache_dir,
        config=config,
    )
    model.to(args.accelerator)
    for name, param in model.named_parameters():
        if inference:
            param.requires_grad = False
            continue

        if name.startswith("classif") or check_unfreeze_layer(
            name, args.trainable_layers
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model
