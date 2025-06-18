from datasets import Dataset, DatasetDict
import pandas as pd
import glob
import os
import numpy as np
from torchvision import transforms
import torch
from rdkit import Chem
from rdkit.Chem import Draw


def generate_image_from_smiles(smiles_string):
    molecule = Chem.MolFromSmiles(smiles_string)
    if molecule is None:
        raise ValueError("Invalid SMILES string")

    image = Draw.MolToImage(
        molecule, size=(512, 512), fitImage=True, kekulize=True, wedgeBonds=True
    )

    img_array = np.array(image)
    non_white_pixels = np.where(img_array[:, :, :-1] != 255)
    x_min, x_max = non_white_pixels[0].min(), non_white_pixels[0].max()
    y_min, y_max = non_white_pixels[1].min(), non_white_pixels[1].max()

    cropped_img = image.crop((y_min, x_min, y_max + 1, x_max + 1))

    return cropped_img


def transform_image(image, transform, swin_pretrained_name):
    assert swin_pretrained_name in ["molnextr", "ocsr"]
    if swin_pretrained_name == "ocsr":
        img = image.convert("L")
        img = img.convert("RGB")
        img = transform(img)
    else:
        img = np.array(image)
        img = transform(image=img, keypoints=[])["image"]
    return img


def prepare_image(molecule, transform, swin_pretrained_name) -> torch.Tensor:
    image = generate_image_from_smiles(molecule)
    image = transform_image(image, transform, swin_pretrained_name)
    return image


def add_prefix_and_postfix(text):
    return "<bom>" + text + "<eom>"


def create_dataset_from_dataframe(dataframe_path, args):
    chosen_feature = args.chosen_feature
    dataframe = pd.read_csv(dataframe_path)
    dataframe = dataframe.dropna(subset=chosen_feature)
    if chosen_feature == "selfies":
        dataframe[chosen_feature] = dataframe[chosen_feature].apply(
            add_prefix_and_postfix
        )
    dataset = Dataset.from_pandas(dataframe)

    def rename_key(sample):
        sample["text"] = sample[chosen_feature]
        return sample

    if args.swin_used == "ocsr":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        from src.swin_ocsr import SwinTransformer

        swin = SwinTransformer(
            img_size=(224, 224),
            num_classes=4,
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
        )
        swin.load_state_dict(torch.load(args.swin_ocsr_path)["encoder"])
        swin.eval()
        swin.to("cuda")
        for param in swin.parameters():
            param.requires_grad = False
    elif args.swin_used == "molnextr":
        from src.MolNexTR.MolNexTR.model import molnextr
        from src.MolNexTR.MolNexTR.components import Encoder
        from src.MolNexTR.MolNexTR.model import loading

        model = molnextr(args.swin_molnextr_path, torch.device("cpu"))
        transform = model.transform
        state_dict = torch.load(args.swin_molnextr_path)
        model_args = model._get_args(state_dict["args"])
        encoder = Encoder(model_args)
        loading(encoder, state_dict["encoder"])
        swin = encoder.transformer
        swin.eval()
        swin.to("cuda")
        for param in swin.parameters():
            param.requires_grad = False

    elif args.swin_used == "none":
        transform = None

    def add_image(sample):
        if args.swin_used == "ocsr":
            image = prepare_image(sample["smiles"], transform, "ocsr")
            swin_features, _ = swin.forward_features(
                image.unsqueeze(0).to("cuda"), return_stages=True
            )
            sample["swin_features"] = swin_features.squeeze(0).mean(dim=0)
        elif args.swin_used == "molnextr":
            image = prepare_image(sample["smiles"], transform, "molnextr")
            swin_features, _ = swin.forward_features(
                image.unsqueeze(0).to("cuda"), return_stages=True
            )
            sample["swin_features"] = swin_features.squeeze(0).mean(dim=0)
        elif args.swin_used == "none":
            sample["image_stages"] = None
        else:
            raise ValueError("swin_used not recognized or implemented")
        return sample

    dataset = dataset.map(rename_key, desc="Renaming keys")
    if transform is not None:
        dataset = dataset.map(add_image, desc="Adding images to dataset")

    return dataset


def create_and_save_datadict(train, val, test, save_path):
    if val is None:
        dataset_dict = DatasetDict({"train": train, "test": test})
        dataset_dict.save_to_disk(save_path)
        return dataset_dict
    dataset_dict = DatasetDict({"train": train, "val": val, "test": test})
    dataset_dict.save_to_disk(save_path)
    return dataset_dict


def prepare_dataset(args):
    fold_folders = sorted(glob.glob(args.data_folder + "/fold_*/"))
    for i, fold_folder in enumerate(fold_folders):
        print(f"Processing fold {i}")
        train_path = os.path.join(fold_folder, "train.csv")
        val_path = os.path.join(fold_folder, "val.csv")
        test_path = os.path.join(fold_folder, "test.csv")

        train = create_dataset_from_dataframe(train_path, args)
        val = create_dataset_from_dataframe(val_path, args)
        test = create_dataset_from_dataframe(test_path, args)
        save_path = os.path.join(
            fold_folder,
            f"dataset_{args.swin_used}_{args.chosen_feature}",
        )
        create_and_save_datadict(train, val, test, save_path)

    print(f"Processing full dataset")
    train_path = os.path.join(args.data_folder, "csv", "train.csv")
    test_path = os.path.join(args.data_folder, "csv", "test.csv")
    train = create_dataset_from_dataframe(train_path, args)
    test = create_dataset_from_dataframe(test_path, args)
    save_path = os.path.join(
        args.data_folder,
        f"dataset_{args.swin_used}_{args.chosen_feature}",
    )
    create_and_save_datadict(train, None, test, save_path)
    print(f"Succesfully created and saved dataset at {save_path}")
