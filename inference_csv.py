import torch
from torchvision import transforms as t
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
from src.model import prepare_model, prepare_tokenizer
import torch.nn.functional as F
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class FinalModel(torch.nn.Module):
    def __init__(self, args):
        super(FinalModel, self).__init__()

        # Load the text model
        self.text_tokenizer = prepare_tokenizer(args)
        self.text_model = prepare_model(args, inference=True)
        self.text_model.to("cuda")
        self.text_model.eval()
        for param in self.text_model.parameters():
            param.requires_grad = False

        # Load the swin model
        if args.swin_used == "ocsr":
            self.image_transform = t.Compose(
                [
                    t.Resize((224, 224)),
                    t.ToTensor(),
                    t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            from src.swin_ocsr import SwinTransformer

            self.swin = SwinTransformer(
                img_size=(224, 224),
                num_classes=4,
                embed_dim=192,
                depths=(2, 2, 18, 2),
                num_heads=(6, 12, 24, 48),
            )
            self.swin.load_state_dict(torch.load(args.swin_ocsr_path)["encoder"])
            self.swin.to("cuda")
            self.swin.eval()
            for param in self.swin.parameters():
                param.requires_grad = False
        elif args.swin_used == "molnextr":
            from src.MolNexTR.MolNexTR.model import molnextr
            from src.MolNexTR.MolNexTR.components import Encoder
            from src.MolNexTR.MolNexTR.model import loading

            model = molnextr(args.swin_molnextr_path, torch.device("cuda"))
            self.image_transform = model.transform
            state_dict = torch.load(args.swin_molnextr_path)
            model_args = model._get_args(state_dict["args"])
            encoder = Encoder(model_args)
            loading(encoder, state_dict["encoder"])
            self.swin: torch.nn.Module = encoder.transformer
            self.swin.to("cuda")
            self.swin.eval()
            for param in self.swin.parameters():
                param.requires_grad = False

    def generate_image_from_smiles(self, smiles_string):
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

    def transform_image(self, image, swin_pretrained_name):
        assert swin_pretrained_name in ["molnextr", "ocsr"]
        if swin_pretrained_name == "ocsr":
            img = image.convert("L")
            img = img.convert("RGB")
            img = self.image_transform(img)
        else:
            img = np.array(image)
            img = self.image_transform(image=img, keypoints=[])["image"]
        return img

    def prepare_image(self, molecule, swin_pretrained_name) -> torch.Tensor:
        image = self.generate_image_from_smiles(molecule)
        image = self.transform_image(image, swin_pretrained_name)
        return image

    def extract_swin_features(self, images):
        features = self.swin.forward_features(images, return_stages=False)
        features = features.mean(dim=1, keepdim=False)

        return features

    def forward(self, image, input_ids, attention_mask):
        swin_features = self.extract_swin_features(image).to("cuda")
        model_output = self.text_model(
            input_ids=input_ids.to("cuda"),
            attention_mask=attention_mask.to("cuda"),
            swin_features=swin_features.to("cuda"),
        )

        return model_output


class InfereceModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, image, text):
        output = self.tokenizer(
            text, truncation=True, padding=True, return_tensors="pt"
        )
        input_ids = output["input_ids"].to("cuda")
        attention_mask = output["attention_mask"].to("cuda")

        model_output = self.model(image, input_ids, attention_mask)["logits"]
        pred, answer_idx = F.softmax(model_output, dim=1).data.cpu().max(dim=1)

        return pred, answer_idx


def random_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        print(f"Cannot convert SMILES {smiles} because of error {e}")
    return Chem.MolToSmiles(mol, canonical=False, doRandom=True)


def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def main(args):
    import selfies as sf
    import pandas as pd
    from sklearn.metrics import accuracy_score as acc
    from sklearn.metrics import f1_score as f1
    from sklearn.metrics import roc_auc_score as auc
    from sklearn.metrics import precision_score as pre
    from sklearn.metrics import recall_score as rec
    from sklearn.metrics import matthews_corrcoef as mcc
    from sklearn.metrics import confusion_matrix as cm

    print(f"Loading model")
    model = FinalModel(args)
    tokenizer = model.text_tokenizer
    inference_model = InfereceModel(model, tokenizer)
    print(f"Finish loading model")

    df = pd.read_csv("/home/ndhieunguyen/mulaqua/in.csv")
    df = df.dropna()

    smiles = df["smiles"].values
    y_true_all = df["label"].values
    y_true, y_pred, y_prob = [], [], []
    for i, s in tqdm(enumerate(smiles)):
        try:
            selfies = "<bom>" + sf.encoder(canonicalize(s)) + "<eom>"
        except:
            continue
        image = model.prepare_image(s, args.swin_used)
        image = image.unsqueeze(0).to("cuda")

        prob, pred = inference_model.predict(image, selfies)

        y_pred.append(pred)
        y_prob.append(prob)
        y_true.append(y_true_all[i])

    result_df = pd.DataFrame(
        {
            "smiles": smiles,
            "y_true": y_true_all,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
    )
    result_df.to_csv("out.csv", index=False)


if __name__ == "__main__":
    import argparse
    from config import config

    parser = argparse.ArgumentParser()
    for k, v in config.__dict__.items():
        if type(v) in [str, int, float]:
            parser.add_argument(f"--{k}", type=type(v), default=v)
        elif type(v) == bool:
            parser.add_argument(f"--{k}", action="store_false" if v else "store_true")
        elif type(v) == list:
            parser.add_argument(f"--{k}", nargs="*", type=type(v[0]), default=v)

    args = parser.parse_args()
    args.pretrained_name = "/home/ndhieunguyen/final/Mulaqua/checkpoints/Mulaqua"
    main(args)
