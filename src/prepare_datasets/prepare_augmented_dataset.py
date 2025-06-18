import pandas as pd
import selfies as sf
from rdkit import Chem
import os
from sklearn.model_selection import StratifiedKFold
import random


def smiles_to_selfies(smiles_string):
    try:
        return sf.encoder(smiles_string)
    except:
        return None


def convert_smiles_to_selfies(df, df_name, dropna=False):
    df_copy = df.copy()
    print(
        f"No. samples before converting smiles to selfies for {df_name}: {len(df_copy)}"
    )
    df_copy["selfies"] = df_copy["smiles"].apply(lambda x: smiles_to_selfies(x))
    if dropna:
        df_copy = df_copy.dropna(subset=["selfies"])
    print(
        f"No. samples after converting smiles to selfies for {df_name}: {len(df_copy) - df_copy['selfies'].isnull().sum()}"
    )
    return df_copy


def check_valid_smiles(smiles_string, print_invalid=False):
    try:
        molecule = Chem.MolFromSmiles(smiles_string)
        if molecule is None:
            if print_invalid:
                print(f"Invalid SMILES: {smiles_string}")
            return None
        return smiles_string
    except:
        if print_invalid:
            print(f"Invalid SMILES: {smiles_string}")
        return None


def remove_invalid_smiles(df, df_name):
    df_copy = df.copy()
    print(f"No. samples before removing invalid smiles from {df_name}: {len(df_copy)}")
    df_copy["smiles"] = df_copy["smiles"].apply(lambda x: check_valid_smiles(x))
    df_copy = df_copy.dropna(subset=["smiles"])
    print(f"No. samples after removing invalid smiles from {df_name}: {len(df_copy)}")
    return df_copy


def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def remove_duplicates(df, df_name):
    df_copy = df.copy()
    print(f"No. samples before removing duplicates from {df_name}: {len(df_copy)}")
    df_copy["canonical_smiles"] = df_copy["smiles"].apply(lambda x: canonicalize(x))
    df_copy = df_copy.drop_duplicates(subset="canonical_smiles").dropna()
    df_copy = df_copy.drop(columns=["canonical_smiles"])
    print(f"No. samples after removing duplicates from {df_name}: {len(df_copy)}")
    return df_copy


def random_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=False, doRandom=True)


def augment_data(df, df_name, num_augmentations=8, limit=50, print_invalid=False):
    print(f"No. samples before augmenting {df_name}: {len(df)}")
    print(df["label"].value_counts())
    augmented_data = {"smiles": [], "label": []}
    num_cannot_augment = 0
    for i in range(len(df)):
        smiles = df.iloc[i]["smiles"]
        label = df.iloc[i]["label"]

        augmented_data["label"].append(label)
        augmented_data["smiles"].append(smiles)

        if label == 0:
            continue

        smiles_list = [smiles]
        count, count_limit = 0, 0
        while count < num_augmentations:
            augmented_smiles = random_smiles(smiles)
            if augmented_smiles not in smiles_list:
                smiles_list.append(augmented_smiles)
                augmented_data["label"].append(label)
                augmented_data["smiles"].append(augmented_smiles)
                count += 1
            else:
                count_limit += 1
                if count_limit > limit:
                    if print_invalid:
                        print(
                            f"Cannot augment {smiles} anymore, num augmentations: {count}"
                        )
                    num_cannot_augment += 1
                    break

    augmented_df = pd.DataFrame(augmented_data)
    print(f"No. samples after augmenting {df_name}: {len(augmented_df)}")
    print(augmented_df["label"].value_counts())
    print(f"Number of SMILES that cannot be augmented: {num_cannot_augment}")
    return augmented_df


def main(args):
    random.seed(args.seed)
    data_folder = args.data_folder

    original_df = pd.read_csv(
        os.path.join(data_folder, "csv", "original_df.csv"), index_col=args.index_col
    )
    train_df = original_df[original_df["split"] == "train"]
    test_df = original_df[original_df["split"] == "test"]
    train_df = train_df.drop(columns=["split"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["split"]).reset_index(drop=True)

    try:
        train_df = train_df.drop(columns=["class"]).reset_index(drop=True)
        test_df = test_df.drop(columns=["class"]).reset_index(drop=True)
    except:
        pass

    print(f"Number of samples in train: {len(train_df)}")
    print(f"Number of samples in test: {len(test_df)}")
    print("==========================================================================")

    # Check valid smiles
    train_df = remove_invalid_smiles(train_df, df_name="train")
    test_df = remove_invalid_smiles(test_df, df_name="test")
    print(f"Value counts for train:")
    print(train_df["label"].value_counts())
    print(f"Value counts for test:")
    print(test_df["label"].value_counts())
    print("==========================================================================")

    # Remove duplicates
    train_df = remove_duplicates(train_df, df_name="train")
    test_df = remove_duplicates(test_df, df_name="test")
    print(f"Value counts for train:")
    print(train_df["label"].value_counts())
    print(f"Value counts for test:")
    print(test_df["label"].value_counts())
    print("==========================================================================")

    # Augment for train_df
    aug_train_df = augment_data(
        train_df, df_name="train", num_augmentations=args.num_augmentations, limit=50
    )
    print(f"Value counts for train:")
    print(aug_train_df["label"].value_counts())
    print(f"Value counts for test:")
    print(test_df["label"].value_counts())

    # create selfies column for test_df
    save_train_df = convert_smiles_to_selfies(
        aug_train_df, df_name="save_train", dropna=True
    )
    test_df = convert_smiles_to_selfies(test_df, df_name="save_test", dropna=True)
    print(f"Value counts for train:")
    print(save_train_df["label"].value_counts())
    print(f"Value counts for test:")
    print(test_df["label"].value_counts())

    print("==========================================================================")

    save_train_df["id"] = range(len(save_train_df))
    test_df["id"] = range(len(test_df))

    save_train_df = save_train_df.sample(frac=1, random_state=args.seed)
    save_train_df.to_csv(os.path.join(data_folder, "csv", "train.csv"), index=False)
    test_df.to_csv(os.path.join(data_folder, "csv", "test.csv"), index=False)

    num_folds = args.num_folds
    k_fold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=args.seed)
    splits = k_fold.split(train_df, train_df["label"])
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_fold_df = train_df.iloc[train_idx]
        val_fold_df = train_df.iloc[val_idx]

        save_dir = os.path.join(data_folder, f"fold_{fold}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=False)

        # Augment for train_fold_df
        aug_train_fold_df = augment_data(
            train_fold_df,
            df_name="train_fold",
            num_augmentations=args.num_augmentations,
            limit=50,
        )

        # Convert smiles to selfies
        train_fold_df = convert_smiles_to_selfies(
            aug_train_fold_df, df_name="train_fold"
        )
        val_fold_df = convert_smiles_to_selfies(val_fold_df, df_name="val_fold")

        train_fold_df = train_fold_df.sample(frac=1, random_state=args.seed)

        train_fold_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
        val_fold_df.to_csv(os.path.join(save_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)

        print(f"Fold {fold}:")
        print(f"Train: {len(train_fold_df)}")
        print(train_fold_df["label"].value_counts())
        print(f"Val: {len(val_fold_df)}")
        print(val_fold_df["label"].value_counts())
        print(f"Test: {len(test_df)}")
        print(test_df["label"].value_counts())
        print(
            "=========================================================================="
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="data/augmented")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--num_augmentations", type=int, default=8)
    parser.add_argument("--index_col", default=None)
    args = parser.parse_args()
    main(args)
