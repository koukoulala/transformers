import os
from datasets import Dataset, DatasetDict

def get_dataset(raw_datasets_dict, data_file, split_list, gt_langs, features):
    for split in split_list:
        for lg in gt_langs:
            print("loading: ", split, lg)
            raw_dict = {features[0]: [], features[1]: []}
            data_file = data_file.format(lg)
            with open(data_file + ".src." + split, encoding="utf-8") as src_f, open(
                    data_file + ".tgt." + split, encoding="utf-8"
            ) as tgt_f:
                for idx, (src_line, tgt_line) in enumerate(zip(src_f, tgt_f)):
                    raw_dict[features[0]].append(src_line.strip())
                    raw_dict[features[1]].append(tgt_line.strip())

            raw_datasets = Dataset.from_dict(raw_dict)
            if split == "dev":
                raw_datasets_dict["validation." + lg] = raw_datasets
            else:
                raw_datasets_dict[split + "." + lg] = raw_datasets

    return raw_datasets_dict

def load_xglue(dataset_name, data_folder, gt_langs, multi_train):
    features = ["news_body", "news_title"]
    raw_datasets_dict = DatasetDict()
    if not multi_train:
        train_path = "xglue." + dataset_name + ".en"
        train_file = os.path.join(data_folder, train_path)
        raw_datasets_dict = get_dataset(raw_datasets_dict, train_file, ["train"], ["en"], features)
        split_list = ["dev", "test"]
        print("not multi_train, done train.en")
    else:
        split_list = ["train", "dev", "test"]

    data_path = "xglue." + dataset_name + ".{}"
    data_file = os.path.join(data_folder, data_path)
    raw_datasets_dict = get_dataset(raw_datasets_dict, data_file, split_list, gt_langs, features)

    raw_datasets_dict["train"] = raw_datasets_dict["train.en"]

    return raw_datasets_dict

if __name__ == '__main__':
    dataset_name = "ntg"
    data_folder = "../datasets/xglue_full_dataset/sampled_NTG"
    gt_langs = ['en', 'fr', 'es', 'ru', 'de']
    multi_train = False
    raw_datasets = load_xglue(dataset_name, data_folder, gt_langs, multi_train)

    print(raw_datasets)
    column_names = raw_datasets["train.en"].column_names
    print(column_names)