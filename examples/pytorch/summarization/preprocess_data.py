import os
from datasets import Dataset

def get_dataset(raw_dict, data_path, split_list, gt_langs, features):
    for split in split_list:
        for lg in gt_langs:
            print(split, lg)
            raw_dict[split + "." + lg] = {features[0]: [], features[1]: []}
            data_file = data_path.format(lg)
            with open(data_file + ".src." + split, encoding="utf-8") as src_f, open(
                    data_file + ".tgt." + split, encoding="utf-8"
            ) as tgt_f:
                for idx, (src_line, tgt_line) in enumerate(zip(src_f, tgt_f)):
                    raw_dict[split + "." + lg][features[0]] = src_line.strip()
                    raw_dict[split + "." + lg][features[1]] = tgt_line.strip()

    return raw_dict

def load_xglue(dataset_name, data_folder, gt_langs, multi_train):
    features = ["news_body", "news_title"]
    raw_dict = {}
    if not multi_train:
        train_path = "xglue." + dataset_name + ".en"
        train_file = os.path.join(data_folder, train_path)
        raw_dict = get_dataset(raw_dict, train_file, ["train"], ["en"], features)
        split_list = ["dev", "test"]
    else:
        split_list = ["train", "dev", "test"]

    data_path = "xglue." + dataset_name + ".{}"
    raw_dict = get_dataset(raw_dict, data_path, split_list, gt_langs, features)
    raw_datasets = Dataset.from_dict(raw_dict)

    return raw_datasets

if __name__ == '__main__':
    dataset_name = "ntg"
    data_folder = "../datasets/xglue_full_dataset/sampled_NTG"
    gt_langs = ['en', 'fr', 'es', 'ru', 'de']
    multi_train = False
    raw_datasets = load_xglue(dataset_name, data_folder, gt_langs, multi_train)

    print(raw_datasets)
    for k, v in raw_datasets:
        print(k, v)

    column_names = raw_datasets["train"].column_names
    print(column_names)