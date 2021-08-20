import os
from datasets import Dataset, DatasetDict

def get_data_epoch(args, train_dataloader, model):

    return

def get_dataset(raw_datasets_dict, data_file, split_list, gt_langs, features):
    for split in split_list:
        if split in ["train"]:
            raw_dict_others = {features[0]: [], features[1]: []}
        for lg in gt_langs:
            raw_dict = {features[0]: [], features[1]: []}
            data_file_tmp = data_file.format(lg)
            print("loading: ", split, lg, data_file_tmp)
            with open(data_file_tmp + ".src." + split, encoding="utf-8") as src_f, open(
                    data_file_tmp + ".tgt." + split, encoding="utf-8") as tgt_f:
                for idx, (src_line, tgt_line) in enumerate(zip(src_f, tgt_f)):
                    raw_dict[features[0]].append(src_line.strip())
                    raw_dict[features[1]].append(tgt_line.strip())
                    if split in ["train"] and lg not in ['en']:
                        raw_dict_others[features[0]].append(src_line.strip())
                        raw_dict_others[features[1]].append(tgt_line.strip())

            raw_datasets = Dataset.from_dict(raw_dict)
            if split == "dev":
                raw_datasets_dict["validation." + lg] = raw_datasets
            else:
                raw_datasets_dict[split + "." + lg] = raw_datasets

        if split in ["train"]:
            raw_datasets_others = Dataset.from_dict(raw_dict_others)
            raw_datasets_dict[split + ".others"] = raw_datasets_others

    return raw_datasets_dict

def load_xglue(dataset_name, data_folder, gt_langs, multi_train):
    features = ["news_body", "news_title"]
    raw_datasets_dict = DatasetDict()
    data_path = "xglue." + dataset_name + ".{}"
    if not multi_train:
        data_file = os.path.join(data_folder, data_path)
        raw_datasets_dict = get_dataset(raw_datasets_dict, data_file, ["train"], ["en"], features)
        split_list = ["dev", "test"]
        print("not multi_train, done train.en")
    else:
        split_list = ["train", "dev", "test"]

    data_file = os.path.join(data_folder, data_path)
    raw_datasets_dict = get_dataset(raw_datasets_dict, data_file, split_list, gt_langs, features)

    raw_datasets_dict["train"] = raw_datasets_dict["train.en"]

    return raw_datasets_dict

if __name__ == '__main__':
    dataset_name = "ntg"
    data_folder = "../datasets/xglue_full_dataset/sampled_NTG"
    gt_langs = ['en', 'fr', 'es', 'ru', 'de']
    multi_train = True
    raw_datasets = load_xglue(dataset_name, data_folder, gt_langs, multi_train)

    print(raw_datasets)
    column_names = raw_datasets["train.en"].column_names
    print(column_names)

    index_list = [0, 5]
    for split in ["train", "validation", "test"]:
        for lg in gt_langs:
            data_tmp = raw_datasets[split + "." + lg]
            print(lg, split, len(data_tmp))
            for index in index_list:
                print(index, data_tmp["news_body"][index][:30])

    print(len(raw_datasets["train.others"]))
