import argparse
import os
import numpy as np
import pandas as pd
from pandas import DataFrame

def get_sample_label(task, sample):
    if task in ["SST-2", "MRPC", "QQP", "STS-B", "MNLI", "SNLI", "QNLI", "RTE", "WNLI", "CoLA"]:
        sample = sample.strip().split('\t')
        if task == 'CoLA':
            return sample[1]
        elif task == 'MNLI':
            return sample[-1]
        elif task == 'MRPC':
            return sample[0]
        elif task == 'QNLI':
            return sample[-1]
        elif task == 'QQP':
            return sample[-1]
        elif task == 'RTE':
            return sample[-1]
        elif task == 'SNLI':
            return sample[-1]
        elif task == 'SST-2':
            return sample[-1]
        elif task == 'STS-B':
            return 0 if float(sample[-1]) < 2.5 else 1
        elif task == 'WNLI':
            return sample[-1]
        else:
            raise NotImplementedError
    else:
        return sample[0]

def load_datasets(data_dir, tasks):
    datasets = {}
    for task in tasks:
        if task in ["SST-2", "MRPC", "QQP", "STS-B", "MNLI", "SNLI", "QNLI", "RTE", "WNLI", "CoLA"]:
            dataset = {}
            dirname = os.path.join(data_dir, task)
            if task == "MNLI":
                splits = ["train", "dev_matched", "dev_mismatched"]
            else:
                splits = ["train", "dev"]
            for split in splits:
                filename = os.path.join(dirname, f"{split}.tsv")
                with open(filename, "r") as f:
                    lines = f.readlines()
                dataset[split] = lines
            datasets[task] = dataset
        else:
            dataset = {}
            dirname = os.path.join(data_dir, task)
            splits = ["train", "test"]
            for split in splits:
                filename = os.path.join(dirname, f"{split}.csv")
                dataset[split] = pd.read_csv(filename, header=None)
            datasets[task] = dataset
    return datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=512, help="Training samples for each class.")
    parser.add_argument("--tasks", type=str, nargs="+",
        default=["SST-2", "sst-5", "mr", "cr", "mpqa", "subj", "trec", "CoLA", "MRPC", "QQP", "STS-B", "MNLI", "SNLI", "QNLI", "RTE"], help="Task names")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42], help="Random seeds")
    parser.add_argument("--data_dir", type=str, default="original", help="Original path")
    parser.add_argument("--output_dir", type=str, default="", help="Output path")
    parser.add_argument("--save_dir", type=str, default="k-data", help="Save path")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.save_dir)
    
    datasets = load_datasets(args.data_dir, args.tasks)

    for seed in args.seeds:
        print("Seed = %d" % (seed))
        for task, dataset in datasets.items():
            np.random.seed(seed)

            print("Task = %s" % (task))
            if task in ["SST-2", "MRPC", "QQP", "STS-B", "MNLI", "SNLI", "QNLI", "RTE", "WNLI", "CoLA"]:
                # GLUE style
                train_data = dataset["train"]
                train_header = train_data[0:1] if task != "CoLA" else []
                train_dataset = train_data[1:]
                np.random.shuffle(train_dataset)
            else:
                train_dataset = dataset['train'].values.tolist()
                np.random.shuffle(train_dataset)

            task_dir = os.path.join(args.output_dir, task)
            setting_dir = os.path.join(task_dir, f"{args.k}-{seed}")
            os.makedirs(setting_dir, exist_ok=True)

            if task in ["SST-2", "MRPC", "QQP", "STS-B", "MNLI", "SNLI", "QNLI", "RTE", "WNLI", "CoLA"]:
                for split, samples in dataset.items():
                    if split.startswith("train"):
                        continue
                    splits = split.replace('dev', 'test')
                    
                    test_header = samples[0:1] if task != "CoLA" else []
                    test_dataset = samples[1:]
                    if len(test_dataset) > 1000:
                        np.random.seed(seed)
                        np.random.shuffle(test_dataset)
                        test_dataset = test_dataset[:1000]
                    with open(os.path.join(setting_dir, f"{splits}.tsv"), "w") as file:
                        for header in test_header:
                            file.write(header)
                        for sample in test_dataset:
                            file.write(sample)
            else:
                test_dataset = dataset['test']
                if len(test_dataset.index) > 1000:
                    test_dataset = test_dataset.sample(n=1000, random_state=seed)
                test_dataset.to_csv(os.path.join(setting_dir, 'test.csv'), header=False, index=False)

            sample_label = {}
            for sample in train_dataset:
                label = get_sample_label(task, sample)
                if label not in sample_label:
                    sample_label[label] = [sample]
                else:
                    sample_label[label].append(sample)

            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                with open(os.path.join(setting_dir, "train.tsv"), "w") as file:
                    for header in train_header:
                        file.write(header)
                    for label in sample_label:
                        for sample in sample_label[label][:args.k]:
                            file.write(sample)
                name = "dev.tsv"
                if task == 'MNLI':
                    name = "dev_matched.tsv"
                with open(os.path.join(setting_dir, name), "w") as file:
                    for header in train_header:
                        file.write(header)
                    for label in sample_label:
                        for sample in sample_label[label][args.k:2 * args.k]:
                            file.write(sample)
            else:
                train_new = []
                for label in sample_label:
                    for sample in sample_label[label][:args.k]:
                        train_new.append(sample)
                train_new = DataFrame(train_new)
                train_new.to_csv(os.path.join(setting_dir, 'train.csv'), header=False, index=False)

                dev_new = []
                for label in sample_label:
                    for sample in sample_label[label][args.k:2 * args.k]:
                        dev_new.append(sample)
                dev_new = DataFrame(dev_new)
                dev_new.to_csv(os.path.join(setting_dir, 'dev.csv'), header=False, index=False)


if __name__ == "__main__":
    main()
