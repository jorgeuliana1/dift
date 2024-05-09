from src.dataloader.loading import PadUfes20

# def get_dataset(args, config):
def get_dataset(dataset_name, dataroot, traindata, testdata)
    # if config.data.dataset == "PAD-UFES-20":
    if dataset_name == "PAD-UFES-20":
        train_dataset = PadUfes20(
            dataroot,
            # config.data.dataroot,
            csv_train=traindata,
            # csv_train=config.data.traindata,
            csv_test=testdata,
            # csv_test=config.data.testdata,
            train=True
        )
        test_dataset = PadUfes20(
            dataroot,
            # config.data.dataroot,
            csv_train=traindata,
            # csv_train=config.data.traindata,
            csv_test=testdata,
            # csv_test=config.data.testdata,
            train=False
        )
    else:
        raise NotImplementedError("Options: PAD-UFES-20.")
    return train_dataset, test_dataset