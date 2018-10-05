import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import os
import shutil
import sys

import src.utils.utils as ut
import src.data.crop as cr

from sklearn.model_selection import train_test_split


class prepare_datasets:
    def __init__(self, split, seed):
        self.split = split
        self.seed = seed

    def create_folders(self):

        dirs = []

        train_dir = ut.dirs.train_dir
        validation_dir = ut.dirs.validation_dir
        test_dir = ut.dirs.test_dir

        dirs.append(train_dir)
        dirs.append(validation_dir)
        dirs.append(test_dir)

        train_good_dir = os.path.join(train_dir, "good")
        dirs.append(train_good_dir)
        train_bad_dir = os.path.join(train_dir, "bad")
        dirs.append(train_bad_dir)

        validation_good_dir = os.path.join(validation_dir, "good")
        dirs.append(validation_good_dir)
        validation_bad_dir = os.path.join(validation_dir, "bad")
        dirs.append(validation_bad_dir)

        test_good_dir = os.path.join(test_dir, "good")
        dirs.append(test_good_dir)
        test_bad_dir = os.path.join(test_dir, "bad")
        dirs.append(test_bad_dir)

        for directory in dirs:
            if not os.path.exists(directory):
                os.mkdir(directory)

    def clean(self):
        shutil.rmtree(ut.dirs.train_dir)
        shutil.rmtree(ut.dirs.validation_dir)
        shutil.rmtree(ut.dirs.test_dir)

    def get_filenames(self, data_dir=ut.dirs.original_dataset_dir,
                      subdir="good"):

        src_dir = os.path.join(data_dir, subdir)
        filenames = []

        for file in Path(src_dir).iterdir():
            if(file.name.endswith(('.jpeg', '.jpg'))):
                filenames.append(file.name)

        return filenames

    def copy(self, filenames, src_data_dir=ut.dirs.original_dataset_dir,
             dst_data_dir=ut.dirs.train_dir, subdir="good"):
        src_dir = os.path.join(src_data_dir, subdir)
        dst_dir = os.path.join(dst_data_dir, subdir)
        counter = 0
        corrupted = 0
        for name in filenames:
            src = os.path.join(src_dir, name)
            dst = os.path.join(dst_dir, name)
            if(os.path.exists(src) and (os.stat(src).st_size != 0)):
                if(os.path.exists(dst)):
                    pass
                else:
                    shutil.copy(src, dst)
                    counter += 1
            elif(os.path.exists(src) and (os.stat(src).st_size == 0)):
                print('...  found corrupted image', os.path.exists(src), src)
                corrupted += 1
        if(counter != 0):
            print('... copied', counter, 'files from', src_dir, 'to', dst_dir)
            if(corrupted != 0):
                print('... thereby found', corrupted, 'files')
        if(counter == 0 and corrupted == 0):
            print('... datasets already created earlier')

    def make_split(self, data=None):
        if(data is not None and len(data) != 0):
            train_tmp, test = train_test_split(data,
                                               test_size=self.split,
                                               random_state=self.seed)
            split_tmp = len(test)/len(train_tmp)
            train, valid = train_test_split(train_tmp,
                                            test_size=split_tmp,
                                            random_state=self.seed)

            assert len(valid) == len(test)
            assert len(test) + len(valid) + len(train) == len(data)

            return train, valid, test
        else:
            print("data set is not specified")
            print("stop execution")
            sys.exit()

    def make_datasets(self, original_dir=ut.dirs.original_dataset_dir):
        self.create_folders()
        for feature in ["good", "bad"]:
            nails = self.get_filenames(original_dir, feature)
            train, valid, test = self.make_split(nails)

            sample_names = [train, valid, test]
            sample_dirs = [ut.dirs.train_dir,
                           ut.dirs.validation_dir,
                           ut.dirs.test_dir]

            for names, dirs in zip(sample_names, sample_dirs):
                self.copy(names, original_dir, dirs, feature)

@click.command()
@click.option('--split', type=float, default=0.12,
                    help='validation and test split (default: 0.12)')
@click.option('--seed', type=int, default=42,
                    help='random seed (default: 42)')
@click.option('--clean', type=int, default=0,
                    help='delete training, validation and test data\n\
                     before creating new\n\
                     0: False, 1: True (default: 0)')
@click.option('--crop', type=int, default=1,
                    help='crop training (train, valid, test) \
                     data to show only the targets\n\
                     0: False, 1: True (default: 1)')
def main(split, seed, crop, clean):
    if(split > 1/3):
        print('data split not possible')
        print('split ratio too large')
        sys.exit()

    prep_d = prepare_datasets(split, seed)
    if(clean == 1):
        print('clean training data')
        prep_d.clean()

    print('creating training data')
    prep_d.make_datasets()

    if(crop == 1):
        print('cropping training data')
        cr.crop()

    print('\nDone')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
