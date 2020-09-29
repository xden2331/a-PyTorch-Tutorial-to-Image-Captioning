from utils import create_input_files

base_path = 'drive/My Drive/NLP/a-PyTorch-Tutorial-to-Image-Captioning'

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path=base_path + '/caption_datasets/dataset_coco.json',
                       image_folder=base_path,
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=base_path + '/caption data',
                       max_len=50)
