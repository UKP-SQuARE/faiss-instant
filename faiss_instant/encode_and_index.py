import argparse
from . import encode, index
import crash_ipdb


def run(
    # encode 
    input_file, 
    output_dir, 
    model_type, 
    model_name_or_path, 
    normalize=False, 
    chunk_size=160000, 
    batch_size_per_gpu=32,

    # index
    index_factory_string=None, 
    distance='IP', 
    nprobe=None,
):
    print('Doing encoding')
    encode.run(
        input_file, 
        output_dir, 
        model_type, 
        model_name_or_path, 
        normalize, 
        chunk_size, 
        batch_size_per_gpu
    )

    print('Doing indexing')
    index.run(
        output_dir, 
        output_dir, 
        index_factory_string, 
        distance, 
        nprobe
    )

    print(f'{__name__}: Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--output_dir')
    parser.add_argument('--model_type')
    parser.add_argument('--model_name_or_path')
    parser.add_argument('--normalize', action='store_true', help='Set this flag if cosine-similarity will be used')
    parser.add_argument('--chunk_size', type=int, default=160000)
    parser.add_argument('--batch_size_per_gpu', type=int, default=32)
    
    parser.add_argument('--index_factory_string', default=None, help='By default, it will use IVFSQ index')
    parser.add_argument('--distance', default='IP', choices=['L2', 'IP'])
    parser.add_argument('--nprobe', type=int, default=None)
    args = parser.parse_args()
    run(**vars(args))
