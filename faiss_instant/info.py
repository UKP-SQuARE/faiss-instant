import argparse
import json
import faiss
import os
import crash_ipdb


metric_map = {
    faiss.METRIC_L1: 'METRIC_L1',
    faiss.METRIC_L2: 'METRIC_L2',
    faiss.METRIC_Lp: 'METRIC_Lp',
    faiss.METRIC_BrayCurtis: 'METRIC_BrayCurtis',
    faiss.METRIC_INNER_PRODUCT: 'METRIC_INNER_PRODUCT',
    faiss.METRIC_JensenShannon: 'METRIC_JensenShannon',
    faiss.METRIC_Linf: 'METRIC_Linf'
}

def get_info(index: faiss.Index):
    info = {}
    for key in dir(index):
        value = getattr(index, key)
        if key == 'metric_type' and value in metric_map:
            value = metric_map[value]
        if type(value) in [int, float, str]:
            info[key] = value
    return info

def run(index_path, output_dir=None):
    info = {}
    if os.path.isdir(index_path):
        for fname in os.listdir(index_path):
            if not fname.endswith('.index'):
                continue
            index_path = os.path.join(index_path, fname)
            index = faiss.read_index(index_path)
            info[fname] = get_info(index)
    else:
        assert os.path.isfile(index_path)
        fname = os.path.basename(index_path)
        info[fname] = get_info(index)
    
    info_string = json.dumps(info, indent=4)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'info.json'), 'w') as f:
            f.write(info_string)
    
    print('Info:', info_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_path', help='Path to the index file or its parent folder')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    run(**vars(args))