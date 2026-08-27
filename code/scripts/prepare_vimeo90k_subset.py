import argparse
import os
import random
from pathlib import Path


def read_id_list(file_path):
    with open(file_path, 'r') as f:
        ids = [line.strip() for line in f.readlines()]
    ids = [item for item in ids if item]
    return ids


def write_id_list(file_path, ids):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write('\n'.join(ids) + '\n')


def unique_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def scan_sequences(data_root, min_frames):
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError('data_root does not exist: {}'.format(data_root))
    sequence_ids = []
    for level_1 in sorted(root.iterdir()):
        if not level_1.is_dir():
            continue
        for level_2 in sorted(level_1.iterdir()):
            if not level_2.is_dir():
                continue
            png_count = len([p for p in level_2.iterdir() if p.suffix.lower() == '.png'])
            if png_count >= min_frames:
                rel = level_2.relative_to(root)
                sequence_ids.append(str(rel).replace(os.sep, '/'))
    return sequence_ids


def filter_existing_with_min_frames(sequence_ids, data_root, min_frames):
    root = Path(data_root)
    filtered = []
    for seq_id in sequence_ids:
        seq_path = root / seq_id
        if not seq_path.exists() or not seq_path.is_dir():
            continue
        png_count = len([p for p in seq_path.iterdir() if p.suffix.lower() == '.png'])
        if png_count >= min_frames:
            filtered.append(seq_id)
    return filtered


def main():
    parser = argparse.ArgumentParser(description='Prepare Vimeo90K prototype subset lists.')
    parser.add_argument('--data-root', required=True,
                        help='Path to Vimeo90K sequences directory (e.g., vimeo90k/sequences).')
    parser.add_argument('--train-list', default=None,
                        help='Optional existing train list path (e.g., vimeo90k/tri_trainlist.txt).')
    parser.add_argument('--test-list', default=None,
                        help='Optional existing test list path (e.g., vimeo90k/tri_testlist.txt).')
    parser.add_argument('--train-out', default='vimeo90k/sep_trainlist_proto.txt',
                        help='Output path for prototype train list.')
    parser.add_argument('--test-out', default='vimeo90k/sep_testlist_proto.txt',
                        help='Output path for prototype val/test list.')
    parser.add_argument('--num-train', type=int, default=15000,
                        help='Number of training sequences for prototype.')
    parser.add_argument('--num-val-pairs', type=int, default=200,
                        help='Number of cover samples for validation/testing.')
    parser.add_argument('--num-secret', type=int, default=1,
                        help='Number of secret sequences per cover sample.')
    parser.add_argument('--min-frames', type=int, default=3,
                        help='Minimum number of png frames required per sequence.')
    parser.add_argument('--seed', type=int, default=10,
                        help='Random seed for reproducible subset generation.')
    args = parser.parse_args()

    random.seed(args.seed)

    has_train_list = args.train_list is not None
    has_test_list = args.test_list is not None

    if has_train_list and has_test_list:
        train_pool = read_id_list(args.train_list)
        test_pool = read_id_list(args.test_list)
    elif has_test_list:
        test_pool = read_id_list(args.test_list)
        all_ids = scan_sequences(args.data_root, args.min_frames)
        test_set = set(test_pool)
        train_pool = [seq_id for seq_id in all_ids if seq_id not in test_set]
    elif has_train_list:
        train_pool = read_id_list(args.train_list)
        all_ids = scan_sequences(args.data_root, args.min_frames)
        train_set = set(train_pool)
        test_pool = [seq_id for seq_id in all_ids if seq_id not in train_set]
    else:
        all_ids = scan_sequences(args.data_root, args.min_frames)
        random.shuffle(all_ids)
        split_index = int(0.9 * len(all_ids))
        train_pool = all_ids[:split_index]
        test_pool = all_ids[split_index:]

    required_val_items = args.num_val_pairs * (args.num_secret + 1)

    train_pool = unique_keep_order(train_pool)
    test_pool = unique_keep_order(test_pool)
    train_pool = filter_existing_with_min_frames(train_pool, args.data_root, args.min_frames)
    test_pool = filter_existing_with_min_frames(test_pool, args.data_root, args.min_frames)

    if len(train_pool) == 0 and has_test_list and not has_train_list:
        fallback_pool = list(test_pool)
        random.shuffle(fallback_pool)
        if len(fallback_pool) <= required_val_items:
            raise ValueError(
                'Not enough sequences to split test-only data into train and val. '
                'Need more than {}, found {}.'.format(required_val_items, len(fallback_pool))
            )
        split_index = len(fallback_pool) - required_val_items
        train_pool = fallback_pool[:split_index]
        test_pool = fallback_pool[split_index:]
        print('Info: train list not provided and all sequences overlap with test list.')
        print('Info: split test list into train pool ({}) and val pool ({}).'.format(
            len(train_pool), len(test_pool)
        ))

    if len(train_pool) < args.num_train:
        raise ValueError(
            'Not enough training sequences. Requested {}, found {}. '
            'Please decrease --num-train.'.format(args.num_train, len(train_pool))
        )

    if len(test_pool) < required_val_items:
        raise ValueError(
            'Not enough validation/testing sequences. Requested at least {}, found {}.'.format(
                required_val_items, len(test_pool)
            )
        )

    random.shuffle(train_pool)
    random.shuffle(test_pool)

    train_subset = train_pool[:args.num_train]
    val_subset = test_pool[:required_val_items]
    cover_ids = val_subset[:args.num_val_pairs]
    secret_flat = val_subset[args.num_val_pairs:]

    test_lines = list(cover_ids)
    for pair_idx in range(args.num_val_pairs):
        for secret_idx in range(args.num_secret):
            secret_id = secret_flat[pair_idx * args.num_secret + secret_idx]
            if secret_id == cover_ids[pair_idx]:
                raise ValueError('A cover id and secret id are identical at index {}.'.format(pair_idx))
            test_lines.append(secret_id)

    write_id_list(args.train_out, train_subset)
    write_id_list(args.test_out, test_lines)

    print('Prepared prototype lists successfully.')
    print('  train_out: {} ({} samples)'.format(args.train_out, len(train_subset)))
    print('  test_out: {} ({} covers, {} secrets each)'.format(
        args.test_out, args.num_val_pairs, args.num_secret
    ))


if __name__ == '__main__':
    main()
