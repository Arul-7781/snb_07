import argparse
import os
import random
from pathlib import Path
from urllib.request import urlopen


DEFAULT_TRAIN_META = (
    'https://raw.githubusercontent.com/XPixelGroup/BasicSR/master/'
    'basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt'
)
DEFAULT_TEST_META = (
    'https://raw.githubusercontent.com/XPixelGroup/BasicSR/master/'
    'basicsr/data/meta_info/meta_info_Vimeo90K_test_GT.txt'
)


def read_lines(path_or_url):
    if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
        with urlopen(path_or_url) as response:
            content = response.read().decode('utf-8')
        return content.splitlines()
    with open(path_or_url, 'r') as f:
        return f.read().splitlines()


def parse_meta_ids(path_or_url):
    lines = read_lines(path_or_url)
    ids = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        item = line.split()[0].strip()
        if item:
            ids.append(item)
    return ids


def write_list(file_path, ids):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write('\n'.join(ids) + '\n')


def build_cover_secret_list(test_ids, num_pairs, num_secret):
    if len(test_ids) < num_pairs * (num_secret + 1):
        raise ValueError(
            'Not enough test ids: need at least {}, found {}.'.format(
                num_pairs * (num_secret + 1), len(test_ids)
            )
        )

    covers = test_ids[:num_pairs]
    pointer = num_pairs
    flat_secrets = []
    for cover_id in covers:
        secrets = []
        while len(secrets) < num_secret:
            if pointer >= len(test_ids):
                pointer = 0
            candidate = test_ids[pointer]
            pointer += 1
            if candidate == cover_id:
                continue
            secrets.append(candidate)
        flat_secrets.extend(secrets)

    output = list(covers)
    output.extend(flat_secrets)
    return output


def check_sequences_exist(ids, data_root, min_frames):
    root = Path(data_root)
    missing = []
    too_short = []
    for item in ids:
        seq_path = root / item
        if not seq_path.exists() or not seq_path.is_dir():
            missing.append(item)
            continue
        frame_count = len([p for p in seq_path.iterdir() if p.suffix.lower() == '.png'])
        if frame_count < min_frames:
            too_short.append((item, frame_count))
    return missing, too_short


def main():
    parser = argparse.ArgumentParser(description='Build LF-VSN-compatible Vimeo90K list files.')
    parser.add_argument('--train-meta', default=DEFAULT_TRAIN_META,
                        help='Path or URL to Vimeo90K train meta file.')
    parser.add_argument('--test-meta', default=DEFAULT_TEST_META,
                        help='Path or URL to Vimeo90K test meta file.')
    parser.add_argument('--output-dir', default='vimeo90k',
                        help='Directory where list files are written.')
    parser.add_argument('--num-test-pairs', type=int, default=200,
                        help='Number of cover sequences in test list.')
    parser.add_argument('--test-secrets', nargs='+', type=int, default=[1, 7],
                        help='Secret counts to generate test lists for (e.g., 1 7).')
    parser.add_argument('--shuffle-train', action='store_true',
                        help='Shuffle train IDs before writing sep_trainlist.txt.')
    parser.add_argument('--seed', type=int, default=10,
                        help='Seed used when --shuffle-train is enabled.')
    parser.add_argument('--data-root', default=None,
                        help='Optional sequences root for integrity check (e.g., vimeo90k/sequences).')
    parser.add_argument('--min-frames', type=int, default=7,
                        help='Minimum png frames expected per sequence when --data-root is set.')
    args = parser.parse_args()

    train_ids = parse_meta_ids(args.train_meta)
    test_ids = parse_meta_ids(args.test_meta)
    if not train_ids:
        raise ValueError('Parsed 0 training IDs from {}'.format(args.train_meta))
    if not test_ids:
        raise ValueError('Parsed 0 testing IDs from {}'.format(args.test_meta))

    if args.shuffle_train:
        random.seed(args.seed)
        random.shuffle(train_ids)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / 'sep_trainlist.txt'
    write_list(train_path, train_ids)

    created = [str(train_path)]
    for num_secret in sorted(set(args.test_secrets)):
        if num_secret < 1:
            continue
        lines = build_cover_secret_list(test_ids, args.num_test_pairs, num_secret)
        if num_secret == 1:
            out_name = 'sep_testlist.txt'
        else:
            out_name = 'sep_testlist_{}video.txt'.format(num_secret)
        out_path = output_dir / out_name
        write_list(out_path, lines)
        created.append(str(out_path))

    print('Created list files:')
    for path in created:
        print('  {}'.format(path))

    if args.data_root is not None:
        all_check_ids = set(train_ids)
        for num_secret in sorted(set(args.test_secrets)):
            lines = build_cover_secret_list(test_ids, args.num_test_pairs, num_secret)
            all_check_ids.update(lines)

        missing, too_short = check_sequences_exist(sorted(all_check_ids), args.data_root, args.min_frames)
        print('Integrity check against {}:'.format(args.data_root))
        print('  Missing sequences: {}'.format(len(missing)))
        print('  Too short sequences (< {} frames): {}'.format(args.min_frames, len(too_short)))
        if missing:
            print('  First missing: {}'.format(missing[0]))
        if too_short:
            print('  First too short: {} ({})'.format(too_short[0][0], too_short[0][1]))


if __name__ == '__main__':
    main()
