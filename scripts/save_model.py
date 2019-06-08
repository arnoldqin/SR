# coding: utf-8
import os, fnmatch
matches = []
for root, dirname, filenames in os.walk('.'):
        for filename in fnmatch.filter(filenames, 'latest*'):
            matches.append(os.path.join(root, filename))
for root, dirname, filenames in os.walk('.'):
        for filename in fnmatch.filter(filenames, 'opt.txt'):
            matches.append(os.path.join(root, filename))
for root, dirname, filenames in os.walk('.'):
        for filename in fnmatch.filter(filenames, '*log.txt'):
            matches.append(os.path.join(root, filename))
for root, dirname, filenames in os.walk('.'):
    if 'web' in root:
        for filename in fnmatch.filter(filenames, '*'):
            matches.append(os.path.join(root, filename))

import tarfile
import tqdm
with tarfile.open('model.tar', 'w') as tar:
    for i in tqdm.tqdm(matches):
        tar.add(i)
