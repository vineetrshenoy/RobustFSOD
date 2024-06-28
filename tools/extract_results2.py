import os
import math
import argparse
import numpy as np
import json
import csv
from tabulate import tabulate
from natsort import natsorted


def main(args):
    

    wf = open(os.path.join(args.res_dir, 'results.txt'), 'w')

    for shot in args.shot_list:

        file_paths = []
        for fid, fname in enumerate(os.listdir(args.res_dir)):
            if fname.split('_')[0] != '{}shot'.format(shot):
                continue
            _dir = os.path.join(args.res_dir, fname)
            if not os.path.isdir(_dir):
                continue
            file_paths.append(os.path.join(_dir, 'log.txt'))

        header, results = [], []
        for fid, fpath in enumerate(sorted(file_paths)):
            lineinfos = open(fpath).readlines()
            if fid == 0:
                res_info = lineinfos[-2].strip()
                header = res_info.split(':')[-1].split(',')
            res_info = lineinfos[-1].strip()
            results.append([fid] + [float(x) for x in res_info.split(':')[-1].split(',')])

        results_np = np.array(results)
        avg = np.mean(results_np, axis=0).tolist()
        cid = [1.96 * s / math.sqrt(results_np.shape[0]) for s in np.std(results_np, axis=0)]
        results.append(['μ'] + avg[1:])
        results.append(['c'] + cid[1:])

        table = tabulate(
            results,
            tablefmt="pipe",
            floatfmt=".2f",
            headers=[''] + header,
            numalign="left",
        )

        wf.write('--> {}-shot\n'.format(shot))
        wf.write('{}\n\n'.format(table))
        wf.flush()
    wf.close()

    print('Reformat all results -> {}'.format(os.path.join(args.res_dir, 'results.txt')))


def get_results(args):
    
    
    with open(os.path.join(args.res_dir, 'results.csv'), "wt") as fp:
        writer = csv.writer(fp, delimiter=",")
        for shot in args.shot_list:
        
            file_paths = []
            not_used = []
            seeds = []
            for fid, fname in enumerate(natsorted(os.listdir(args.res_dir))):
                if fname.split('_')[0] != '{}shot'.format(shot):
                    continue
                seed = int(os.path.join(args.res_dir, fname)[-1])
                _dir = os.path.join(args.res_dir, fname, 'inference')
                if not os.path.isdir(_dir):
                    not_used.append(_dir)
                    continue
                seeds.append(seed)
                file_paths.append(os.path.join(_dir, 'res_final.json'))
        
            header, results = [], []
            results_np = np.ones((1, 9))
            for fid, fpath in enumerate(zip(seeds, file_paths)):
                sd, filep = fpath 
                with open(filep, 'r') as f:
                    inf = json.load(f)
                if fid == 0:
                    keys = list(inf['bbox'].keys())
                    csv_keys = ['-']  + keys
                    writer.writerows([csv_keys])
                
                results = [round(inf['bbox'][key], 3) for key in keys]
                results_np = np.vstack((results_np, np.array(results)))
                results = [sd] + results
                writer.writerows([results])
    
            results_np = results_np[1:, :]
            avg = np.round(np.mean(results_np, axis=0), 3).tolist()
            cid = [np.round(1.96 * s / math.sqrt(results_np.shape[0]), 3) for s in np.std(results_np, axis=0)]
            
            avg = ['mu'] + avg
            cid = ['z'] + cid
            writer.writerows([avg])
            writer.writerows([cid])
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--res-dir', type=str, default='', help='Path to the results')
    parser.add_argument('--shot-list', type=int, nargs='+', default=[10], help='')
    args = parser.parse_args()
    
    get_results(args)
    #main(args)
