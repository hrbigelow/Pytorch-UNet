# accept
import sys
import os
import numpy as np
import random
from collections import defaultdict
from mahotas.labeled import label
from PIL import Image


class Anno():
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.rl = []

    def img(self):
        d = np.zeros(self.h * self.w, dtype=np.uint8)
        for obj in self.rl:
            for beg, n in obj:
                d[beg-1:beg-1+n] = 255

        img = Image.fromarray(d.reshape(self.h, self.w), mode='L')
        return img




def parse_anno(anno_file):
    ann = {}
    with open(anno_file, 'r') as fh:
        next(fh) # discard header
        for line in fh:
            id, anno, w, h, _ = line.split(',', 4)
            w = int(w)
            h = int(h)
            anno = anno.split(' ')
            anno = list((int(a), int(b)) for a, b in zip(anno[0::2], anno[1::2]))
            if id not in ann:
                ann[id] = Anno(w, h)
            ann[id].rl.append(anno)

    return ann


def mask_to_anno(img_path):
    img = Image.open(img_path)
    dat = np.asarray(img)
    return _mask_to_anno(dat)


def _mask_to_anno(dat):
    h, w = dat.shape

    labels, numL = label(dat >= dat.mean())
    assert(dat.shape == labels.shape)
    
    rls = [[] for _ in range(numL+1)]
    for r in range(h):
        v = labels[r][0]
        if v != 0:
            # open a new region
            rls[v].append(r * w)
            st = 0

        pv = v
        for c in range(1, w):
            v = labels[r][c]
            if v != pv:
                if pv != 0:
                    # close out previous open region
                    rls[pv].append(c - st)
                if v != 0:
                    # open a new region
                    st = c
                    rls[v].append(r * w + c)
            pv = v

        if pv != 0:
            # close out the last region
            rls[pv].append(w - st)

    rls = [list(zip(rl[::2], rl[1::2])) for rl in rls[1:]]
    return rls, h, w


def anno_to_mask(rls, h, w):
    dat = _anno_to_mask(rls, h, w)
    return Image.fromarray(dat, mode='L')


def _anno_to_mask(rls, h, w):
    dat = np.zeros(h * w, dtype=np.uint8)
    for rl in rls:
        for pos, ct in rl:
            dat[pos-1:pos-1+ct] = 255 
    return dat.reshape(h, w)


def check_convert(dat):
    rls, h, w = _mask_to_anno(dat)
    dat_recons = _anno_to_mask(rls, h, w)
    assert dat == dat_recons


def gen_colored(mask):
    dat, labels = _gen_colored(mask)
    return Image.fromarray(dat, mode='RGB'), \
            Image.fromarray(labels.astype(np.uint8), mode='L')


def _gen_colored(mask):
    labels, numL = label((mask > mask.mean()))
    colors = np.random.randint(0, 255, (numL+1, 3), dtype=np.uint8)
    colors[0,:] = 0
    dat = np.take(colors, labels, 0)
    return dat, labels

def iou(act_ival, prd_ival):
    if act_ival[-1][0] <= prd_ival[0][0]: return 0
    if prd_ival[-1][0] <= act_ival[0][0]: return 0

    tot = sum(e - b for b, e in act_ival + prd_ival)
    ait, pit = iter(act_ival), iter(prd_ival)
    abeg, aend = next(ait)
    pbeg, pend = next(pit)
    ixn = 0
    while True:
        ixn += max(0, min(aend, pend) - max(abeg, pbeg))
        try:
            if aend < pend: abeg, aend = next(ait)
            else: pbeg, pend = next(pit)
        except StopIteration:
            break

    union = tot - ixn
    return ixn / union

def precision(act_rles, prd_rles, thresh):
    # one rle is [(beg, ct), (beg, ct), ...]
    act_ivals = [[(b, b+ct) for b, ct in rle] for rle in act_rles]
    prd_ivals = [[(b, b+ct) for b, ct in rle] for rle in prd_rles]

            
    act_hits = [0] * len(act_ivals)
    prd_hits = [0] * len(prd_ivals)

    for ai, act_ival in enumerate(act_ivals):
        for pi, prd_ival in enumerate(prd_ivals):
            if iou(act_ival, prd_ival) > thresh:
                act_hits[ai] += 1
                prd_hits[pi] += 1

    tp = len([h for h in act_hits if h == 1])
    fn = len(act_hits) - tp
    fp = len([h for h in prd_hits if h == 0])
    return tp / (tp + fp + fn)



if __name__ == '__main__':
    app = sys.argv[1]
    if app == 'convert':
        anno_file, out_dir = sys.argv[2:]
        annos = parse_anno(anno_file)
        ct = 0
        for id, a in annos.items():
            if ct % 100 == 0:
                print(f'processing {id} ({a.w} x {a.h})')
            ct += 1
            img = a.img()
            img.save(out_dir + '/' + id + '_mask.gif')

    elif app == 'rls':
        mask_dir, out_dir, out_file = sys.argv[2:]
        with open(out_file, 'w') as fh, os.scandir(mask_dir) as scan:
            print('id,predicted', file=fh)
            for mask_file in scan:
                img_id, _ = os.path.splitext(mask_file.name)
                img_path = os.path.join(mask_dir, mask_file.name)
                rls, h, w = mask_to_anno(img_path)
                mask_dat = _anno_to_mask(rls, h, w)
                colored_img, label_img = gen_colored(mask_dat)
                colored_img.save(os.path.join(out_dir, img_id + '_colored.png'))
                label_img.save(os.path.join(out_dir, img_id + '_labels.png'))

                for ent in rls:
                    ent_str = ' '.join(str(pos) + ' ' + str(ct) for pos, ct in ent)
                    out = '%s,%s' % (img_id, ent_str)
                    print(out, file=fh)

    elif app == 'eval':
        gt_mask_dir, pred_mask_dir, out_file = sys.argv[2:]
        run_lengths = defaultdict(lambda: {})
        for k, d in { 'gt': gt_mask_dir, 'pr': pred_mask_dir }.items():
            with os.scandir(d) as scan:
                for mask in scan:
                # for mask in list(scan)[:3]:
                    img_id, _ = os.path.splitext(mask.name)
                    img_id = img_id.split('_')[0]
                    img_path = os.path.join(d, mask.name)
                    rls, h, w = mask_to_anno(img_path)
                    run_lengths[img_id][k] = (rls, img_path)

        with open(out_file, 'w') as fh:
            print('<html><body>', file=fh)

            for img_id, d in run_lengths.items():
                print('processing ', img_id)
                if 'gt' not in d or 'pr' not in d:
                    print(f'skipping {img_id}', file=fh)
                    continue
                gts, gt_path = d['gt']
                prs, pr_path = d['pr']
                precs = [precision(gts, prs, th) 
                        for th in np.arange(0.5, 1.0, 0.05)]

                avg_prec = sum(precs) / len(precs)
                prec_str = ' '.join(f'{p:3.2f}' for p in precs)
                print(f'<img src="{gt_path}" />', file=fh)
                print(f'<img src="{pr_path}" />', file=fh)
                print(f'<p>{prec_str}</p>', file=fh)
                print(f'<p>{img_id}: {avg_prec:3.2f}</p>', file=fh)

            print('</body></html>', file=fh)

