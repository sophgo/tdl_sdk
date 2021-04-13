from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation', metavar='annotation', help='Annotations json file path')
    parser.add_argument('result', metavar='result', help='results json path')
    parser.add_argument('--catids', metavar='catids', help='category ids', default=None, type=int, nargs='+')
    args = parser.parse_args()

    coco_gt = COCO(args.annotation)
    coco_dt = coco_gt.loadRes(args.result)
    # img_ids = sorted(coco_gt.getImgIds())
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    if args.catids:
        coco_eval.params.catIds = args.catids
    # coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    print(args.result)
    coco_eval.summarize()
