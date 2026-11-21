# Evaluation summary

- Photos processed: 12 (annotated with ground truth: 12)
- **Overall success rate (IoU >= 0.90): 10/12 = 83.3%**
- Mean IoU: 0.950 (median 0.989)
- Mean processing time: 6 ms per photo

## By detection method

| method | n | successes | success_rate | mean_iou |
|---|---|---|---|---|
| canny | 9 | 9 | 100% | 0.989 |
| otsu | 1 | 1 | 100% | 0.984 |
| minarearect | 2 | 0 | 0% | 0.757 |

## By condition

### angle

| angle | n | successes | success_rate | mean_iou |
|---|---|---|---|---|
| straight | 2 | 2 | 100% | 0.991 |
| tilted | 8 | 7 | 88% | 0.964 |
| steep | 2 | 1 | 50% | 0.852 |

### lighting

| lighting | n | successes | success_rate | mean_iou |
|---|---|---|---|---|
| good | 8 | 8 | 100% | 0.987 |
| dim | 1 | 1 | 100% | 0.993 |
| shadow | 2 | 1 | 50% | 0.854 |
| glare | 1 | 0 | 0% | 0.800 |

### background

| background | n | successes | success_rate | mean_iou |
|---|---|---|---|---|
| plain | 9 | 8 | 89% | 0.970 |
| cluttered | 2 | 1 | 50% | 0.842 |
| low-contrast | 1 | 1 | 100% | 0.984 |

### page

| page | n | successes | success_rate | mean_iou |
|---|---|---|---|---|
| flat | 10 | 8 | 80% | 0.942 |
| curled | 1 | 1 | 100% | 0.991 |
| folded | 1 | 1 | 100% | 0.990 |

