# Failure analysis

Success criterion: IoU >= 0.90 between detected and ground-truth page outline.
Failures: 2 of 12 annotated photos.

The *likely cause* lines were generated automatically from the geometry of the detection and the recorded conditions. Check each one against the image in `figures/fig3_failure_gallery.png` (red = detected, green = ground truth) and edit the text below by hand.

## sample_06.jpg

- **Method:** minarearect
- **IoU:** 0.800
- **Conditions:** angle=tilted, lighting=glare, background=plain, page=flat
- **Likely cause (auto):** detected quad is larger than the page (125%): page edge merged with a background object, shadow or table edge; fell back to minAreaRect: the outline was not a clean 4-sided polygon, and a rotated rectangle cannot follow perspective; glare: over-exposed area washes out part of the page border
- **Notes (manual):** _TODO: confirm or correct the cause; suggest a fix._

## sample_11.jpg

- **Method:** minarearect
- **IoU:** 0.714
- **Conditions:** angle=steep, lighting=shadow, background=cluttered, page=flat
- **Likely cause (auto):** detected quad is larger than the page (140%): page edge merged with a background object, shadow or table edge; fell back to minAreaRect: the outline was not a clean 4-sided polygon, and a rotated rectangle cannot follow perspective; steep angle: far edge is short, blurry and strongly foreshortened; shadow: the shadow boundary is a strong edge that can be mistaken for the page border; cluttered background: objects touching the page break or extend its outline
- **Notes (manual):** _TODO: confirm or correct the cause; suggest a fix._
