# References

IEEE style. Online sources were accessed on 21 Nov. 2026. Please open each link
once yourself before submitting to confirm it still works.

## Tutorial code that this project adapts

These must be cited because parts of the code follow their structure:

| Our code | What was adapted | Source |
|---|---|---|
| `scanner/transform.py` → `four_point_transform()` | Computing the output width/height from the longer pair of opposite edges, building the destination rectangle, `getPerspectiveTransform` + `warpPerspective` | [2], [3] |
| `scanner/transform.py` → `order_points()` | The idea of ordering corners TL, TR, BR, BL. **Modified:** we sort by angle around the centre instead of using x+y / y−x sums, which avoids ties for pages rotated about 45° | [2] |
| `scanner/preprocess.py`, `scanner/detect.py` | Overall document-scanner recipe: resize to 500 px height, grayscale, 5×5 Gaussian blur, Canny (75, 200), top-5 contours by area, `approxPolyDP` with 2 % of the perimeter, accept the first 4-point contour | [1] |
| `scanner/enhance.py` | `adaptiveThreshold` with the Gaussian method | [4] |

Our own additions (not from the tutorials): the edge-closing step, the convexity,
area and perimeter-ratio checks, the Otsu and `minAreaRect` fallback chain with
method tracking, the polygon IoU evaluation, the synthetic test-image generator,
the annotation tool, the failure-analysis generator, the CER edit-distance code
and the Streamlit app.

## Reference list

[1] A. Rosebrock, "How to build a kick-ass mobile document scanner in just 5 minutes," *PyImageSearch*, Sep. 1, 2014. [Online]. Available: https://pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/

[2] A. Rosebrock, "4 point OpenCV getPerspective Transform example," *PyImageSearch*, Aug. 25, 2014. [Online]. Available: https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

[3] OpenCV, "Geometric transformations of images," *OpenCV Documentation*. [Online]. Available: https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html

[4] OpenCV, "Image thresholding," *OpenCV Documentation*. [Online]. Available: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

[5] OpenCV, "Canny edge detection," *OpenCV Documentation*. [Online]. Available: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

[6] OpenCV, "Contour features," *OpenCV Documentation*. [Online]. Available: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html

[7] OpenCV, "Morphological transformations," *OpenCV Documentation*. [Online]. Available: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

[8] J. Canny, "A computational approach to edge detection," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. PAMI-8, no. 6, pp. 679–698, Nov. 1986.

[9] N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Trans. Syst., Man, Cybern.*, vol. SMC-9, no. 1, pp. 62–66, Jan. 1979.

[10] D. H. Douglas and T. K. Peucker, "Algorithms for the reduction of the number of points required to represent a digitized line or its caricature," *Cartographica*, vol. 10, no. 2, pp. 112–122, 1973.

[11] S. Suzuki and K. Abe, "Topological structural analysis of digitized binary images by border following," *Comput. Vis. Graph. Image Process.*, vol. 30, no. 1, pp. 32–46, 1985.

[12] R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed. New York, NY, USA: Pearson, 2018.

[13] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, "The PASCAL Visual Object Classes (VOC) challenge," *Int. J. Comput. Vis.*, vol. 88, no. 2, pp. 303–338, 2010. (Intersection over Union as a localisation metric.)

[14] V. I. Levenshtein, "Binary codes capable of correcting deletions, insertions, and reversals," *Soviet Physics Doklady*, vol. 10, no. 8, pp. 707–710, 1966.

[15] R. Smith, "An overview of the Tesseract OCR engine," in *Proc. 9th Int. Conf. Document Analysis and Recognition (ICDAR)*, Curitiba, Brazil, 2007, pp. 629–633.

[16] G. Bradski, "The OpenCV library," *Dr. Dobb's Journal of Software Tools*, vol. 25, no. 11, pp. 120–125, 2000.
