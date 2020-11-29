# Rethinking Text Segmentation: A Novel Dataset and A Text-Specific Refinement Approach



**Note:**
Our dataset and code will be released here, stay tuned.

## Introduction
Text in the real world is extremely diverse, yet current text dataset does not reflect such diversity very well. To bridge this gap, we proposed TextSeg, a large-scale fine-annotated and multi-purpose text dataset, collecting scene and design text with six types of annotations: word- and character-wise bounding polygons, masks and transcriptions. We also introduce Text Refinement Network (TexRNet), a novel text segmentation approach that adapts to the unique properties of text, e.g. non-convex boundary, diverse texture, etc., which often impose burdens on traditional segmentation models. TexRNet refines results from common segmentation approach via key features pooling and attention, so that wrong-activated text regions can be adjusted. We also introduce trimap and discriminator losses that show significant improvement on text segmentation. 

## TextSeg Dataset

### Image Collection

<p align="center">
  <img src=".figure/image_only.jpg" width="99%">
</p>

### Annotation

<p align="center">
  <img src=".figure/image_anno.jpg" width="65%">
</p>

## TexRNet Structure and Results

<p align="center">
  <img src=".figure/network.jpg" width="90%">
</p>

In this table, we report the performance of our TexRNet on 5 text segmentation dataset including ours. 

<font size="11" face="Courier New">
<table>
  <tr>
    <td></td>
    <td colspan="2">TextSeg(Ours)</td>
    <td colspan="2">ICDAR13 FST</td>
    <td colspan="2">COCO_TS</td>
    <td colspan="2">MLT_S</td>
    <td colspan="2">Total-Text</td>
  </tr>
  <tr>
    <td>Method</td>
    <td>fgIoU</td><td>F-score</td>
    <td>fgIoU</td><td>F-score</td>
    <td>fgIoU</td><td>F-score</td>
    <td>fgIoU</td><td>F-score</td>
    <td>fgIoU</td><td>F-score</td>
  </tr>
  <tr>
    <td>DeeplabV3+</td>
    <td>84.07</td><td>0.914</td>
    <td>69.27</td><td>0.802</td>
    <td>72.07</td><td>0.641</td>
    <td>84.63</td><td>0.837</td>
    <td>74.44</td><td>0.824</td>
  </tr>
  <tr>
    <td>HRNetV2-W48</td>
    <td>85.03</td><td>0.914</td>
    <td>70.98</td><td>0.822</td>
    <td>68.93</td><td>0.629</td>
    <td>83.26</td><td>0.836</td>
    <td>75.29</td><td>0.825</td>
  </tr>
  <tr>
    <td>HRNetV2-W48 + OCR</td>
    <td>85.98</td><td>0.918</td>
    <td>72.45</td><td>0.830</td>
    <td>69.54</td><td>0.627</td>
    <td>83.49</td><td>0.838</td>
    <td>76.23</td><td>0.832</td>
  </tr>
  <tr>
    <td>Ours: TexRNet + DeeplabV3+</td>
    <td>   86.06    </td><td>   0.921    </td>
    <td>   72.16    </td><td>   0.835    </td>
    <td><b>73.98</b></td><td><b>0.722</b></td>
    <td><b>86.31</b></td><td>   0.830    </td>
    <td>   76.53    </td><td>   0.844    </td>
  </tr>
  <tr>
    <td>Ours: TexRNet + HRNetV2-W48</td>
    <td><b>86.84</b></td><td><b>0.924</b></td>
    <td><b>73.38</b></td><td><b>0.850</b></td>
    <td>   72.39    </td><td>   0.720    </td>
    <td>   86.09    </td><td><b>0.865</b></td>
    <td><b>78.47</b></td><td><b>0.848</b></td>
  </tr>
</table>
</font>
