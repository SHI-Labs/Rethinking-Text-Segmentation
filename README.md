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

<table>
  <tr>
    <td></td>
    <td colspan="2" text-align=center>TextSeg(Ours)</td>
    <td colspan="2" style="text-align:center;">ICDAR13 FST</td>
    <td colspan="2" style="text-align:center;">COCO_TS</td>
    <td colspan="2" style="text-align:center;">MLT_S</td>
    <td colspan="2" style="text-align:center;">Total-Text</td>
  </tr>
  <tr>
    <td>Method</td>
    <td>fgIoU</td><td>F-score</td>
    <td>fgIoU</td><td>F-score</td>
    <td>fgIoU</td><td>F-score</td>
    <td>fgIoU</td><td>F-score</td>
    <td>fgIoU</td><td>F-score</td>
  </tr>
</table>
