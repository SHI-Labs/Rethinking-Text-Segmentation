import torch

import os
import os.path as osp
import numpy as np
from PIL import Image

import argparse

from lib.model_zoo.texrnet import TexRNet
from lib.model_zoo.hrnet import HRNet_Base
from lib.model_zoo.deeplab import DeepLabv3p_Base
from lib.model_zoo.resnet import ResNet_Dilated_Base
from lib import torchutils

import tqdm

class TextRNet_HRNet_Wrapper(object):
    """
    This is the UltraSRWrapper with render-level batchification.
    """
    def __init__(self,
                 device,
                 pth=None,):
        """
        Create uspcale instance
        :param device: device on run the upscale pipeline (if GPU is accessible should be 'cuda')
        :param pth: path to model
        """
        self.model = self.make_model(pth)
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device

    @staticmethod
    def make_model(pth=None):
        backbone = HRNet_Base(
            oc_n=720, 
            align_corners=True, 
            ignore_label=999, 
            stage1_para={
                'BLOCK'       : 'BOTTLENECK',
                'FUSE_METHOD' : 'SUM',
                'NUM_BLOCKS'  : [4],
                'NUM_BRANCHES': 1,
                'NUM_CHANNELS': [64],
                'NUM_MODULES' : 1 },
            stage2_para={
                'BLOCK'       : 'BASIC',
                'FUSE_METHOD' : 'SUM',
                'NUM_BLOCKS'  : [4, 4],
                'NUM_BRANCHES': 2,
                'NUM_CHANNELS': [48, 96],
                'NUM_MODULES' : 1 },
            stage3_para={
                'BLOCK'       : 'BASIC',
                'FUSE_METHOD' : 'SUM',
                'NUM_BLOCKS'  : [4, 4, 4],
                'NUM_BRANCHES': 3,
                'NUM_CHANNELS': [48, 96, 192],
                'NUM_MODULES' : 4 },
            stage4_para={
                'BLOCK'       : 'BASIC',
                'FUSE_METHOD' : 'SUM',
                'NUM_BLOCKS'  : [4, 4, 4, 4],
                'NUM_BRANCHES': 4,
                'NUM_CHANNELS': [48, 96, 192, 384],
                'NUM_MODULES' : 3 },
            final_conv_kernel = 1,
        )
        
        model = TexRNet(
            bbn_name='hrnet',
            bbn=backbone,
            ic_n=720,
            rfn_c_n=[725, 64, 64],
            sem_n=2,
            conv_type='conv',
            bn_type='bn',
            relu_type='relu',
            align_corners=True,
            ignore_label=None,
            bias_att_type='cossim',
            ineval_output_argmax=False,
        )
        if pth is not None:
            paras = torch.load(pth, map_location=torch.device('cpu'))
            new_paras = model.state_dict()
            new_paras.update(paras)
            model.load_state_dict(new_paras)
        return model

    def process(self, pil_image):
        im = np.array(pil_image.convert("RGB"))
        im = im/255
        im = im - np.array([0.485, 0.456, 0.406])
        im = im / np.array([0.229, 0.224, 0.225])
        im = np.transpose(im, (2, 0, 1))[None]
        im = torch.FloatTensor(im).to(self.device)

        # This step will auto-adjust model if it is torch-DDP
        netm = getattr(self.model, 'module', self.model)
        _, _, oh, ow = im.shape
        ac = True

        prfnc_ms, pcount_ms = {}, {}

        for mstag, mssize in [
                ['0.75x', 385],
                ['1.00x', 513],
                ['1.25x', 641],
                ['1.50x', 769],
                ['1.75x', 897],
                ['2.00x', 1025],
                ['2.25x', 1153],
                ['2.50x', 1281], ]:
            # by area
            ratio = np.sqrt(mssize**2 / (oh*ow))
            th, tw = int(oh*ratio), int(ow*ratio)
            tw = tw//32*32+1
            th = th//32*32+1

            imi = {
                'nofp' : torchutils.interpolate_2d(
                    size=(th, tw), mode='bilinear', 
                    align_corners=ac)(im)}                    
            imi['flip'] = torch.flip(imi['nofp'], dims=[-1])

            for fliptag, imii in imi.items():
                with torch.no_grad():
                    pred = netm(imii)
                    psem = torchutils.interpolate_2d(
                        size=(oh, ow), 
                        mode='bilinear', align_corners=ac)(pred['predsem']) 
                    prfn = torchutils.interpolate_2d(
                        size=(oh, ow), 
                        mode='bilinear', align_corners=ac)(pred['predrfn']) 

                    if fliptag == 'flip':
                        psem = torch.flip(psem, dims=[-1])
                        prfn = torch.flip(prfn, dims=[-1])
                    elif fliptag == 'nofp':
                        pass
                    else:
                        raise ValueError
                
                try:
                    prfnc_ms[mstag]  += prfn
                    pcount_ms[mstag] += 1
                except:
                    prfnc_ms[mstag]  = prfn
                    pcount_ms[mstag] = 1

        pred = sum([pi for pi in prfnc_ms.values()])
        pred /= sum([ni for ni in pcount_ms.values()])
        pred = torch.argmax(psem, dim=1)
        pred = pred[0].cpu().detach().numpy()
        pred = (pred * 255).astype(np.uint8)
        return Image.fromarray(pred)

class TextRNet_Deeplab_Wrapper(TextRNet_HRNet_Wrapper):
    @staticmethod
    def make_model(pth=None):
        raise NotImplementedError
        # resnet = ResNet_Dilated_Base(
        #     block = 
        #     layer_n = 
        # )
        
        # model = TexRNet(
        #     bbn_name='hrnet',
        #     bbn=backbone,
        #     ic_n=720,
        #     rfn_c_n=[725, 64, 64],
        #     sem_n=2,
        #     conv_type='conv',
        #     bn_type='bn',
        #     relu_type='relu',
        #     align_corners=True,
        #     ignore_label=None,
        #     bias_att_type='cossim',
        #     ineval_output_argmax=False,
        # )
        # if pth is not None:
        #     paras = torch.load(pth, map_location=torch.device('cpu'))
        #     new_paras = model.state_dict()
        #     new_paras.update(paras)
        #     model.load_state_dict(new_paras)
        # return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input folder or a single input file")
    parser.add_argument("--output", type=str, required=True, help="output folder or a single output file")
    parser.add_argument("--method", '-m', type=str, default='textrnet_hrnet')
    args = parser.parse_args()

    if osp.isdir(args.input):
        if not osp.exists(args.output):
            os.makedirs(args.output)
        assert osp.isdir(args.output), \
            "When --input is a directory, --output must be a directory!"
    elif osp.isfile(args.input):
        assert not osp.isdir(args.output), \
            "When --input is a file, --output must be a file!"
    else:
        assert False, "No such input!"

    assert args.input != args.output, \
        "--input and --output points to the same location, "\
        "this is not allowed because it will override the input files."

    if args.method == 'textrnet_hrnet':
        wrapper = TextRNet_HRNet_Wrapper
        model_path = 'pretrained/texrnet_hrnet.pth'
    elif args.method == 'textrnet_deeplab':
        wrapper = TextRNet_Deeplab_Wrapper
        model_path = 'pretrained/texrnet_deeplab.pth'
    else:
        assert False, 'No such model.'

    enl = wrapper(torch.device("cuda:0"), model_path)

    if osp.isfile(args.input):
        imgs = [args.input]
        outs = [args.output]
    else:
        imgs = sorted(os.listdir(args.input))
        outs = [
            osp.join(args.output, '{}.png'.format(osp.splitext(fi)[0]))
                for fi in imgs
        ]
        imgs = [osp.join(args.input, fi) for fi in imgs]
    
    for fin, fout in tqdm(zip(imgs, outs), total=len(imgs)):
        x = Image.open(fin).convert('RGB')
        y = enl.process(x)
        y.save(fout)
