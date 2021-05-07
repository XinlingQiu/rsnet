#!/usr/bin/env python3
import networkx as nx
import numpy as np
from numba import jit
import math
import numpy as np
from skimage.morphology import skeletonize
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import numpy as np
from numpy import ma
import skimage
import os
import cv2
import torch
from torch.utils.data import Dataset
from osgeo import gdal
from collections import OrderedDict
import os.path as osp
from pathlib import Path
import time
import argparse
from osgeo import gdal
from torch.nn.modules.utils import _pair
import copy
# get neighbors d index
def neighbors(shape):
    dim = len(shape)
    block = np.ones([3] * dim)
    block[tuple([1] * dim)] = 0
    idx = np.where(block > 0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx - [1] * dim)
    acc = np.cumprod((1,) + shape[::-1][:-1])
    return np.dot(idx, acc[::-1])
def scandir(dir_path, suffix=None, recursive=False):

    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must ne a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(entry.path,
                                        suffix=suffix,
                                        recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


@jit(nopython=True)  # my mark
def mark(img, nbs):  # mark the array use (0, 1, 2)
    img = img.ravel()
    for p in range(len(img)):
        if img[p] == 0:
            continue
        s = 0
        for dp in nbs:
            if img[p + dp] != 0:
                s += 1
        if s == 2:
            img[p] = 1
        else:
            img[p] = 2


@jit(nopython=True)  # trans index to r, c...
def idx2rc(idx, acc):
    rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i, j] = idx[i] // acc[j]
            idx[i] -= rst[i, j] * acc[j]
    rst -= 1
    return rst


@jit(nopython=True)  # fill a node (may be two or more points)
def fill(img, p, num, nbs, acc, buf):
    back = img[p]
    img[p] = num
    buf[0] = p
    cur = 0
    s = 1

    while True:
        p = buf[cur]
        for dp in nbs:
            cp = p + dp
            if img[cp] == back:
                img[cp] = num
                buf[s] = cp
                s += 1
        cur += 1
        if cur == s:
            break
    return idx2rc(buf[:s], acc)


# trace the edge and use a buffer, then buf.copy, if use [] numba not works
@jit(nopython=True)
def trace(img, p, nbs, acc, buf):
    c1 = 0
    c2 = 0
    newp = 0
    cur = 1
    while True:
        buf[cur] = p
        img[p] = 0
        cur += 1
        for dp in nbs:
            cp = p + dp
            if img[cp] >= 10:
                if c1 == 0:
                    c1 = img[cp]
                    buf[0] = cp
                else:
                    c2 = img[cp]
                    buf[cur] = cp
            if img[cp] == 1:
                newp = cp
        p = newp
        if c2 != 0:
            break
    return (c1 - 10, c2 - 10, idx2rc(buf[:cur + 1], acc))


@jit(nopython=True)  # parse the image then get the nodes and edges
def parse_struc(img, pts, nbs, acc):
    img = img.ravel()
    buf = np.zeros(131072, dtype=np.int64)
    num = 10
    nodes = []
    for p in pts:
        if img[p] == 2:
            nds = fill(img, p, num, nbs, acc, buf)
            num += 1
            nodes.append(nds)
    edges = []
    for p in pts:
        for dp in nbs:
            if img[p + dp] == 1:
                edge = trace(img, p + dp, nbs, acc, buf)
                edges.append(edge)
    return nodes, edges

# use nodes and edges build a networkx graph


def build_graph(nodes, edges, multi=False):
    graph = nx.MultiGraph() if multi else nx.Graph()
    for i in range(len(nodes)):
        graph.add_node(i, pts=nodes[i], o=nodes[i].mean(axis=0))
    for s, e, pts in edges:
        ln = np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum()
        graph.add_edge(s, e, pts=pts, weight=ln)
    return graph


def buffer(ske):
    buf = np.zeros(tuple(np.array(ske.shape) + 2), dtype=np.uint16)
    buf[tuple([slice(1, -1)] * buf.ndim)] = ske
    return buf


def mark_node(ske):
    buf = buffer(ske)
    nbs = neighbors(buf.shape)
    acc = np.cumprod((1,)+buf.shape[::-1][:-1])[::-1]
    mark(buf, nbs)
    return buf


def build_sknw(ske, multi=False):
    buf = buffer(ske)
    nbs = neighbors(buf.shape)
    acc = np.cumprod((1,)+buf.shape[::-1][:-1])[::-1]
    mark(buf, nbs)
    pts = np.array(np.where(buf.ravel() == 2))[0]
    nodes, edges = parse_struc(buf, pts, nbs, acc)
    return build_graph(nodes, edges, multi)

# draw the graph


def draw_graph(img, graph, cn=255, ce=128):
    acc = np.cumprod((1,) + img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    for (s, e) in graph.edges():
        eds = graph[s][e]
        if isinstance(graph, nx.MultiGraph):
            for i in eds:
                pts = eds[i]['pts']
                img[np.dot(pts, acc)] = ce
        else:
            img[np.dot(eds['pts'], acc)] = ce
    for idx in graph.nodes():
        pts = graph.nodes[idx]['pts']
        img[np.dot(pts, acc)] = cn



class RSInferenceDataset(Dataset):
    def __init__(
            self,
            rs_file,
            patch_size=(1024, 1024),
            slide_step=(1024, 1024),
            pad=256,
    ):
        super().__init__()
        self.rs_file = rs_file
        self.patch_size = _pair(patch_size)
        self.slide_step = _pair(slide_step)

        # get data info

        ds = gdal.Open(rs_file)
        self.data_info = self._get_data_info(ds)
        self.ids = self._get_patch_ids()
        ds = None
        self.pad=pad

    def __getitem__(self, idx):
        img,center,center_ = self._read_patch(idx)
        ids=np.array((self.ids[idx],center,center_))
       # print(ids)
        return img, ids

    def __len__(self):
        return len(self.ids)

    def _get_data_info(self, src):
        return {
            'width': src.RasterXSize,
            'height': src.RasterYSize,
            'driver': src.GetDriver().ShortName,
            'dtype': gdal.GetDataTypeName(src.GetRasterBand(1).DataType),
            'bands': src.RasterCount,
            'proj': src.GetProjection(),
            'geotransform': src.GetGeoTransform(),
        }

    def _get_patch_ids(self):
        left, top = 0, 0
        width, height = self.data_info['width'], self.data_info['height']
        if width%self.patch_size[0]!=0:
            width=(width//self.patch_size[0]+1)*self.patch_size[0]
        if height%self.patch_size[1]!=0:
            height=(height//self.patch_size[1]+1)*self.patch_size[1]
        left_top_xy = []  # left-top corner coordinates (xmin, ymin)
        while left < width:
            top = 0
            while top < height:
                left_top_xy.append((left, top))
                if top + self.patch_size[1] >= height:
                    break
                else:
                    top += self.slide_step[1]

            if left + self.patch_size[0] >= width:
                break
            else:
                left += self.slide_step[0]

        return left_top_xy# 

    def _read_patch(self, idx):
        xmin, ymin = self.ids[idx]
        width, height = self.data_info['width'], self.data_info['height']
        center_xsize,center_ysize=self.patch_size[0],self.patch_size[1]
        center_x,center_y=self.pad,self.pad
        xsize,ysize=self.patch_size[0]+2*self.pad,self.patch_size[1]+2*self.pad
        x_min,y_min=xmin-self.pad,ymin-self.pad
        left=0#x
        top=0#left-top corner coordinates (xmin, ymin)
        if x_min<0:
            xsize=xsize+x_min
            center_x=center_x+x_min
            x_min=0
        elif x_min+self.patch_size[0]+2*self.pad>width:
            xsize=width-x_min
            center_xsize=min(xsize-self.pad,self.patch_size[0])
        if y_min<0:
            ysize=ysize+y_min
            center_y=center_y+y_min
            y_min=0
        elif y_min+self.patch_size[1]+2*self.pad>height:
            ysize=height-y_min
            center_ysize=min(ysize-self.pad,self.patch_size[1])
            
        # to use multi-processing
        ds = gdal.Open(self.rs_file)
       
        band = ds.GetRasterBand(1)
        
        img = band.ReadAsArray(
            xoff=x_min,
            yoff=y_min,
            win_xsize=xsize,
            win_ysize=ysize,
        )
        center=(center_xsize,center_ysize)
        center_=(center_x,center_y)
        ds = None
        return img,center,center_

    @property
    def width(self):
        return self.data_info['width']

    @property
    def height(self):
        return self.data_info['height']

def distance(x,y):
    return np.sqrt(np.sum(np.square([x[0]-y[0],x[1]-y[1]])))
def patch_regular(gt,tau,thickness):
    #gt=tifffile.imread(img_path)
    ske = skeletonize(gt).astype(np.uint16)
    graph = build_sknw(ske)
    points=[]
    nodes=set()
    # draw edges by pts
    for (s,e) in graph.edges():
        ps = graph[s][e]['pts']
        p1=[float(ps[0,0]),float(ps[0,1])]
        p2=[float(ps[-1,0]),float(ps[-1,1])]
        nodes.add(str(p1))
        nodes.add(str(p2))
        points.append({str(p1),str(p2)})
        for i in range(0,len(ps)-1):
            cv2.line(gt,(int(ps[i,1]),int(ps[i,0])), (int(ps[i+1,1]),int(ps[i+1,0])), 1,thickness=thickness)
    ps=[eval(i) for i in list(nodes)]
    for num in range(len(ps)):
        mindis=float("inf")
        for other in range(len(ps)):
            if other!=num and {str(ps[num]),str(ps[other])} not in points:
                dis= distance(ps[num],ps[other])
                if dis<mindis:
                    mindis=dis
                    mindis_point=other
        if mindis<tau:
            cv2.line(gt,(int(ps[num][1]),int(ps[num][0])), (int(ps[mindis_point][1]),int(ps[mindis_point][0])), 1,thickness=thickness)
    return gt
def large_image_regular(image_file,out_dir,label,patch_size=1024,remove_obj_holes=11,tau=50,thickness=10):
    suffixs = ('.tif', '.tiff', '.img')
    if osp.isfile(image_file) and image_file.endswith(suffixs):
        image_list = [image_file]
    else:
        image_list = [
            osp.join(image_file, f)
            for f in scandir(image_file, suffix=suffixs)
        ]
    for img_file in image_list:
        dataset = RSInferenceDataset(img_file,
                                     patch_size=patch_size,
                                     slide_step=patch_size,
                                     pad=patch_size//4)
        basename = Path(img_file).stem
        out_file = osp.join(out_dir, f'{basename}_regular.tif')
        driver = gdal.GetDriverByName('GTiff')
        src_ds = gdal.Open(img_file)
        out_raster = driver.Create(out_file, dataset.width, dataset.height,
                                   1, gdal.GDT_Byte)
        gt = src_ds.GetGeoTransform()
        if gt is not None:
            out_raster.SetGeoTransform(gt)
        out_raster.SetProjection(src_ds.GetProjection())
        src_ds = None

        out_band = out_raster.GetRasterBand(1)
        pbar = tqdm(dataset)
        for img, offset in pbar:
            offset=np.squeeze(offset)
            center_xsize,center_ysize=offset[1]
            center_x, center_y = offset[2]
            offset=offset[0]
            pbar.set_description(Path(img_file).name)
            x_offset, y_offset = offset
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (remove_obj_holes, remove_obj_holes))
            img_1=copy.deepcopy(img)
            img_1[img_1!=label]=0
            img_1[img_1==label]=1
            img_1 = cv2.morphologyEx(img_1, cv2.MORPH_CLOSE, kernel)
            img_1=patch_regular(img_1,tau,thickness)
            img[img_1==1]=label
            img = img[center_y:(center_y+center_ysize), center_x:(center_x+center_xsize)]
            out_band.WriteArray(img.astype(np.uint8),
                                        xoff=x_offset.item(),
                                        yoff=y_offset.item())

        out_raster = None


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='road regular')

    parser.add_argument('image_file', help='input file path or directory')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('--label',
                        type=int,
                        default=3,
                        help='the label of road')
    parser.add_argument('--patch_size',
                        type=int,
                        default=1024,
                        help='patch size')
    parser.add_argument('--remove_obj_holes',
                        type=int,
                        default=11,
                        help='remove small objects and holes')
    parser.add_argument('--tau',
                        type=int,
                        default=100,
                        help='connect value')
    parser.add_argument('--thickness',
                        type=int,
                        default=10,
                        help='thickness value')         
                        
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse_args()
    large_image_regular(args.image_file,args.out_dir,args.label,remove_obj_holes=args.remove_obj_holes,tau=args.tau,thickness=args.thickness)


    
