# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae
from .detr_vae import build_cnnmlp as build_cnnmlp

def build_ACT_model(args):
    return build_vae(args)   # 调用detr_vae中build函数

def build_CNNMLP_model(args):
    return build_cnnmlp(args)