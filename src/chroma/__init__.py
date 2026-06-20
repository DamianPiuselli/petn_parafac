"""
PETN Chroma package.
Contains physics-embedded models, data generators, and routines for chromatography retention time shifting (e.g. GC-MS, HPLC-DAD).
"""

from src.chroma.base import BaseChromaPETN
from src.chroma.hplc import HPLC_PETN
from src.chroma.gcms import GCMS_PETN
from src.chroma.plots import extract_loadings, extract_loadings_df, plot_alignment_verification, plot_scores_comparison
from src.chroma.generator import GCMSDataGenerator, HPLCDADDataGenerator
from src.chroma.dataset import ChromatographyCOODataset, get_chroma_dataloader

__all__ = [
    'BaseChromaPETN',
    'HPLC_PETN',
    'GCMS_PETN',
    'extract_loadings',
    'extract_loadings_df',
    'plot_alignment_verification',
    'plot_scores_comparison',
    'GCMSDataGenerator',
    'HPLCDADDataGenerator',
    'ChromatographyCOODataset',
    'get_chroma_dataloader',
]
