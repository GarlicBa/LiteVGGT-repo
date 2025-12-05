# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch


def apply_distortion(points, distortion_params):
    """
    Apply distortion to normalized camera coordinates.
    
    Args:
        points: Array of normalized camera coordinates
        distortion_params: Distortion parameters
        
    Returns:
        Distorted coordinates
    """
    # Simple passthrough for now - implement actual distortion if needed
    return points


def iterative_undistortion(points, distortion_params, max_iter=10):
    """
    Remove distortion from normalized camera coordinates using iterative method.
    
    Args:
        points: Array of distorted normalized camera coordinates
        distortion_params: Distortion parameters
        max_iter: Maximum number of iterations
        
    Returns:
        Undistorted coordinates
    """
    # Simple passthrough for now - implement actual undistortion if needed
    return points


def single_undistortion(points, distortion_params):
    """
    Remove distortion from normalized camera coordinates using single step.
    
    Args:
        points: Array of distorted normalized camera coordinates
        distortion_params: Distortion parameters
        
    Returns:
        Undistorted coordinates
    """
    # Simple passthrough for now - implement actual undistortion if needed
    return points 