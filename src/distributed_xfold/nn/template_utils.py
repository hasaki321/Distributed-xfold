from alphafold3.model import protein_data_processing
import jax.numpy as jnp
import jax
import numpy as np
import torch
from alphafold3.common import base_config
from alphafold3.jax import geometry
from typing import Optional, Tuple, Union, Callable, List, Self
from jaxlib.xla_extension import ArrayImpl

def cross(e0, e1):
  """
  Compute cross product between e0 and e1.
  # e0: [num_res, 3]
  # e1: [num_res, 3]
  # out: [num_res,]
  """
  # e2 = torch.zeros_like(e0)
  # e2[:, 0] = e0[:, 1] * e1[:, 2] - e0[:, 2] * e1[:, 1]
  # e2[:, 1] = e0[:, 2] * e1[:, 0] - e0[:, 0] * e1[:, 2]
  # e2[:, 2] = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
  return torch.cross(e0, e1, dim=-1)

def dot(e0, e1):
  """Compute dot product between 'self' and 'other'."""
  # e0: [num_res, 3]
  # e1: [num_res, 3]
  # out: [num_res,]
  # return self.x * other.x + self.y * other.y + self.z * other.z
  return (e0 * e1).sum(-1)

def norm(e, epsilon: float = 1e-6):
  """Compute Norm of Vec3Array, clipped to epsilon."""
  # e: [num_res, 3]
  # To avoid NaN on the backward pass, we must use maximum before the sqrt
  norm2 = dot(e, e) # [num_res,]
  if epsilon:
    norm2 = torch.maximum(norm2, torch.tensor(epsilon**2, dtype=norm2.dtype))
  return torch.sqrt(norm2) # [num_res,]

def normalized(e, epsilon: float = 1e-6):
  """
  Return unit vector with optional clipping.
  e: [num_res, 3]
  out: [num_res, 3]
  """
  return e / norm(e, epsilon).unsqueeze(-1)

def from_two_vectors(e0, e1):
  e0 = normalized(e0)  # [num_res, 3] 
  # make e1 perpendicular to e0.
  c = dot(e1, e0)  # [num_res] 
  e1 = normalized(e1 - (c[:, None] * e0)) # [num_res, 3] 
  # Compute e2 as cross product of e0 and e1.
  e2 = cross(e0, e1) # [num_res, 3] 

  rotation = torch.stack((e0, e1, e2), dim=-1) # [num_res, 3, 3] 
  # (e0[:, 0], e1[:, 0], e2[:, 0], 
  #  e0[:, 1], e1[:, 1], e2[:, 1], 
  #  e0[:, 2], e1[:, 2], e2[:, 2])
  """
  (
    xx,xy,xz
    yx,yy,yz
    zx,zy,zz
  )
  """
  return rotation # Tuple[num_res,]

def rot_apply_to_point(rotation, point):
  """Applies Rot3Array to point."""
  """
  rotation: [num_res, 3, 3] 
  point: [num_res, 3] 
  return vector.Vec3Array(
      self.xx * point.x + self.xy * point.y + self.xz * point.z,
      self.yx * point.x + self.yy * point.y + self.yz * point.z,
      self.zx * point.x + self.zy * point.y + self.zz * point.z,
  )
  """
  point_out = (rotation * point.unsqueeze(-2)).sum(-1) # [num_res, 3] 
  return point_out

def apply_to_point(rotation, translation, point):
  """Apply Rigid3Array transform to point."""
  return rot_apply_to_point(rotation, point) + translation

def inverse_rot(rotation):
  """Returns inverse of Rot3Array."""
  inverse = rotation.transpose(-1, -2)
  """
  return Rot3Array(
      *(self.xx, self.yx, self.zx),
      *(self.xy, self.yy, self.zy),
      *(self.xz, self.yz, self.zz),
  )
  """
  return inverse
  

def inverse(rotation, translation):
  """Return Rigid3Array corresponding to inverse transform."""
  inv_rotation = inverse_rot(rotation)
  inv_translation = rot_apply_to_point(inv_rotation, -translation)
  return inv_rotation, inv_translation

def make_backbone_vectors(
    positions: torch.Tensor,
    mask: torch.Tensor,
    group_indices: torch.Tensor,
):
    """
    Args:
        positions: [num_res, num_atoms, 3] 原子位置 (Vec3Array)
        mask: [num_res, num_atoms] 原子掩码
        group_indices: [num_res, num_group, 3] 原子索引组
    
    Returns:
        (Rot3Array, [num_res] 掩码)
    """
    batch_index = torch.arange(mask.shape[0], dtype=torch.long)

    # 提取主干原子索引 (N, CA, C)
    backbone_indices = group_indices[:, 0].long()  # [num_res, 3]
    c, b, a = torch.unbind(backbone_indices, -1)

    # 计算刚性变换掩码 (所有三个原子必须有效)
    slice_index = lambda x, idx: x[batch_index, idx]
    rigid_mask = (
        slice_index(mask, a) * slice_index(mask, b) * slice_index(mask, c)
    ).float()  # [num_res]

    frame_positions = []
    # positions  [num_res, num_atoms, 3] 
    for indices in [a, b, c]:
      frame_positions.append(
        slice_index(positions, indices)
    )
      
    # frame_positions  List[num_res, num_atoms, 3] 
    # 构建旋转矩阵 (从 CA->C 和 CA->N 向量)
    e0 = frame_positions[2] - frame_positions[1]  # [num_res, 3] 
    e1 = frame_positions[0] - frame_positions[1]  # [num_res, 3] 

    rotation = from_two_vectors(e0, e1)
    translation = frame_positions[1]
    rigid = (rotation[:, None, :, :], translation[:, None, :])

    points = translation
    rigid_vec = apply_to_point(*inverse(*rigid), points) # [num_res, 3] 
    unit_vector = normalized(rigid_vec)
    unit_vector = torch.unbind(unit_vector, -1)

    return unit_vector, rigid_mask