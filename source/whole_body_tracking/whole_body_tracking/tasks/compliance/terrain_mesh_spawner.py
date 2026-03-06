# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom terrain mesh spawner that parses URDF files and uses create_prim_from_mesh."""

from __future__ import annotations

import functools
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import trimesh
from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.sim.utils import clone, find_matching_prim_paths

if TYPE_CHECKING:
    from .terrain_mesh_spawner_cfg import TerrainMeshSpawnerCfg
else:
    TerrainMeshSpawnerCfg = None

# Marker used in urdf_paths when a motion has no matching terrain; spawner uses flat plane.
FLAT_PLANE_MARKER = "__FLAT_PLANE__"


def spawn_multi_terrain_per_env(func):
    """Decorator that spawns terrain for EACH environment individually, instead of spawn-once-then-clone.

    The standard @clone decorator spawns once at prim_paths[0] and clones to all other envs,
    causing all environments to have identical terrain. This decorator spawns each environment
    separately with its own URDF (based on env_id % len(urdf_paths)), enabling different
    terrains per environment.

    When replicate_physics=True (IsaacLab default), only env_0 exists at spawn time, so
    find_matching_prim_paths would return a single path. The cfg.num_envs must be set
    to generate all environment paths explicitly.
    """
    @functools.wraps(func)
    def wrapper(prim_path: str, cfg, *args, **kwargs):
        prim_path = str(prim_path)
        if not prim_path.startswith("/"):
            raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")
        root_path, asset_path = prim_path.rsplit("/", 1)
        # When num_envs is set (required for replicate_physics=True), generate paths explicitly
        # since only env_0 exists at spawn time and find_matching_prim_paths would return just one path
        num_envs = getattr(cfg, "num_envs", None)
        if num_envs is not None:
            # Extract base: /World/envs/env_.* -> /World/envs
            base = re.sub(r"/env_.*$", "", root_path)
            prim_paths = [f"{base}/env_{i}/{asset_path}" for i in range(num_envs)]
        else:
            is_regex = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None
            if is_regex and root_path:
                source_prim_paths = find_matching_prim_paths(root_path)
                if not source_prim_paths:
                    raise RuntimeError(
                        f"Unable to find source prim path: '{root_path}'. "
                        "For multi-terrain with replicate_physics=True, set num_envs on the spawn config."
                    )
            else:
                source_prim_paths = [root_path]
            prim_paths = [f"{p}/{asset_path}" for p in source_prim_paths]
        # Spawn for EACH path individually (no cloning)
        first_prim = None
        for path in prim_paths:
            prim = func(path, cfg, *args, **kwargs)
            if first_prim is None:
                first_prim = prim
        return first_prim
    return wrapper


def parse_urdf_mesh_info(urdf_path: str) -> tuple[str, tuple[float, float, float]]:
    """Parse URDF file to extract mesh path and scale.
    
    Args:
        urdf_path: Path to the URDF file
        
    Returns:
        Tuple of (mesh_path, scale) where:
        - mesh_path: Path to the mesh file (relative to URDF directory)
        - scale: Tuple of (x, y, z) scale values
        
    Raises:
        ValueError: If mesh information cannot be found in URDF
    """
    urdf_file = Path(urdf_path)
    if not urdf_file.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    
    # Parse URDF XML
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Find mesh element in visual geometry
    mesh_path = None
    scale = (1.0, 1.0, 1.0)  # Default scale
    
    # Search in visual geometry first (as per URDF format)
    for visual in root.findall(".//visual"):
        geometry = visual.find("geometry")
        if geometry is not None:
            mesh_elem = geometry.find("mesh")
            if mesh_elem is not None:
                # Get mesh filename
                filename_attr = mesh_elem.get("filename")
                if filename_attr:
                    mesh_path = filename_attr
                    # Get scale if present
                    scale_attr = mesh_elem.get("scale")
                    if scale_attr:
                        # Parse scale string like "0.75 0.75 0.75"
                        scale_values = scale_attr.split()
                        if len(scale_values) == 3:
                            scale = tuple(float(s) for s in scale_values)
                    break
    
    if mesh_path is None:
        raise ValueError(f"Could not find mesh element in URDF file: {urdf_path}")
    
    # Resolve mesh path relative to URDF directory
    urdf_dir = urdf_file.parent
    mesh_file = urdf_dir / mesh_path
    
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
    
    return str(mesh_file.resolve()), scale


def extract_env_id_from_prim_path(prim_path: str) -> int:
    """Extract environment ID from prim path.
    
    Args:
        prim_path: Prim path like "/World/envs/env_0/Terrain" or "/World/envs/env_123/Terrain"
        
    Returns:
        Environment ID as integer
    """
    # Match patterns like env_0, env_123, etc.
    match = re.search(r"env_(\d+)", prim_path)
    if match:
        return int(match.group(1))
    else:
        # Fallback: try to extract number from path
        # This handles edge cases where naming might differ
        raise ValueError(f"Could not extract environment ID from prim_path: {prim_path}")


@clone
def spawn_terrain_from_urdf(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn terrain mesh from URDF file using create_prim_from_mesh.
    
    This function:
    1. Parses the URDF file to extract mesh path and scale
    2. Loads the mesh using trimesh
    3. Applies scale to the mesh
    4. Creates prim using create_prim_from_mesh with material properties
    
    Args:
        prim_path: Path to spawn the terrain prim at
        cfg: Configuration for terrain mesh spawner
        translation: Translation of the terrain (unused, kept for API compatibility)
        orientation: Orientation of the terrain (unused, kept for API compatibility)
        
    Returns:
        The created USD prim
    """
    # Parse URDF to get mesh path and scale
    mesh_path, urdf_scale = parse_urdf_mesh_info(cfg.urdf_path)
    
    # Load mesh using trimesh
    mesh = trimesh.load(mesh_path, process=False)
    
    # Handle Scene objects from multi-mesh files
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded object is not a valid Trimesh: {type(mesh)}")
    
    # Apply scale from URDF
    if urdf_scale != (1.0, 1.0, 1.0):
        # Scale vertices
        import numpy as np
        mesh.vertices = mesh.vertices * np.array(urdf_scale)
    
    # Configure visual material
    visual_material = cfg.visual_material if cfg.visual_material is not None else sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.5, 0.4, 0.3)
    )
    
    # Configure physics material
    physics_material = cfg.physics_material if cfg.physics_material is not None else sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    
    # Create the terrain mesh prim
    create_prim_from_mesh(
        prim_path,
        mesh,
        visual_material=visual_material,
        physics_material=physics_material,
        translation=translation,
        orientation=orientation,
    )
    
    # Apply rigid body properties if specified
    # Note: create_prim_from_mesh creates a mesh at {prim_path}/mesh, but we apply
    # rigid body properties to the parent Xform at prim_path
    if cfg.rigid_props is not None:
        sim_utils.define_rigid_body_properties(prim_path, cfg.rigid_props)
    
    # Get the created prim (the Xform parent)
    import isaacsim.core.utils.prims as prim_utils
    prim = prim_utils.get_prim_at_path(prim_path)
    
    return prim


@spawn_multi_terrain_per_env
def spawn_terrain_from_urdf_list(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn terrain mesh from a list of URDF files, selecting based on environment ID.
    
    This function:
    1. Extracts environment ID from prim_path
    2. Selects URDF file from list based on environment ID (using modulo)
    3. Parses the selected URDF file to extract mesh path and scale
    4. Loads the mesh using trimesh
    5. Applies scale to the mesh
    6. Creates prim using create_prim_from_mesh with material properties
    
    Args:
        prim_path: Path to spawn the terrain prim at (contains environment ID)
        cfg: Configuration for terrain mesh spawner with urdf_paths list
        translation: Translation of the terrain (unused, kept for API compatibility)
        orientation: Orientation of the terrain (unused, kept for API compatibility)
        
    Returns:
        The created USD prim
    """
    # Extract environment ID from prim_path
    env_id = extract_env_id_from_prim_path(prim_path)
    
    # Select URDF file based on environment ID (use modulo to cycle through available terrains)
    urdf_paths = cfg.urdf_paths
    if not urdf_paths:
        raise ValueError("urdf_paths list is empty in config")

    selected_urdf_path = urdf_paths[env_id % len(urdf_paths)]

    # Default flat plane when motion has no matching terrain
    if selected_urdf_path == FLAT_PLANE_MARKER:
        # trimesh has no plane(); use thin box (20m x 20m x 0.01m) as flat terrain
        mesh = trimesh.creation.box(extents=[20.0, 20.0, 0.01])
    else:
        # Parse URDF to get mesh path and scale
        mesh_path, urdf_scale = parse_urdf_mesh_info(selected_urdf_path)

        # Load mesh using trimesh
        mesh = trimesh.load(mesh_path, process=False)

        # Handle Scene objects from multi-mesh files
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        if urdf_scale != (1.0, 1.0, 1.0):
            import numpy as np
            mesh.vertices = mesh.vertices * np.array(urdf_scale)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded object is not a valid Trimesh: {type(mesh)}")
    
    # Configure visual material
    visual_material = cfg.visual_material if cfg.visual_material is not None else sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.5, 0.4, 0.3)
    )
    
    # Configure physics material
    physics_material = cfg.physics_material if cfg.physics_material is not None else sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    
    # Create the terrain mesh prim
    create_prim_from_mesh(
        prim_path,
        mesh,
        visual_material=visual_material,
        physics_material=physics_material,
        translation=translation,
        orientation=orientation,
    )
    
    # Apply rigid body properties if specified
    # Note: create_prim_from_mesh creates a mesh at {prim_path}/mesh, but we apply
    # rigid body properties to the parent Xform at prim_path
    if cfg.rigid_props is not None:
        sim_utils.define_rigid_body_properties(prim_path, cfg.rigid_props)
    
    # Get the created prim (the Xform parent)
    import isaacsim.core.utils.prims as prim_utils
    prim = prim_utils.get_prim_at_path(prim_path)
    
    return prim
