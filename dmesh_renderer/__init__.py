from typing import NamedTuple
import torch as th
from . import _C

'''
===============================================================================
* TriRenderer

Renderer for (semi-transparent) triangles.
Efficient, but do not support accurate depth testing.
===============================================================================
'''
class TriRenderSettings(NamedTuple):
    image_height: int
    image_width: int
    bg : th.Tensor

def render_tri(
    verts: th.Tensor,
    faces: th.Tensor,
    verts_color: th.Tensor,
    faces_opacity: th.Tensor,

    mv_mats: th.Tensor,
    proj_mats: th.Tensor,
    verts_depth: th.Tensor,
    faces_intense: th.Tensor,

    render_settings: TriRenderSettings,
):
    return _RenderTri.apply(
        verts,
        faces,
        verts_color,
        faces_opacity,

        mv_mats,
        proj_mats,
        verts_depth,
        faces_intense,

        render_settings,
    )

class _RenderTri(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        verts,
        faces,
        verts_color,
        faces_opacity,

        mv_mats,
        proj_mats,
        verts_depth,
        faces_intense,

        render_settings,
    ):
        
        inv_mv_mats = th.inverse(mv_mats)
        inv_proj_mats = th.inverse(proj_mats)

        # Restructure arguments the way that the C++ lib expects them
        args = (
            render_settings.bg,
            
            verts,
            faces,
            verts_color,
            faces_opacity,

            mv_mats,
            proj_mats,
            inv_mv_mats,
            inv_proj_mats,
            verts_depth,
            faces_intense,

            render_settings.image_height,
            render_settings.image_width
        )

        # Invoke C++/CUDA rasterizer
        try:
            # depth: range in [-1, 1], -1 is near, 1 is far
            num_rendered, color, depth, pointBuffer, faceBuffer, binningBuffer, imgBuffer = _C.render_tris(*args)
        except Exception as ex:
            print("\nAn error occured in forward.")
            print(ex)
            raise ex
        
        # Keep relevant tensors for backward
        ctx.render_settings = render_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            verts, 
            faces, 
            verts_color, 
            faces_opacity, 

            mv_mats,
            proj_mats,
            inv_mv_mats,
            inv_proj_mats,
            verts_depth,
            faces_intense,

            pointBuffer, faceBuffer, binningBuffer, imgBuffer)
        return color, depth

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        render_settings = ctx.render_settings
        verts, faces, verts_color, faces_opacity, \
            mv_mats, proj_mats, inv_mv_mats, inv_proj_mats, verts_depth, faces_intense, \
            pointBuffer, faceBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (render_settings.bg,
                
                verts, 
                faces, 
                verts_color, 
                faces_opacity, 
                
                mv_mats, 
                proj_mats,
                inv_mv_mats,
                inv_proj_mats,
                verts_depth,
                faces_intense,

                grad_out_color, 
                grad_out_depth,

                num_rendered,

                pointBuffer,
                faceBuffer,
                binningBuffer,
                imgBuffer)

        try:
            grad_results = _C.render_tris_backward(*args)
            grad_verts, grad_verts_color, grad_faces_opacity, grad_verts_depth, grad_faces_intense = \
                grad_results[0], grad_results[1], grad_results[2], grad_results[3], grad_results[4]
        except Exception as ex:
            print("\nAn error occured in backward.\n")
            raise ex
        
        grads = (
            grad_verts,
            None,
            grad_verts_color,
            grad_faces_opacity,

            None,
            None,
            grad_verts_depth,
            grad_faces_intense,

            None,
        )

        return grads

class TriRenderer(th.nn.Module):

    def __init__(self, render_settings: TriRenderSettings):
        super().__init__()
        self.render_settings = render_settings

    def forward(self, 
                verts: th.Tensor, 
                faces: th.Tensor,
                verts_color: th.Tensor,
                faces_opacity: th.Tensor,

                mv_mats: th.Tensor,
                proj_mats: th.Tensor,
                verts_depth: th.Tensor,
                faces_intense: th.Tensor):

        '''
        Verts, faces, verts_color, and faces opacity are independent to view point (batch).

        Verts_depth and faces_intense are dependent to view point (batch).
        
        Renders color map based on interpolating vert-wise colors and applying
        face-wise light intensities.

        Renders depth map based on interpolating vert-wise depths.

        @ verts: [# vert, 3]
        @ faces: [# faces, 3]
        @ verts_color: [# vert, 3]
        @ faces_opacity: [# faces,]

        @ mv_mats: [# batch, 4, 4]
        @ proj_mats: [# batch, 4, 4]
        @ verts_depth: [# batch, # vert,]
        @ faces_intense: [# batch, # faces,]
        '''
        
        render_settings = self.render_settings

        # Invoke C++/CUDA rasterization routine
        return render_tri(
            verts,
            faces.to(dtype=th.int32),
            verts_color,
            faces_opacity,

            mv_mats.transpose(1, 2),
            proj_mats.transpose(1, 2),
            verts_depth,
            faces_intense,
            
            render_settings, 
        )


'''
===============================================================================
* TetRenderer

Renderer for faces in a compact set of tetrahedra.
Efficient, support accurate depth testing, but only propagate gradients to opacities.
===============================================================================
'''

class TetRenderSettings(NamedTuple):
    image_height: int
    image_width: int
    bg : th.Tensor
    ray_random_seed : int

def render_tet(
    verts: th.Tensor, 
    faces: th.Tensor, 
    verts_color: th.Tensor,
    faces_opacity: th.Tensor,
    
    mv_mats: th.Tensor,
    proj_mats: th.Tensor,
    verts_depth: th.Tensor,
    faces_intense: th.Tensor,

    tets: th.Tensor,
    face_tets: th.Tensor,
    tet_faces: th.Tensor,

    render_settings: TetRenderSettings
):
    return _RenderTet.apply(
        verts,
        faces,
        verts_color,
        faces_opacity,

        mv_mats,
        proj_mats,
        verts_depth,
        faces_intense,
        
        tets,
        face_tets,
        tet_faces,
        render_settings
    )

class _RenderTet(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        
        verts,
        faces,
        verts_color,
        faces_opacity,

        mv_mats,
        proj_mats,
        verts_depth,
        faces_intense,

        tets,
        face_tets,
        tet_faces,

        render_settings: TetRenderSettings
    ):
        inv_mv_mats = th.inverse(mv_mats)
        inv_proj_mats = th.inverse(proj_mats)

        # Restructure arguments the way that the C++ lib expects them
        args = (
            render_settings.bg,
        
            verts,
            faces,
            verts_color,
            faces_opacity,
            
            mv_mats,
            proj_mats,
            inv_mv_mats,
            inv_proj_mats,
            verts_depth,
            faces_intense,
            
            tets,
            face_tets,
            tet_faces,

            render_settings.image_height,
            render_settings.image_width,
            render_settings.ray_random_seed
        )

        # Invoke C++/CUDA renderer
        try:
            color, depth, active, pointBuffer, faceBuffer, binningBuffer, imgBuffer = _C.render_tets(*args)
        except Exception as ex:
            print("\nAn error occured in forward.")
            raise ex
        
        active = (active > 0.5) # change to boolean tensor;
        
        # Keep relevant tensors for backward
        ctx.render_settings = render_settings
        ctx.save_for_backward(
                            verts,
                            faces,
                            verts_color,
                            faces_opacity,

                            mv_mats,
                            proj_mats,
                            inv_mv_mats,
                            inv_proj_mats,
                            verts_depth,
                            faces_intense,

                            tets,
                            face_tets,
                            tet_faces,

                            pointBuffer,
                            faceBuffer,
                            binningBuffer,
                            imgBuffer)
        
        return color, depth, active

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_depth, grad_out_active):

        # Restore necessary values from context
        render_settings = ctx.render_settings
        verts, faces, verts_color, faces_opacity, \
            mv_mats, proj_mats, inv_mv_mats, inv_proj_mats, \
            verts_depth, faces_intense, \
            tets, face_tets, tet_faces, \
            pointBuffer, faceBuffer, binningBuffer, imgBuffer = ctx.saved_tensors
        
        # Restructure args as C++ method expects them
        args = (render_settings.bg,
        
                verts, 
                faces,
                verts_color,
                faces_opacity,

                mv_mats,
                proj_mats,
                inv_mv_mats,
                inv_proj_mats,
                verts_depth,
                faces_intense,
                
                tets,
                face_tets,
                tet_faces,

                grad_out_color,
                grad_out_depth,

                pointBuffer,
                faceBuffer,
                binningBuffer,
                imgBuffer)

        # Compute gradients for relevant tensors by invoking backward method
        try:
            grad_results = _C.render_tets_backward(*args)
            grad_verts_color, grad_faces_opacity = grad_results[0], grad_results[1]
        except Exception as ex:
            print("\nAn error occured in backward.\n")
            raise ex
        
        grads = (
            None,
            None,
            grad_verts_color,
            grad_faces_opacity,

            None,
            None,
            None,
            None,
            
            None,
            None,
            None,

            None)

        return grads

class TetRenderer(th.nn.Module):
    def __init__(self, render_settings: TetRenderSettings):
        super().__init__()
        self.render_settings = render_settings

    def forward(self, 
                verts: th.Tensor, 
                faces: th.Tensor, 
                verts_color: th.Tensor,
                faces_opacity: th.Tensor,
                
                mv_mats: th.Tensor,
                proj_mats: th.Tensor,
                verts_depth: th.Tensor,
                faces_intense: th.Tensor,
                
                tets: th.Tensor,
                face_tets: th.Tensor,
                tet_faces: th.Tensor):
        
        '''
        Gradients are only provided for [verts_color] and [faces_opacity].
        @ Note: [verts_depth] is not used, because depth is computed using
        normalization based on w coords, which is non linear.

        @ verts: [# vertex, 3], positions of vertices
        @ faces: [# face, 3], indices of vertices of faces
        @ verts_color: [# vertex, 3], colors of vertices
        @ faces_opacity: [# face,], opacity of faces
        
        @ mv_mats: [# batch, 4, 4], batch of modelview matrices
        @ proj_mats: [# batch, 4, 4], batch of projection matrices
        @ verts_depth: [# batch, # vertex,], depth of vertices
        @ faces_intense: [# batch, # face,], intensity of faces
        
        @ tets: [# tet, 4], indices of vertices of tets
        @ face_tets: [# face, 2], indices of tets that a face belongs to
        @ tet_faces: [# tet, 4], indices of faces that a tet owns
        '''
        
        render_settings = self.render_settings

        # reshape;

        color, depth, active = render_tet(
            verts.to(dtype=th.float32),
            faces.to(dtype=th.int32),
            verts_color.to(dtype=th.float32),
            faces_opacity.to(dtype=th.float32),

            mv_mats.to(dtype=th.float32).transpose(1, 2),
            proj_mats.to(dtype=th.float32).transpose(1, 2),
            verts_depth.to(dtype=th.float32),
            faces_intense.to(dtype=th.float32),

            tets.to(dtype=th.int32),
            face_tets.to(dtype=th.int32),
            tet_faces.to(dtype=th.int32),

            render_settings
        )

        return color, depth, active