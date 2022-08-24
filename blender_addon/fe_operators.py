import os
import sys

sys.path.append("D:/_STUDIA/Praca_magisterska/Fluid_Engine/build/src/python_wrapper\Debug/")
os.environ['PATH'] = os.path.join(os.environ['CUDA_PATH'], 'bin')

import PyFluidEngine as pyf

import bpy
import bmesh
from bpy.types import Operator
import mathutils


def find_all_meshes_with_given_property(property):
    objects = []
    for ob in bpy.context.scene.objects:
        if ob.type=='MESH':
            if ob.my_global_enum == property:
                objects.append(ob)
    return objects

def reset_objects_origins(objects):
    saved_location = bpy.context.scene.cursor.location
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    for obj in objects:
        mw = obj.matrix_world
        imw = mw.inverted()
        me = obj.data
        origin = bpy.context.scene.cursor.location
        local_origin = imw @ origin
        me.transform(mathutils.Matrix.Translation(-local_origin))
        mw.translation += (origin - mw.translation)
    bpy.context.scene.cursor.location = saved_location
    
def triangulate_object(obj):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    
def triangulate_objects(obj_collection):
    for obj in obj_collection:
        triangulate_object(obj)
    
def triangle_mesh_2_blender_object(mesh):
    return 0
    
def blender_object_2_triangle_mesh(mesh):
    return 0

    
class FE_OT_RunFluidSimulation(Operator):
    bl_idname = "scene.run_fluid_simulation"
    bl_label = "Run simulation"
    bl_description = "Run fluid simulation."
    
    def select_all_simulator_meshes(self):
        meshes = []
        
        fluid_objects = find_all_meshes_with_given_property("FLUID")
        self.report({'INFO'}, "FLUID objects: " + str([obj.name for obj in fluid_objects]))
        
        collider_objects = find_all_meshes_with_given_property("COLLIDER")
        self.report({'INFO'}, "COLLIDER objects:" + str([obj.name for obj in collider_objects]))

        emmiter_objects = find_all_meshes_with_given_property("EMMITER")
        self.report({'INFO'}, "EMMITER objects: " + str([obj.name for obj in emmiter_objects]))
        
        terminator_objects = find_all_meshes_with_given_property("TERMINATOR")
        self.report({'INFO'}, "TERMINATOR objects: " + str([obj.name for obj in terminator_objects]))
        
        force_objects = find_all_meshes_with_given_property("FORCE")
        self.report({'INFO'}, "FORCE objects: " + str([obj.name for obj in force_objects]))
        
        meshes.extend(fluid_objects)
        meshes.extend(collider_objects)
        meshes.extend(emmiter_objects)
        meshes.extend(terminator_objects)
        meshes.extend(force_objects)
        return meshes
    
    def reset_simulator_meshes_origins(self, meshes):
        reset_objects_origins(meshes)

    def triangulate_simulator_meshes(self, meshes):
        triangulate_objects(meshes)

    def preprocess_simulator_meshes(self):
        meshes = self.select_all_simulator_meshes()
        self.reset_simulator_meshes_origins(meshes)
        triangulate_simulator_meshes(meshes)
    
    def execute(self, context):
        props = context.scene.fe_props

        fps = props.FPS
        number_of_frames = props.number_of_frames
        cache_directory = props.cache_directory
        domain_origin = props.domain_origin
        domain_size = props.domain_size
        
        self.report({'INFO'}, "FPS: " + str(fps))
        self.report({'INFO'}, "number_of_frames: " + str(number_of_frames))
        self.report({'INFO'}, "cache_directory: " + str(cache_directory))
        self.report({'INFO'}, "domain_origin: " + str(domain_origin))
        self.report({'INFO'}, "domain_size: " + str(domain_size))

        self.preprocess_simulator_meshes()
                       
        return {'FINISHED'}


