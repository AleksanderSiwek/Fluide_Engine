import bpy

from bpy.types import Operator


class FE_OT_AddCube(Operator):
    bl_idname = "object.add_cube"
    bl_label = "Add cube"
    bl_description = "Add cube to the scene at (0, 0, 0)."
    
    def  execute(self, context):
        bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

        return {'FINISHED'}

