import bpy

from bpy.types import Panel, PropertyGroup


class FluidEngineProperties(PropertyGroup):
    FPS : bpy.props.IntProperty(name="FPS", default=30)
    number_of_frames : bpy.props.IntProperty(name="Number of frames:", soft_min= 1, default=250)
    resolution : bpy.props.IntProperty(name="Resolution", soft_min=1, default=20)
    cache_directory : bpy.props.StringProperty(name="Path to cache directory", default="", subtype="DIR_PATH")
    domain_origin : bpy.props.FloatVectorProperty(name="Domain origin", default=(0, 0, 0))
    domain_size : bpy.props.FloatVectorProperty(name="Domain size", default=(0, 0, 0))


class FE_PT_FluidEngine(Panel):
    bl_idname = "FLUID_ENGINE_MAIN_PANEL"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_label = "Fluid Engine"
    bl_category = "Fluid Engine"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.fe_props
        
        row = layout.row()
        row.prop(props, "FPS")
        
        row = layout.row()
        row.prop(props, "number_of_frames")
        
        row = layout.row()
        row.prop(props, "resolution")
        
        row = layout.row()
        row.prop(props, "cache_directory")
        
        row = layout.row()
        row.prop(props, "domain_origin")
        
        row = layout.row()
        row.prop(props, "domain_size")

        row = layout.row()
        row.operator(operator="scene.run_fluid_simulation", text="Simulate!")
        
class FE_PT_SimulatorPanel(bpy.types.Panel):
    bl_label = "Simulator"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Fluid Engine"
    bl_parent_id = "FLUID_ENGINE_MAIN_PANEL"

    def draw(self, context):
        layout = self.layout
        layout.label(text="This is main simulator panel.")
    
    
class FE_PT_ObjectInspector(bpy.types.Panel):
    bl_label = "Object Inspector"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Fluid Engine"
    bl_parent_id = "FLUID_ENGINE_MAIN_PANEL"

    def draw(self, context):
        layout = self.layout
        layout.label(text="This is Inspector panel.")
        
        ob = context.object
        if ob is not None:
            props = layout.operator(SimpleOperator.bl_idname)
            props.my_enum = ob.my_global_enum
            row = layout.row()
            row.prop(ob, "my_global_enum", expand=True)
            
            if props.my_enum == "NONE":
                layout.label(text="None object, nothing to see here.")
            elif props.my_enum == "FLUID":
                layout.label(text="Fluid object!")
            elif props.my_enum == "COLLIDER":
                layout.label(text="COLLIDER object!")
            elif props.my_enum == "EMMITER":
                layout.label(text="EMMITER object!")
            elif props.my_enum == "TERMINATOR":
                layout.label(text="TERMINATOR object!")
            elif props.my_enum == "FORCE":
                layout.label(text="FORCE object!")


class FE_PT_Stats(bpy.types.Panel):
    bl_label = "Simulation Stats"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Fluid Engine"
    bl_parent_id = "FLUID_ENGINE_MAIN_PANEL"

    def draw(self, context):
        layout = self.layout
        layout.label(text="This is stats panel.")


class SimpleOperator(bpy.types.Operator):
    bl_label = "Simple Operator"
    bl_idname = "object.simple_operator"
    
    my_enum: bpy.props.EnumProperty(
        items=(
            ('NONE', "None", ""),
            ('FLUID', "Fluid", ""),
            ('COLLIDER', "Collider", ""),
            ('EMMITER', "Emmiter", ""),
            ('TERMINATOR', "Terminator", ""),
            ('FORCE', "Force", "")
        ),
        default='NONE'
    )
    
    def execute(self, context):
        ob = context.object
        if ob is not None:
            self.report({'INFO'}, self.my_enum)
        return {'FINISHED'}
