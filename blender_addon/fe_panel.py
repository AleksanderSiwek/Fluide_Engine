import bpy

from bpy.types import Panel


class FE_PT_Panel(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_label = "Test Panel"
    bl_category = "Test Panel"
    
    def draw(self, context):
        layout = self.layout
        
        row = layout.row()
        col = row.column()  
        col.operator("object.add_cube", text="Add cube")
        