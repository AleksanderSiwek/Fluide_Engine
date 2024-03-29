# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name" : "FluidEngine",
    "author" : "Aleksander Siwek",
    "description" : "",
    "blender" : (2, 80, 0),
    "version" : (0, 0, 1),
    "location" : "View3D",
    "warning" : "",
    "category" : "Object"
}

import bpy

from . fe_operators import FE_OT_RunFluidSimulation
from . fe_panel import FE_PT_FluidEngine, FE_PT_SimulatorPanel, FE_PT_ObjectInspector, FE_PT_Stats, SimpleOperator, FluidEngineProperties

classes = (FE_OT_RunFluidSimulation, FluidEngineProperties,  FE_PT_FluidEngine, FE_PT_SimulatorPanel, FE_PT_ObjectInspector, FE_PT_Stats, SimpleOperator)

def register():
    for c in classes:
        bpy.utils.register_class(c)
        
    bpy.types.Object.my_global_enum = bpy.props.EnumProperty(
    items=(
        ('NONE', "None", ""),
        ('FLUID', "Fluid", ""),
        ('COLLIDER', "Collider", ""),
        ('EMMITER', "Emmiter", ""),
        ('TERMINATOR', "Terminator", ""),
        ('FORCE', "Force", "")
    ),
    default='NONE')
    
    bpy.types.Scene.fe_props = bpy.props.PointerProperty(type=FluidEngineProperties)

def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.fe_props
